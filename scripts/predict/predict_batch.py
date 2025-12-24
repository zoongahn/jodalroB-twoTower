#!/usr/bin/env python3
"""
Predict Batch: N개 공고에 대한 효율적인 배치 예측

기존 predict_for_notice를 N번 호출하는 방식과 달리,
모든 임베딩을 한 번에 생성한 뒤 메모리에서 유사도 계산을 수행하여 효율성을 극대화합니다.

주요 흐름:
1. N개 공고 피처를 DB에서 로드 → 배치 임베딩 생성
2. 전체 회사 피처를 DB에서 로드 → 배치 임베딩 생성
3. 코사인 유사도 계산 (N x M 행렬 연산)
4. 각 공고별로 top-k (k=bid_count) 회사 추출

사용법:
    # CSV 파일 입력 (bidntceno, bidntceord, bid_count 컬럼 필요)
    python scripts/predict/predict_batch.py \
        --checkpoint output/models/best_model.pt \
        --input notices.csv \
        --output ./output_dir

    # CSV 형식:
    # bidntceno,bidntceord,bid_count
    # 20230106038,00,15
    # 20230106039,00,8

출력 파일:
    --output 디렉터리 지정 시 다음 파일들이 생성됨:
    - predictions.csv: 예측 결과 (공고-회사 쌍, 유사도)
    - notice_embeddings.csv: 공고 임베딩 벡터
    - company_embeddings.csv: 회사 임베딩 벡터
"""

import sys
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# Add project root to path (for database, src, preprocess modules)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from database.database_connector import DatabaseConnector
from src.prediction.embedding_service import EmbeddingService
from preprocess.torchrec.feature_store import FeatureStore


class BatchPredictor:
    """
    N개 공고에 대한 효율적인 배치 예측기

    모든 임베딩을 먼저 생성한 뒤 메모리에서 유사도 계산 수행
    """

    def __init__(
        self,
        checkpoint_path: str,
        db_engine,
        device: str = "cuda"
    ):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로
            db_engine: SQLAlchemy 엔진
            device: 디바이스 ("cuda" or "cpu")
        """
        self.device = device
        self.db_engine = db_engine

        # EmbeddingService 초기화 (모델, 스키마, preprocessor 포함)
        print("=" * 80)
        print("BatchPredictor 초기화")
        print("=" * 80)

        self.embedding_service = EmbeddingService(
            checkpoint_path=checkpoint_path,
            db_engine=db_engine,
            device=device
        )

        # 자주 사용하는 참조 저장
        self.model = self.embedding_service.model
        self.schema = self.embedding_service.schema
        self.config = self.embedding_service.config
        self.preprocessor = self.embedding_service.preprocessor

        # 회사 임베딩 캐시 (한 번 로드하면 재사용)
        self._company_ids: Optional[List[str]] = None
        self._company_embeddings: Optional[np.ndarray] = None

        # company_id → index 매핑 캐시
        self._company_id_to_idx: Optional[Dict[str, int]] = None

        print("✓ BatchPredictor 초기화 완료")

    def _get_eligible_companies_batch(
        self,
        notice_ids: List[Tuple[str, str]],
        chunk_size: int = 500
    ) -> Dict[Tuple[str, str], Optional[set]]:
        """
        N개 공고의 업종코드에 해당하는 참여가능 업체 목록 일괄 조회

        Args:
            notice_ids: [(bidntceno, bidntceord), ...] 리스트
            chunk_size: DB 쿼리당 공고 수 (기본: 500, PostgreSQL 스택 제한 회피)

        Returns:
            {(bidntceno, bidntceord): set(bizno) or None} 딕셔너리
            - None이면 해당 공고에 업종코드가 없음 (전체 업체 대상)
        """
        if not notice_ids:
            return {}

        result = {nid: None for nid in notice_ids}

        try:
            # 1. N개 공고의 업종코드 청크 단위로 조회
            notice_to_codes = {}
            num_chunks = (len(notice_ids) + chunk_size - 1) // chunk_size

            print(f"  1. 공고 업종코드 조회 중 ({len(notice_ids)}개, {num_chunks}개 청크)...")

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(notice_ids))
                chunk_ids = notice_ids[start_idx:end_idx]

                values = ", ".join([
                    f"('{ntce_no}', '{ntce_ord}')"
                    for ntce_no, ntce_ord in chunk_ids
                ])
                industry_query = f"""
                SELECT bidntceno, bidntceord, lcnslmtnm_code
                FROM public.notice_industry_type
                WHERE (bidntceno, bidntceord) IN ({values})
                """
                chunk_df = pd.read_sql(industry_query, self.db_engine)

                for _, row in chunk_df.iterrows():
                    key = (row['bidntceno'], row['bidntceord'])
                    if key not in notice_to_codes:
                        notice_to_codes[key] = []
                    notice_to_codes[key].append(row['lcnslmtnm_code'])

            if len(notice_to_codes) == 0:
                print(f"  ⚠️  {len(notice_ids)}개 공고 모두 업종코드 없음. 전체 업체 대상으로 예측합니다.")
                return result

            # 업종코드가 있는 공고 수
            notices_with_codes = len(notice_to_codes)
            notices_without_codes = len(notice_ids) - notices_with_codes
            print(f"  ✓ 업종코드 조회: {notices_with_codes}개 공고에 업종코드 있음, {notices_without_codes}개 없음")

            # 2. 모든 고유 업종코드에 해당하는 업체 일괄 조회
            all_codes = set()
            for codes in notice_to_codes.values():
                all_codes.update(codes)

            if not all_codes:
                return result

            print(f"  2. 업종별 업체 조회 중 ({len(all_codes)}개 업종코드)...")

            codes_str = "','".join(str(code) for code in all_codes)
            company_query = f"""
            SELECT DISTINCT bizno, indstrytycd
            FROM public.company_industry_type
            WHERE indstrytycd IN ('{codes_str}')
            """
            company_df = pd.read_sql(company_query, self.db_engine)

            # 업종코드별 업체 목록 생성
            code_to_companies = {}
            for _, row in company_df.iterrows():
                code = row['indstrytycd']
                if code not in code_to_companies:
                    code_to_companies[code] = set()
                code_to_companies[code].add(row['bizno'])

            print(f"  ✓ 업종별 업체 조회: {len(all_codes)}개 업종코드, {len(company_df)}개 업체-업종 매핑")

            # 3. 각 공고별 참여가능 업체 set 생성
            for notice_id, codes in notice_to_codes.items():
                eligible = set()
                for code in codes:
                    if code in code_to_companies:
                        eligible.update(code_to_companies[code])
                if eligible:
                    result[notice_id] = eligible

            # 통계 출력
            eligible_counts = [len(v) for v in result.values() if v is not None]
            if eligible_counts:
                print(f"  ✓ 참여가능 업체 수: 평균 {np.mean(eligible_counts):,.0f}개, "
                      f"최소 {min(eligible_counts):,}개, 최대 {max(eligible_counts):,}개")

            return result

        except Exception as e:
            print(f"  ⚠️  업종 기반 업체 조회 실패: {e}. 전체 업체 대상으로 예측합니다.")
            return result

    def load_all_company_embeddings(self, batch_size: int = 1024) -> Tuple[List[str], np.ndarray]:
        """
        전체 회사 피처를 DB에서 로드하여 임베딩 생성

        Returns:
            Tuple of (company_ids, company_embeddings)
            - company_ids: List[str] - 사업자등록번호 리스트
            - company_embeddings: np.ndarray [M, embedding_dim]
        """
        if self._company_ids is not None and self._company_embeddings is not None:
            print("✓ 캐시된 회사 임베딩 사용")
            return self._company_ids, self._company_embeddings

        print("\n" + "-" * 60)
        print("STEP: 전체 회사 임베딩 생성")
        print("-" * 60)

        start_time = time.time()

        # 1. DB에서 전체 회사 피처 로드
        print("1. DB에서 회사 피처 로드 중...")
        company_store = FeatureStore(
            engine=self.db_engine,
            side_schema=self.schema.company,
            chunksize=10000,
            limit=None,  # 전체
            where_condition=None
        )
        company_store.build(show_progress=True)

        num_companies = len(company_store._key_to_row)
        print(f"   ✓ 회사 수: {num_companies:,}개")

        # 2. 피처 추출 (projection 적용)
        print("2. 피처 전처리 (projection) 중...")
        company_data = self.embedding_service._extract_company_features(company_store)

        # 3. 임베딩 생성
        print(f"3. 회사 임베딩 생성 중 (배치 크기: {batch_size})...")

        dense_projected = company_data['dense_projected']
        categorical = company_data['categorical']

        embedding_dim = self.config['final_embedding_dim']
        company_embeddings = np.zeros((num_companies, embedding_dim), dtype=np.float32)

        num_batches = (num_companies + batch_size - 1) // batch_size

        self.model.eval()
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="   회사 임베딩"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_companies)

                # 배치 데이터 준비
                batch_dense = torch.from_numpy(
                    dense_projected[start_idx:end_idx]
                ).float().to(self.device)

                batch_cat = torch.from_numpy(
                    categorical[start_idx:end_idx]
                ).long().to(self.device)

                # KJT 생성
                from src.towers.kjt_utils import create_kjt_from_batch_gpu
                batch_kjt = create_kjt_from_batch_gpu(
                    batch_cat,
                    self.schema.company.categorical,
                    self.device
                )

                # Company Tower forward
                company_input = {"dense": batch_dense, "kjt": batch_kjt}
                batch_embeddings = self.model.company_tower(company_input)

                company_embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()

        # company_ids 추출 (tuple 형태면 첫 번째 요소)
        company_ids = [
            key[0] if isinstance(key, tuple) else key
            for key in company_store._key_to_row.keys()
        ]

        elapsed = time.time() - start_time
        print(f"   ✓ 회사 임베딩 완료: shape={company_embeddings.shape}, 소요시간={elapsed:.1f}초")

        # 캐시 저장
        self._company_ids = company_ids
        self._company_embeddings = company_embeddings
        self._company_id_to_idx = {cid: idx for idx, cid in enumerate(company_ids)}

        return company_ids, company_embeddings

    def _build_in_clause_condition(
        self,
        notice_ids: List[Tuple[str, str]],
        chunk_size: int = 1000
    ) -> str:
        """
        IN 절을 사용한 효율적인 WHERE 조건 생성

        대량의 공고 ID를 처리할 때 OR 조건 대신 IN 절 사용
        PostgreSQL은 (col1, col2) IN ((v1, v2), ...) 형태 지원

        Args:
            notice_ids: [(bidntceno, bidntceord), ...] 리스트
            chunk_size: 청크당 공고 수 (기본: 1000)

        Returns:
            WHERE 조건 문자열
        """
        if len(notice_ids) <= chunk_size:
            # 소량일 경우 단일 IN 절
            values = ", ".join([
                f"('{ntce_no}', '{ntce_ord}')"
                for ntce_no, ntce_ord in notice_ids
            ])
            return f"(bidntceno, bidntceord) IN ({values})"
        else:
            # 대량일 경우 여러 IN 절을 OR로 연결
            conditions = []
            for i in range(0, len(notice_ids), chunk_size):
                chunk = notice_ids[i:i + chunk_size]
                values = ", ".join([
                    f"('{ntce_no}', '{ntce_ord}')"
                    for ntce_no, ntce_ord in chunk
                ])
                conditions.append(f"(bidntceno, bidntceord) IN ({values})")
            return " OR ".join(conditions)

    def load_notice_embeddings(
        self,
        notice_ids: List[Tuple[str, str]],
        batch_size: int = 512,
        db_chunk_size: int = 1000
    ) -> Tuple[List[Tuple[str, str]], np.ndarray]:
        """
        N개 공고 피처를 DB에서 로드하여 임베딩 생성

        Args:
            notice_ids: [(bidntceno, bidntceord), ...] 리스트
            batch_size: 임베딩 생성 배치 크기
            db_chunk_size: DB 쿼리당 공고 수 (기본: 1000)

        Returns:
            Tuple of (valid_notice_ids, notice_embeddings)
            - valid_notice_ids: 실제로 DB에 존재하는 공고 ID 리스트
            - notice_embeddings: np.ndarray [N', embedding_dim]
        """
        print("\n" + "-" * 60)
        print(f"STEP: {len(notice_ids)}개 공고 임베딩 생성")
        print("-" * 60)

        start_time = time.time()
        total_notices = len(notice_ids)

        # 대량 공고 처리: 청크 단위로 DB 조회 후 병합
        if total_notices > db_chunk_size:
            print(f"1. DB에서 공고 피처 로드 중 (청크 크기: {db_chunk_size})...")

            all_key_to_row = {}
            all_num_buf = []
            all_cat_buf = []
            all_txt_buf = {}

            num_chunks = (total_notices + db_chunk_size - 1) // db_chunk_size
            current_row_idx = 0

            for chunk_idx in tqdm(range(num_chunks), desc="   DB 청크 로드"):
                chunk_start = chunk_idx * db_chunk_size
                chunk_end = min(chunk_start + db_chunk_size, total_notices)
                chunk_ids = notice_ids[chunk_start:chunk_end]

                # IN 절 사용한 효율적인 조건 생성
                condition = self._build_in_clause_condition(chunk_ids, chunk_size=db_chunk_size)

                notice_store = FeatureStore(
                    engine=self.db_engine,
                    side_schema=self.schema.notice,
                    chunksize=10000,
                    limit=None,
                    where_condition=condition
                )
                notice_store.build(show_progress=False)

                # 결과 병합
                for key, row_idx in notice_store._key_to_row.items():
                    all_key_to_row[key] = current_row_idx + row_idx

                if notice_store._num_mat is not None:
                    all_num_buf.append(notice_store._num_mat)
                if notice_store._cat_mat is not None:
                    all_cat_buf.append(notice_store._cat_mat)
                if notice_store._txt_mat is not None:
                    for col, mat in notice_store._txt_mat.items():
                        if col not in all_txt_buf:
                            all_txt_buf[col] = []
                        all_txt_buf[col].append(mat)

                current_row_idx += len(notice_store._key_to_row)

            # 병합된 FeatureStore 생성
            notice_store = FeatureStore(
                engine=self.db_engine,
                side_schema=self.schema.notice,
                chunksize=10000,
                limit=None,
                where_condition=None
            )
            notice_store._key_to_row = all_key_to_row
            if all_num_buf:
                notice_store._num_mat = np.vstack(all_num_buf)
            if all_cat_buf:
                notice_store._cat_mat = np.vstack(all_cat_buf)
            if all_txt_buf:
                notice_store._txt_mat = {
                    col: np.vstack(mats) for col, mats in all_txt_buf.items()
                }
        else:
            # 소량: 기존 방식 (IN 절 사용)
            print("1. DB에서 공고 피처 로드 중...")
            condition = self._build_in_clause_condition(notice_ids)

            notice_store = FeatureStore(
                engine=self.db_engine,
                side_schema=self.schema.notice,
                chunksize=10000,
                limit=None,
                where_condition=condition
            )
            notice_store.build(show_progress=True)

        num_notices = len(notice_store._key_to_row)
        print(f"   ✓ 공고 수: {num_notices:,}개 (요청: {len(notice_ids)}개)")

        if num_notices == 0:
            return [], np.array([])

        # 2. 피처 추출 (projection 적용)
        print("2. 피처 전처리 (projection) 중...")
        notice_data = self.embedding_service._extract_notice_features(notice_store)

        # 3. 임베딩 생성
        print(f"3. 공고 임베딩 생성 중 (배치 크기: {batch_size})...")

        dense_projected = notice_data['dense_projected']
        categorical = notice_data['categorical']

        embedding_dim = self.config['final_embedding_dim']
        notice_embeddings = np.zeros((num_notices, embedding_dim), dtype=np.float32)

        num_batches = (num_notices + batch_size - 1) // batch_size

        self.model.eval()
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="   공고 임베딩"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_notices)

                batch_dense = torch.from_numpy(
                    dense_projected[start_idx:end_idx]
                ).float().to(self.device)

                batch_cat = torch.from_numpy(
                    categorical[start_idx:end_idx]
                ).long().to(self.device)

                from src.towers.kjt_utils import create_kjt_from_batch_gpu
                batch_kjt = create_kjt_from_batch_gpu(
                    batch_cat,
                    self.schema.notice.categorical,
                    self.device
                )

                notice_input = {"dense": batch_dense, "kjt": batch_kjt}
                batch_embeddings = self.model.notice_tower(notice_input)

                notice_embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()

        # valid_notice_ids 추출
        valid_notice_ids = list(notice_store._key_to_row.keys())

        elapsed = time.time() - start_time
        print(f"   ✓ 공고 임베딩 완료: shape={notice_embeddings.shape}, 소요시간={elapsed:.1f}초")

        return valid_notice_ids, notice_embeddings

    def predict_batch(
        self,
        notices: List[Dict],  # [{"bidntceno": str, "bidntceord": str, "bid_count": int}, ...]
        company_batch_size: int = 1024,
        notice_batch_size: int = 512,
        db_chunk_size: int = 1000,
        use_industry_filter: bool = True
    ) -> Dict:
        """
        N개 공고에 대해 효율적인 배치 예측 수행

        Args:
            notices: 공고 정보 리스트 [{"bidntceno", "bidntceord", "bid_count"}, ...]
            company_batch_size: 회사 임베딩 생성 배치 크기
            notice_batch_size: 공고 임베딩 생성 배치 크기
            db_chunk_size: DB 쿼리당 공고 수 (대량 처리시 청크 단위)
            use_industry_filter: 업종 필터 사용 여부 (기본값: True)
                - True: 공고의 업종코드에 해당하는 업체만 대상으로 예측
                - False: 전체 업체 대상으로 예측

        Returns:
            Dict: {
                "predictions": List[Dict] - 예측 결과,
                "notice_ids": List[Tuple] - 공고 ID 리스트,
                "notice_embeddings": np.ndarray - 공고 임베딩 [N, dim],
                "company_ids": List[str] - 회사 ID 리스트,
                "company_embeddings": np.ndarray - 회사 임베딩 [M, dim]
            }
        """
        print("\n" + "=" * 80)
        print(f"배치 예측 시작: {len(notices)}개 공고")
        print(f"업종 필터: {'사용' if use_industry_filter else '미사용'}")
        print("=" * 80)

        total_start = time.time()

        # 1. 전체 회사 임베딩 로드/생성
        company_ids, company_embeddings = self.load_all_company_embeddings(
            batch_size=company_batch_size
        )

        # L2 정규화 (코사인 유사도용)
        company_norms = np.linalg.norm(company_embeddings, axis=1, keepdims=True) + 1e-8
        company_embeddings_normalized = company_embeddings / company_norms

        # company_id → index 매핑
        company_id_to_idx = {cid: idx for idx, cid in enumerate(company_ids)}

        # 2. N개 공고 임베딩 로드/생성
        notice_ids = [(n["bidntceno"], n["bidntceord"]) for n in notices]
        valid_notice_ids, notice_embeddings = self.load_notice_embeddings(
            notice_ids=notice_ids,
            batch_size=notice_batch_size,
            db_chunk_size=db_chunk_size
        )

        if len(valid_notice_ids) == 0:
            print("⚠️  유효한 공고가 없습니다.")
            return {
                "predictions": [],
                "notice_ids": [],
                "notice_embeddings": np.array([]),
                "company_ids": company_ids,
                "company_embeddings": company_embeddings
            }

        # L2 정규화
        notice_norms = np.linalg.norm(notice_embeddings, axis=1, keepdims=True) + 1e-8
        notice_embeddings_normalized = notice_embeddings / notice_norms

        # valid_notice_id → index 매핑
        notice_id_to_idx = {nid: idx for idx, nid in enumerate(valid_notice_ids)}

        # 입력 notices에서 bid_count 매핑 생성
        bid_count_map = {
            (n["bidntceno"], n["bidntceord"]): n["bid_count"]
            for n in notices
        }

        # 3. 업종 필터링 정보 조회 (선택적)
        eligible_companies_map = {}
        if use_industry_filter:
            print("\n" + "-" * 60)
            print("STEP: 업종 기반 참여가능 업체 조회")
            print("-" * 60)
            eligible_companies_map = self._get_eligible_companies_batch(valid_notice_ids)

        # 4. 코사인 유사도 계산 (N x M 행렬 연산)
        print("\n" + "-" * 60)
        print("STEP: 유사도 계산 및 Top-K 추출")
        print("-" * 60)

        print("1. 유사도 행렬 계산 중...")
        sim_start = time.time()

        # [N, embedding_dim] @ [embedding_dim, M] = [N, M]
        similarity_matrix = np.dot(
            notice_embeddings_normalized,
            company_embeddings_normalized.T
        )

        sim_elapsed = time.time() - sim_start
        print(f"   ✓ 유사도 행렬: shape={similarity_matrix.shape}, 소요시간={sim_elapsed:.2f}초")

        # 5. 각 공고별 Top-K 추출 (업종 필터링 적용)
        print("2. Top-K 회사 추출 중...")
        if use_industry_filter:
            print("   (업종 필터 적용)")
        predictions = []

        for notice_id in tqdm(valid_notice_ids, desc="   Top-K 추출"):
            notice_idx = notice_id_to_idx[notice_id]
            similarities = similarity_matrix[notice_idx]

            # bid_count 가져오기
            k = bid_count_map.get(notice_id, 10)  # 기본값 10

            # 업종 필터링 적용
            eligible_set = eligible_companies_map.get(notice_id) if use_industry_filter else None

            if eligible_set is not None:
                # 참여가능 업체만 필터링하여 Top-K 추출
                # eligible_set에 있는 업체의 인덱스만 추출
                eligible_indices = [
                    self._company_id_to_idx[cid]
                    for cid in eligible_set
                    if cid in self._company_id_to_idx
                ]

                if len(eligible_indices) == 0:
                    # 참여가능 업체가 없으면 빈 결과
                    top_k_companies = []
                else:
                    eligible_indices = np.array(eligible_indices)
                    eligible_similarities = similarities[eligible_indices]

                    # Top-K 추출 (eligible 범위 내에서)
                    actual_k = min(k, len(eligible_indices))
                    if actual_k >= len(eligible_indices):
                        top_k_local_indices = np.argsort(eligible_similarities)[::-1]
                    else:
                        top_k_local_indices = np.argpartition(eligible_similarities, -actual_k)[-actual_k:]
                        top_k_local_indices = top_k_local_indices[
                            np.argsort(eligible_similarities[top_k_local_indices])[::-1]
                        ]

                    top_k_companies = [
                        (company_ids[eligible_indices[local_idx]], float(eligible_similarities[local_idx]))
                        for local_idx in top_k_local_indices
                    ]
            else:
                # 업종 필터 미적용 또는 업종코드 없음 → 전체 대상
                if k >= len(company_ids):
                    top_k_indices = np.argsort(similarities)[::-1]
                else:
                    # argpartition으로 효율적으로 top-k 추출
                    top_k_indices = np.argpartition(similarities, -k)[-k:]
                    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

                top_k_companies = [
                    (company_ids[idx], float(similarities[idx]))
                    for idx in top_k_indices
                ]

            predictions.append({
                "notice_id": notice_id,
                "bidntceno": notice_id[0],
                "bidntceord": notice_id[1],
                "bid_count": k,
                "top_k_companies": top_k_companies,
                "industry_filtered": eligible_set is not None
            })

        total_elapsed = time.time() - total_start
        print(f"\n✓ 배치 예측 완료: {len(predictions)}개 공고, 총 소요시간={total_elapsed:.1f}초")

        return {
            "predictions": predictions,
            "notice_ids": valid_notice_ids,
            "notice_embeddings": notice_embeddings,
            "company_ids": company_ids,
            "company_embeddings": company_embeddings
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="N개 공고에 대한 효율적인 배치 예측",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="모델 체크포인트 경로"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 CSV 파일 (bidntceno, bidntceord, bid_count 컬럼)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 디렉터리 경로 (predictions.csv, notice_embeddings.csv, company_embeddings.csv 생성)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="디바이스 (기본: cuda)"
    )

    parser.add_argument(
        "--company-batch-size",
        type=int,
        default=1024,
        help="회사 임베딩 생성 배치 크기 (기본: 1024)"
    )

    parser.add_argument(
        "--notice-batch-size",
        type=int,
        default=512,
        help="공고 임베딩 생성 배치 크기 (기본: 512)"
    )

    parser.add_argument(
        "--db-chunk-size",
        type=int,
        default=1000,
        help="DB 쿼리당 공고 수 (대량 처리시 청크 단위, 기본: 1000)"
    )

    parser.add_argument(
        "--no-industry-filter",
        action="store_true",
        help="업종 필터 미사용. 전체 업체 대상으로 예측 (기본값: 업종 필터 사용)"
    )

    return parser.parse_args()


def save_embeddings_to_csv(
    notice_ids: List[Tuple[str, str]],
    notice_embeddings: np.ndarray,
    company_ids: List[str],
    company_embeddings: np.ndarray,
    output_dir: Path
):
    """
    공고 및 회사 임베딩을 CSV로 저장

    Args:
        notice_ids: 공고 ID 리스트 [(bidntceno, bidntceord), ...]
        notice_embeddings: 공고 임베딩 [N, dim]
        company_ids: 회사 ID 리스트
        company_embeddings: 회사 임베딩 [M, dim]
        output_dir: 출력 디렉터리
    """
    embedding_dim = notice_embeddings.shape[1] if len(notice_embeddings) > 0 else company_embeddings.shape[1]

    # 공고 임베딩 저장
    if len(notice_ids) > 0:
        notice_emb_path = output_dir / "notice_embeddings.csv"
        print(f"   공고 임베딩 저장 중: {notice_emb_path}")

        # 컬럼명: bidntceno, bidntceord, emb_0, emb_1, ..., emb_{dim-1}
        emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
        notice_rows = []
        for idx, (bidntceno, bidntceord) in enumerate(notice_ids):
            row = {
                "bidntceno": bidntceno,
                "bidntceord": bidntceord
            }
            for i, col in enumerate(emb_cols):
                row[col] = notice_embeddings[idx, i]
            notice_rows.append(row)

        notice_emb_df = pd.DataFrame(notice_rows)
        notice_emb_df.to_csv(notice_emb_path, index=False)
        print(f"   ✓ 공고 임베딩 저장 완료: {len(notice_ids)}개, {embedding_dim}차원")

    # 회사 임베딩 저장
    if len(company_ids) > 0:
        company_emb_path = output_dir / "company_embeddings.csv"
        print(f"   회사 임베딩 저장 중: {company_emb_path}")

        # 컬럼명: bizno, emb_0, emb_1, ..., emb_{dim-1}
        emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
        company_rows = []
        for idx, bizno in enumerate(company_ids):
            row = {"bizno": bizno}
            for i, col in enumerate(emb_cols):
                row[col] = company_embeddings[idx, i]
            company_rows.append(row)

        company_emb_df = pd.DataFrame(company_rows)
        company_emb_df.to_csv(company_emb_path, index=False)
        print(f"   ✓ 회사 임베딩 저장 완료: {len(company_ids)}개, {embedding_dim}차원")


def main():
    args = parse_args()

    # 입력 파일 로드
    print(f"\n입력 파일 로드: {args.input}")
    df = pd.read_csv(args.input, dtype=str)

    # 필수 컬럼 체크
    required_cols = ["bidntceno", "bidntceord", "bid_count"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    # bid_count를 int로 변환
    df["bid_count"] = df["bid_count"].astype(int)

    notices = df.to_dict("records")
    print(f"✓ {len(notices)}개 공고 로드 완료")

    # 출력 디렉터리 설정
    if args.output:
        output_dir = Path(args.output)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"predictions_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉터리: {output_dir}")

    # DB 연결
    db_connector = DatabaseConnector()

    try:
        # BatchPredictor 초기화
        predictor = BatchPredictor(
            checkpoint_path=args.checkpoint,
            db_engine=db_connector.engine,
            device=args.device
        )

        # 업종 필터 옵션 처리
        use_industry_filter = not args.no_industry_filter

        # 배치 예측 수행
        result = predictor.predict_batch(
            notices=notices,
            company_batch_size=args.company_batch_size,
            notice_batch_size=args.notice_batch_size,
            db_chunk_size=args.db_chunk_size,
            use_industry_filter=use_industry_filter
        )

        predictions = result["predictions"]
        notice_ids = result["notice_ids"]
        notice_embeddings = result["notice_embeddings"]
        company_ids = result["company_ids"]
        company_embeddings = result["company_embeddings"]

        # 1. 예측 결과 CSV 저장
        print("\n" + "-" * 60)
        print("STEP: 결과 파일 저장")
        print("-" * 60)

        predictions_path = output_dir / "predictions.csv"
        print(f"   예측 결과 저장 중: {predictions_path}")

        rows = []
        for pred in predictions:
            bidntceno = pred["bidntceno"]
            bidntceord = pred["bidntceord"]
            bid_count = pred["bid_count"]
            industry_filtered = pred.get("industry_filtered", False)

            for rank, (company_id, similarity) in enumerate(pred["top_k_companies"], 1):
                rows.append({
                    "bidntceno": bidntceno,
                    "bidntceord": bidntceord,
                    "bid_count": bid_count,
                    "rank": rank,
                    "bizno": company_id,
                    "similarity": similarity,
                    "industry_filtered": industry_filtered
                })

        result_df = pd.DataFrame(rows)
        result_df.to_csv(predictions_path, index=False)
        print(f"   ✓ 예측 결과 저장 완료: {len(result_df):,}개 행 (공고-회사 쌍)")

        # 2. 임베딩 CSV 저장
        save_embeddings_to_csv(
            notice_ids=notice_ids,
            notice_embeddings=notice_embeddings,
            company_ids=company_ids,
            company_embeddings=company_embeddings,
            output_dir=output_dir
        )

        print("\n" + "=" * 60)
        print("✓ 모든 파일 저장 완료")
        print("=" * 60)
        print(f"  - {output_dir}/predictions.csv")
        print(f"  - {output_dir}/notice_embeddings.csv")
        print(f"  - {output_dir}/company_embeddings.csv")

    finally:
        db_connector.close()


if __name__ == "__main__":
    main()
