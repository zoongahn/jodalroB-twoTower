"""
Two-Tower Predictor: 새로운 Notice에 대해 적합한 Company 추천

주요 기능:
1. 새 Notice의 bidntceno, bidntceord로 DB에서 데이터 로드
2. 기존 전처리 파이프라인 실행
3. Notice 임베딩 생성
4. Vector DB에서 Company 임베딩 로드
5. 유사도 계산 및 Top-K 추천
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from tqdm import tqdm

from sqlalchemy.engine import Engine
from src.prediction.model_loader import ModelLoader
from src.vectorize import EmbeddingStore
from preprocess.torchrec.feature_store import FeatureStore
from src.towers.kjt_utils import create_kjt_from_batch_gpu


class TwoTowerPredictor:
    """
    Two-Tower 모델을 사용한 Notice-Company 추천 예측기

    사용 예제:
        predictor = TwoTowerPredictor(
            checkpoint_path="output/models/20251023_055430/best_model.pt",
            vector_db_path="data/vectorize/embeddings",
            db_engine=engine
        )

        # 새 Notice에 대해 Top-10 Company 추천
        results = predictor.predict_for_notice(
            bidntceno="20230106038",
            bidntceord="000",
            top_k=10
        )
    """

    def __init__(
        self,
        checkpoint_path: str,
        db_engine: Engine,
        vector_db_path: Optional[str] = None,
        use_vector_db: bool = True,
        config_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로
            db_engine: 데이터베이스 엔진
            vector_db_path: Vector DB 경로 (use_vector_db=True일 때 필수)
            use_vector_db: Vector DB 사용 여부 (False면 DB에서 직접 계산)
            config_path: 모델 설정 파일 경로 (None이면 자동 탐색)
            device: 사용할 디바이스
        """
        self.db_engine = db_engine
        self.device = device
        self.use_vector_db = use_vector_db

        # 1. 모델 로더 초기화 및 로드
        print("=" * 80)
        print("모델 로딩 중...")
        print("=" * 80)
        self.model_loader = ModelLoader(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device
        )
        self.model_loader.load_model()
        self.model = self.model_loader.get_model()
        self.schema = self.model_loader.get_schema()
        self.config = self.model_loader.get_config()

        # 2. Preprocessor 초기화
        self.preprocessor = self.model_loader.init_preprocessor()

        # 3. Vector DB 로드 (선택적)
        self.vector_store = None
        if self.use_vector_db:
            if vector_db_path is None:
                raise ValueError("use_vector_db=True이지만 vector_db_path가 제공되지 않았습니다")

            print("\n" + "=" * 80)
            print("Vector DB 로딩 중...")
            print("=" * 80)
            self.vector_store = EmbeddingStore()
            self.vector_store.load(vector_db_path)
            print(f"✓ Vector DB 로드 완료")
            print(f"  - Notice: {len(self.vector_store.notice_ids):,}개")
            print(f"  - Company: {len(self.vector_store.company_ids):,}개")
        else:
            print("\n⚠️  Vector DB 미사용 모드: DB에서 직접 Company 임베딩 생성")

        print("\n✓ 예측기 초기화 완료")

    def _get_notice_title(self, bidntceno: str, bidntceord: str) -> Optional[str]:
        """
        DB로부터 공고명 조회

        Args:
            bidntceno: 공고번호
            bidntceord: 공고차수

        Returns:
            공고명 또는 None
        """
        query = f"""
        SELECT bidntcenm
        FROM public.notice
        WHERE bidntceno = '{bidntceno}' AND bidntceord = '{bidntceord}'
        LIMIT 1
        """
        try:
            df = pd.read_sql(query, self.db_engine)
            if len(df) > 0:
                return df.iloc[0]['bidntcenm']
        except Exception as e:
            print(f"⚠️  공고명 조회 실패: {e}")
        return None

    def _get_company_names(self, company_ids: List[str]) -> Dict[str, str]:
        """
        DB로부터 업체명 일괄 조회

        Args:
            company_ids: 사업자등록번호 리스트

        Returns:
            {bizno: company_name} 딕셔너리
        """
        if not company_ids:
            return {}

        # IN 절을 위한 포맷팅
        ids_str = "','".join(str(cid) for cid in company_ids)
        query = f"""
        SELECT bizno, corpnm
        FROM public.company
        WHERE bizno IN ('{ids_str}')
        """
        try:
            df = pd.read_sql(query, self.db_engine)
            return dict(zip(df['bizno'], df['corpnm']))
        except Exception as e:
            print(f"⚠️  업체명 조회 실패: {e}")
            return {}

    def _load_notice_from_db(
        self,
        bidntceno: str,
        bidntceord: str
    ) -> Optional[Dict]:
        """
        DB로부터 특정 Notice 데이터 로드 및 전처리

        Args:
            bidntceno: 공고번호
            bidntceord: 공고차수

        Returns:
            전처리된 Notice 데이터 (dense, categorical) 또는 None
        """
        # WHERE 조건 생성
        where_condition = f"bidntceno = '{bidntceno}' AND bidntceord = '{bidntceord}'"

        # FeatureStore로 데이터 로드
        notice_store = FeatureStore(
            engine=self.db_engine,
            side_schema=self.schema.notice,
            chunksize=1000,
            limit=1,
            where_condition=where_condition
        )
        notice_store.build(show_progress=False)

        # 데이터 존재 여부 확인
        if len(notice_store._key_to_row) == 0:
            return None

        # ID 추출
        notice_ids = list(notice_store._key_to_row.keys())
        notice_id = notice_ids[0]

        # Numeric 데이터
        if notice_store._num_mat is not None:
            numeric_data = notice_store._num_mat
            print(f"   [DEBUG] numeric_data from store: shape={numeric_data.shape}, has_nan={np.isnan(numeric_data).any()}")
            if np.isnan(numeric_data).any():
                nan_cols = np.where(np.isnan(numeric_data[0]))[0]
                print(f"   [DEBUG] NaN columns indices: {nan_cols.tolist()}")
                print(f"   [DEBUG] NaN values: {numeric_data[0][nan_cols]}")
        else:
            numeric_data = np.zeros((1, len(self.schema.notice.numeric)), dtype=np.float32)
            print(f"   [DEBUG] numeric_data initialized with zeros")

        # Text 임베딩 - 딕셔너리 형태로 준비
        text_dict = {}
        if notice_store._txt_mat is not None and len(notice_store._txt_mat) > 0:
            for col in self.schema.notice.text:
                if col in notice_store._txt_mat:
                    text_dict[col] = notice_store._txt_mat[col]
                else:
                    text_dict[col] = np.zeros((1, 768), dtype=np.float32)
        else:
            # 텍스트 컬럼이 없으면 빈 딕셔너리 또는 기본값
            for col in self.schema.notice.text:
                text_dict[col] = np.zeros((1, 768), dtype=np.float32)

        # Categorical 데이터
        if notice_store._cat_mat is not None:
            categorical_data = notice_store._cat_mat
        else:
            categorical_data = np.zeros((1, len(self.schema.notice.categorical)), dtype=np.int64)

        # Projection 적용 (numeric + text → dense_projected)
        numeric_tensor = torch.from_numpy(numeric_data).float().to(self.device)
        print(f"   [DEBUG] numeric_tensor: shape={numeric_tensor.shape}, has_nan={torch.isnan(numeric_tensor).any().item()}")

        text_dict_tensor = {
            col: torch.from_numpy(emb).float().to(self.device)
            for col, emb in text_dict.items()
        }

        for col, t in text_dict_tensor.items():
            print(f"   [DEBUG] text_dict_tensor[{col}]: shape={t.shape}, has_nan={torch.isnan(t).any().item()}")

        with torch.no_grad():
            dense_projected, text_projected = self.preprocessor.projectors['notice'](
                numeric_tensor, text_dict_tensor
            )

            # dense_projected 체크
            print(f"   [DEBUG] After projection - dense_projected: shape={dense_projected.shape}, has_nan={torch.isnan(dense_projected).any().item()}")

            # text_projected 체크
            if text_projected:
                for col in sorted(text_projected.keys()):
                    proj = text_projected[col]
                    print(f"   [DEBUG] After projection - text_projected[{col}]: shape={proj.shape}, has_nan={torch.isnan(proj).any().item()}")

            # dense_projected: (B, 128), text_projected: {col: (B, 128)}
            # 텍스트 projection들을 concat
            if text_projected:
                text_proj_list = [text_projected[col] for col in sorted(text_projected.keys())]
                text_proj_concat = torch.cat(text_proj_list, dim=1)  # (B, 128 * num_text_cols)
                print(f"   [DEBUG] text_proj_concat: shape={text_proj_concat.shape}, has_nan={torch.isnan(text_proj_concat).any().item()}")

                # dense와 text를 concat하여 최종 dense_projected
                final_dense = torch.cat([dense_projected, text_proj_concat], dim=1)
            else:
                final_dense = dense_projected

            print(f"   [DEBUG] final_dense: shape={final_dense.shape}, has_nan={torch.isnan(final_dense).any().item()}")
            dense_projected_np = final_dense.cpu().numpy()

        return {
            'id': notice_id,
            'dense_projected': dense_projected_np,
            'categorical': categorical_data
        }

    def _create_notice_input(
        self,
        notice_data: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        전처리된 Notice 데이터를 모델 입력 형식으로 변환

        Args:
            notice_data: _load_notice_from_db()의 출력

        Returns:
            모델 입력 형식 {"dense": Tensor, "kjt": KJT}
        """
        # Dense 데이터
        dense_tensor = torch.from_numpy(notice_data['dense_projected']).float().to(self.device)

        # Categorical 데이터 → KJT
        cat_tensor = torch.from_numpy(notice_data['categorical']).long().to(self.device)
        kjt = create_kjt_from_batch_gpu(cat_tensor, self.schema.notice.categorical, self.device)

        return {
            "dense": dense_tensor,
            "kjt": kjt
        }

    def _generate_all_company_embeddings(
        self,
        batch_size: int = 1024,
        limit: Optional[int] = None
    ) -> Tuple[List, np.ndarray]:
        """
        DB에서 모든 Company 데이터를 로드하고 임베딩 생성

        Args:
            batch_size: 배치 크기
            limit: 최대 Company 개수 (None이면 전체)

        Returns:
            Tuple of (company_ids, company_embeddings)
            - company_ids: List of company IDs (bizno)
            - company_embeddings: np.ndarray [N, embedding_dim]
        """
        from preprocess.torchrec.feature_store import build_feature_store

        print("\n  🔄 DB에서 Company 데이터 로딩 중...")
        company_store = build_feature_store(
            self.db_engine,
            self.schema.company,
            chunksize=5000,
            limit=limit,
            show_progress=False
        )

        company_ids_raw = company_store['ids']
        # company_ids가 tuple인 경우 첫 번째 요소만 추출
        company_ids = [cid[0] if isinstance(cid, tuple) else cid for cid in company_ids_raw]
        categorical = company_store['categorical']

        # Preprocessor로 dense projection 수행
        print(f"  🔄 피처 전처리 (projection) 중...")

        # Company만 전처리 (preprocessor의 company_projector 사용)
        np_numeric = company_store.get('numeric')
        np_text = company_store.get('text')

        n_samples = len(np_numeric) if np_numeric is not None else 0

        # 배치 단위로 projection 적용
        projected_batches = []

        with torch.no_grad():
            for start_idx in range(0, n_samples, self.preprocessor.batch_size):
                end_idx = min(start_idx + self.preprocessor.batch_size, n_samples)

                # Numeric projection
                batch_numeric = torch.from_numpy(np_numeric[start_idx:end_idx]).to(
                    self.preprocessor.device, dtype=torch.float32
                )
                projected_numeric = self.preprocessor.projectors['company'].num_proj(batch_numeric).cpu().numpy()

                # Text projection (있는 경우)
                if np_text is not None and len(np_text) > 0:
                    batch_text_dict = {k: torch.from_numpy(v[start_idx:end_idx]).to(
                        self.preprocessor.device, dtype=torch.float32
                    ) for k, v in np_text.items()}

                    # 각 텍스트 컬럼 projection 후 concatenate
                    text_projected_list = []
                    for col, text_tensor in batch_text_dict.items():
                        text_proj = self.preprocessor.projectors['company'].text_proj(text_tensor).cpu().numpy()
                        text_projected_list.append(text_proj)

                    if text_projected_list:
                        projected_text = np.concatenate(text_projected_list, axis=1)
                        # Concatenate with numeric
                        projected_batch = np.concatenate([projected_numeric, projected_text], axis=1)
                    else:
                        projected_batch = projected_numeric
                else:
                    projected_batch = projected_numeric

                projected_batches.append(projected_batch)

        dense_projected = np.concatenate(projected_batches, axis=0)

        print(f"  ✓ 피처 전처리 완료: dense_projected shape={dense_projected.shape}")

        num_companies = len(company_ids)
        embedding_dim = self.config['final_embedding_dim']

        print(f"  ✓ Company 데이터 로드 완료: {num_companies:,}개")
        print(f"  🔄 Company 임베딩 생성 중 (배치 크기: {batch_size})...")

        company_embeddings = np.zeros((num_companies, embedding_dim), dtype=np.float32)
        num_batches = (num_companies + batch_size - 1) // batch_size

        self.model.eval()
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="  Generating embeddings"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_companies)

                # 배치 데이터 준비
                batch_dense = torch.from_numpy(dense_projected[start_idx:end_idx]).float().to(self.device)
                batch_cat = torch.from_numpy(categorical[start_idx:end_idx]).long().to(self.device)

                # KJT 생성
                batch_kjt = create_kjt_from_batch_gpu(batch_cat, self.schema.company.categorical, self.device)

                # Company 입력 구성
                company_input = {
                    "dense": batch_dense,
                    "kjt": batch_kjt
                }

                # Company Tower forward
                batch_embeddings = self.model.company_tower(company_input)

                # CPU로 이동하여 저장
                company_embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()

        print(f"  ✓ Company 임베딩 생성 완료: {company_embeddings.shape}")

        return company_ids, company_embeddings

    def predict_for_notice(
        self,
        bidntceno: str,
        bidntceord: str,
        top_k: int = 10,
        min_similarity: float = None,
        return_embeddings: bool = False
    ) -> Dict:
        """
        새로운 Notice에 대해 적합한 Company 추천

        Args:
            bidntceno: 공고번호
            bidntceord: 공고차수
            top_k: 반환할 상위 K개 Company (None이면 제한 없음)
            min_similarity: 최소 유사도 threshold (None이면 제한 없음)
            return_embeddings: 임베딩도 함께 반환할지 여부

        Returns:
            Dict: {
                'notice_id': (bidntceno, bidntceord),
                'top_k_companies': List[Tuple[company_id, similarity_score]],
                'notice_embedding': Optional[np.ndarray],  # if return_embeddings
                'company_embeddings': Optional[Dict]  # if return_embeddings
            }
        """
        print(f"\n{'='*80}")
        print(f"예측 시작: Notice {bidntceno}-{bidntceord}")
        print(f"{'='*80}")

        notice_id = (bidntceno, bidntceord)

        # 1. Notice 임베딩 생성
        if self.use_vector_db and self.vector_store is not None:
            # Vector DB 사용
            print("1. Vector DB에서 Notice 임베딩 조회 중...")
            # Vector DB에 저장된 형식에 맞게 변환 (tuple → "bidntceno_bidntceord")
            notice_id_for_db = f"{bidntceno}_{bidntceord}"
            notice_embedding_np = self.vector_store.get_notice_embedding(notice_id_for_db)

            if notice_embedding_np is not None:
                print(f"✓ Vector DB에서 Notice 임베딩 로드 완료: {notice_embedding_np.shape}")
                notice_embedding_np = notice_embedding_np.reshape(1, -1)  # [1, embedding_dim]
            else:
                print("⚠️  Vector DB에 없음. 새로 생성합니다...")
                print("⚠️  경고: FeatureProjector가 학습되지 않아 결과가 부정확할 수 있습니다!")

                # DB에서 Notice 데이터 로드
                print("1-1. DB에서 Notice 데이터 로드 중...")
                notice_data = self._load_notice_from_db(bidntceno, bidntceord)

                if notice_data is None:
                    raise ValueError(f"Notice를 찾을 수 없습니다: {bidntceno}-{bidntceord}")

                print(f"✓ Notice 데이터 로드 완료: {notice_data['id']}")

                # Notice 임베딩 생성
                print("1-2. Notice 임베딩 생성 중...")
                notice_input = self._create_notice_input(notice_data)

                self.model.eval()
                with torch.no_grad():
                    notice_embedding = self.model.notice_tower(notice_input)
                    notice_embedding_np = notice_embedding.cpu().numpy()  # [1, embedding_dim]

                print(f"✓ Notice 임베딩 생성 완료: {notice_embedding_np.shape}")
        else:
            # Vector DB 미사용 - DB에서 직접 생성
            print("1. DB에서 Notice 데이터 로드 및 임베딩 생성 중...")
            notice_data = self._load_notice_from_db(bidntceno, bidntceord)

            if notice_data is None:
                raise ValueError(f"Notice를 찾을 수 없습니다: {bidntceno}-{bidntceord}")

            print(f"✓ Notice 데이터 로드 완료: {notice_data['id']}")

            # 디버깅: notice_data 체크
            print(f"   dense_projected: has_nan={np.isnan(notice_data['dense_projected']).any()}, shape={notice_data['dense_projected'].shape}")
            print(f"   categorical: has_nan={np.isnan(notice_data['categorical']).any()}, shape={notice_data['categorical'].shape}")

            # Notice 임베딩 생성
            notice_input = self._create_notice_input(notice_data)

            # 디버깅: notice_input 체크
            print(f"   notice_input dense: has_nan={torch.isnan(notice_input['dense']).any().item()}, shape={notice_input['dense'].shape}")

            self.model.eval()
            with torch.no_grad():
                notice_embedding = self.model.notice_tower(notice_input)
                notice_embedding_np = notice_embedding.cpu().numpy()  # [1, embedding_dim]

            print(f"✓ Notice 임베딩 생성 완료: {notice_embedding_np.shape}")

        # 2. Company 임베딩 준비
        if self.use_vector_db and self.vector_store is not None:
            # Vector DB에서 로드
            print("2. Vector DB에서 Company 임베딩 로드 중...")
            company_ids = self.vector_store.company_ids
            num_companies = len(company_ids)

            # 모든 Company 임베딩을 행렬로 구성
            company_embeddings_matrix = np.zeros(
                (num_companies, self.config['final_embedding_dim']),
                dtype=np.float32
            )

            for idx, company_id in enumerate(company_ids):
                emb = self.vector_store.get_company_embedding(company_id)
                if emb is not None:
                    company_embeddings_matrix[idx] = emb

            print(f"✓ Company 임베딩 로드 완료: {company_embeddings_matrix.shape}")
        else:
            # DB에서 직접 생성
            print("2. DB에서 Company 데이터 로드 및 임베딩 생성 중...")
            company_ids, company_embeddings_matrix = self._generate_all_company_embeddings(
                batch_size=1024,
                limit=None  # 전체 Company
            )

        # 3. 유사도 계산 (코사인 유사도)
        print("3. 유사도 계산 중...")

        # 디버깅: 임베딩 체크
        print(f"   Notice 임베딩: shape={notice_embedding_np.shape}, has_nan={np.isnan(notice_embedding_np).any()}, norm={np.linalg.norm(notice_embedding_np):.4f}")
        print(f"   Company 임베딩: shape={company_embeddings_matrix.shape}, has_nan={np.isnan(company_embeddings_matrix).any()}, mean_norm={np.linalg.norm(company_embeddings_matrix, axis=1).mean():.4f}")

        # L2 정규화
        notice_norm = notice_embedding_np / (np.linalg.norm(notice_embedding_np, axis=1, keepdims=True) + 1e-8)
        company_norms = company_embeddings_matrix / (np.linalg.norm(company_embeddings_matrix, axis=1, keepdims=True) + 1e-8)

        # 코사인 유사도 = 내적 (정규화된 벡터)
        similarities = np.dot(notice_norm, company_norms.T).squeeze()  # [num_companies]

        print(f"✓ 유사도 계산 완료: min={similarities.min():.4f}, max={similarities.max():.4f}")

        # 4. Top-K 추출 및 유사도 필터링
        filter_desc = []
        if top_k:
            filter_desc.append(f"Top-{top_k}")
        if min_similarity is not None:
            filter_desc.append(f"유사도 >= {min_similarity}")

        filter_str = " & ".join(filter_desc) if filter_desc else "모든"
        print(f"4. Company 추출 중 ({filter_str})...")

        # 유사도 기준 내림차순 정렬 (stable sort로 일관성 보장)
        sorted_indices = np.argsort(similarities, kind='stable')[::-1]

        # 필터링 로직: top_k개 중에서 min_similarity 이상만 반환
        filtered_indices = []

        for idx in sorted_indices:
            # top_k 제한 확인
            if top_k is not None and len(filtered_indices) >= top_k:
                break

            # min_similarity 조건 확인
            if min_similarity is not None and similarities[idx] < min_similarity:
                continue

            filtered_indices.append(idx)

        top_k_companies = [
            (company_ids[idx], float(similarities[idx]))
            for idx in filtered_indices
        ]

        print(f"✓ 추출 완료: {len(top_k_companies)}개 Company")
        if len(top_k_companies) > 0:
            print(f"  유사도 범위: {top_k_companies[-1][1]:.4f} ~ {top_k_companies[0][1]:.4f}")

        # 5. DB에서 공고명과 업체명 조회
        print("\n5. 공고명 및 업체명 조회 중...")
        notice_title = self._get_notice_title(bidntceno, bidntceord)

        # 업체명 일괄 조회
        company_id_list = [company_id for company_id, _ in top_k_companies]
        company_names = self._get_company_names(company_id_list)

        print(f"✓ 공고명 조회 완료: {'있음' if notice_title else '없음'}")
        print(f"✓ 업체명 조회 완료: {len(company_names)}/{len(company_id_list)}개")

        # 6. 결과 구성
        result = {
            'notice_id': notice_data['id'] if notice_embedding_np is not None and 'notice_data' in locals() else notice_id,
            'notice_title': notice_title,
            'top_k_companies': top_k_companies,
            'company_names': company_names  # {bizno: company_name} 딕셔너리
        }

        if return_embeddings:
            result['notice_embedding'] = notice_embedding_np.squeeze()
            result['company_embeddings'] = {
                company_ids[idx]: company_embeddings_matrix[idx]
                for idx in filtered_indices
            }

        print(f"\n{'='*80}")
        print("✓ 예측 완료")
        print(f"{'='*80}")

        return result

    def predict_for_notices_batch(
        self,
        notice_ids: List[Tuple[str, str]],
        top_k: int = 10,
        min_similarity: float = None,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        여러 Notice에 대해 배치 예측

        Args:
            notice_ids: [(bidntceno, bidntceord), ...] 리스트
            top_k: 각 Notice에 대해 반환할 상위 K개 (None이면 제한 없음)
            min_similarity: 최소 유사도 threshold (None이면 제한 없음)
            show_progress: 진행 상황 표시 여부

        Returns:
            예측 결과 리스트
        """
        results = []

        iterator = tqdm(notice_ids, desc="배치 예측") if show_progress else notice_ids

        for bidntceno, bidntceord in iterator:
            try:
                result = self.predict_for_notice(
                    bidntceno=bidntceno,
                    bidntceord=bidntceord,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    return_embeddings=False
                )
                results.append(result)
            except Exception as e:
                print(f"\n⚠️  Notice {bidntceno}-{bidntceord} 예측 실패: {e}")
                results.append({
                    'notice_id': (bidntceno, bidntceord),
                    'top_k_companies': [],
                    'error': str(e)
                })

        return results
