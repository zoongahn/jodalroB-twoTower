"""
Embedding Service: 단일 공고/업체 ID로부터 임베딩 벡터 추출

사용 예제:
    from src.prediction.embedding_service import EmbeddingService

    service = EmbeddingService(
        checkpoint_path="output/models/best_model.pt",
        db_engine=engine
    )

    # 공고 임베딩 추출
    notice_emb = service.get_notice_embedding("20230106038", "000")

    # 업체 임베딩 추출
    company_emb = service.get_company_embedding("1234567890")
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from sqlalchemy.engine import Engine

from src.prediction.model_loader import ModelLoader
from preprocess.torchrec.feature_store import FeatureStore, build_feature_store_with_condition
from src.towers.kjt_utils import create_kjt_from_batch_gpu


class EmbeddingService:
    """
    단일 공고/업체 ID로부터 임베딩 벡터를 추출하는 서비스

    step1.notice_preprocessed, step1.company_preprocessed 테이블에서
    데이터를 조회하여 모델을 통해 임베딩 벡터로 변환합니다.
    """

    def __init__(
        self,
        checkpoint_path: str,
        db_engine: Engine,
        config_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로
            db_engine: 데이터베이스 엔진
            config_path: 모델 설정 파일 경로 (None이면 자동 탐색)
            device: 사용할 디바이스 ("cuda" 또는 "cpu")
        """
        self.db_engine = db_engine
        self.device = device

        # 모델 로더 초기화
        print("=" * 80)
        print("EmbeddingService 초기화 중...")
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

        # Preprocessor 초기화
        self.preprocessor = self.model_loader.init_preprocessor()

        print("\n✓ EmbeddingService 초기화 완료")
        print(f"  - Embedding dim: {self.config['final_embedding_dim']}")

    def get_notice_embedding(
        self,
        bidntceno: str,
        bidntceord: str = "00"
    ) -> Optional[np.ndarray]:
        """
        공고번호로부터 임베딩 벡터 추출

        Args:
            bidntceno: 공고번호
            bidntceord: 공고차수 (기본값: "00")

        Returns:
            np.ndarray: 임베딩 벡터 [embedding_dim] 또는 None (데이터 없음)
        """
        # 1. step1.notice_preprocessed에서 데이터 로드
        where_condition = f"bidntceno = '{bidntceno}' AND bidntceord = '{bidntceord}'"

        store = FeatureStore(
            engine=self.db_engine,
            side_schema=self.schema.notice,
            chunksize=1000,
            limit=1,
            where_condition=where_condition
        )
        store.build(show_progress=False)

        # 데이터 존재 여부 확인
        if len(store._key_to_row) == 0:
            print(f"⚠️  공고를 찾을 수 없습니다: {bidntceno}-{bidntceord}")
            return None

        # 2. Feature 추출
        notice_data = self._extract_notice_features(store)

        # 3. 모델 입력 생성 및 임베딩 추출
        notice_input = self._create_notice_input(notice_data)

        self.model.eval()
        with torch.no_grad():
            embedding = self.model.notice_tower(notice_input)
            embedding_np = embedding.cpu().numpy().squeeze()  # [embedding_dim]

        return embedding_np

    def get_company_embedding(
        self,
        bizno: str
    ) -> Optional[np.ndarray]:
        """
        사업자등록번호로부터 임베딩 벡터 추출

        Args:
            bizno: 사업자등록번호

        Returns:
            np.ndarray: 임베딩 벡터 [embedding_dim] 또는 None (데이터 없음)
        """
        # 1. step1.company_preprocessed에서 데이터 로드
        where_condition = f"bizno = '{bizno}'"

        store = FeatureStore(
            engine=self.db_engine,
            side_schema=self.schema.company,
            chunksize=1000,
            limit=1,
            where_condition=where_condition
        )
        store.build(show_progress=False)

        # 데이터 존재 여부 확인
        if len(store._key_to_row) == 0:
            print(f"⚠️  업체를 찾을 수 없습니다: {bizno}")
            return None

        # 2. Feature 추출
        company_data = self._extract_company_features(store)

        # 3. 모델 입력 생성 및 임베딩 추출
        company_input = self._create_company_input(company_data)

        self.model.eval()
        with torch.no_grad():
            embedding = self.model.company_tower(company_input)
            embedding_np = embedding.cpu().numpy().squeeze()  # [embedding_dim]

        return embedding_np

    def get_notice_embeddings_batch(
        self,
        notice_ids: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], np.ndarray]:
        """
        여러 공고의 임베딩을 배치로 추출

        Args:
            notice_ids: [(bidntceno, bidntceord), ...] 리스트

        Returns:
            Tuple of (valid_ids, embeddings)
            - valid_ids: 실제로 존재하는 공고 ID 리스트
            - embeddings: np.ndarray [N, embedding_dim]
        """
        if not notice_ids:
            return [], np.array([])

        # WHERE IN 절 생성
        conditions = " OR ".join([
            f"(bidntceno = '{ntce_no}' AND bidntceord = '{ntce_ord}')"
            for ntce_no, ntce_ord in notice_ids
        ])

        store = FeatureStore(
            engine=self.db_engine,
            side_schema=self.schema.notice,
            chunksize=10000,
            limit=None,
            where_condition=conditions
        )
        store.build(show_progress=False)

        if len(store._key_to_row) == 0:
            return [], np.array([])

        # Feature 추출
        notice_data = self._extract_notice_features(store)

        # 모델 입력 생성
        notice_input = self._create_notice_input(notice_data)

        # 임베딩 추출
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.notice_tower(notice_input)
            embeddings_np = embeddings.cpu().numpy()  # [N, embedding_dim]

        valid_ids = list(store._key_to_row.keys())
        return valid_ids, embeddings_np

    def get_company_embeddings_batch(
        self,
        biznos: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        """
        여러 업체의 임베딩을 배치로 추출

        Args:
            biznos: 사업자등록번호 리스트

        Returns:
            Tuple of (valid_ids, embeddings)
            - valid_ids: 실제로 존재하는 업체 ID 리스트
            - embeddings: np.ndarray [N, embedding_dim]
        """
        if not biznos:
            return [], np.array([])

        # WHERE IN 절 생성
        ids_str = "','".join(biznos)
        where_condition = f"bizno IN ('{ids_str}')"

        store = FeatureStore(
            engine=self.db_engine,
            side_schema=self.schema.company,
            chunksize=10000,
            limit=None,
            where_condition=where_condition
        )
        store.build(show_progress=False)

        if len(store._key_to_row) == 0:
            return [], np.array([])

        # Feature 추출
        company_data = self._extract_company_features(store)

        # 모델 입력 생성
        company_input = self._create_company_input(company_data)

        # 임베딩 추출
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.company_tower(company_input)
            embeddings_np = embeddings.cpu().numpy()  # [N, embedding_dim]

        # company key는 tuple 형태 (bizno,)로 저장되므로 첫 번째 요소 추출
        valid_ids = [key[0] if isinstance(key, tuple) else key for key in store._key_to_row.keys()]
        return valid_ids, embeddings_np

    def _extract_notice_features(self, store: FeatureStore) -> Dict:
        """
        FeatureStore에서 Notice 피처 추출 및 projection 적용
        """
        # Numeric 데이터
        if store._num_mat is not None:
            numeric_data = store._num_mat
        else:
            numeric_data = np.zeros((len(store._key_to_row), len(self.schema.notice.numeric)), dtype=np.float32)

        # Text 임베딩
        text_dict = {}
        if store._txt_mat is not None and len(store._txt_mat) > 0:
            for col in self.schema.notice.text:
                if col in store._txt_mat:
                    text_dict[col] = store._txt_mat[col]
                else:
                    text_dict[col] = np.zeros((len(store._key_to_row), 768), dtype=np.float32)
        else:
            for col in self.schema.notice.text:
                text_dict[col] = np.zeros((len(store._key_to_row), 768), dtype=np.float32)

        # Categorical 데이터
        if store._cat_mat is not None:
            categorical_data = store._cat_mat
        else:
            categorical_data = np.zeros((len(store._key_to_row), len(self.schema.notice.categorical)), dtype=np.int64)

        # Projection 적용 (numeric + text → dense_projected)
        numeric_tensor = torch.from_numpy(numeric_data).float().to(self.device)
        text_dict_tensor = {
            col: torch.from_numpy(emb).float().to(self.device)
            for col, emb in text_dict.items()
        }

        with torch.no_grad():
            dense_projected, text_projected = self.preprocessor.projectors['notice'](
                numeric_tensor, text_dict_tensor
            )

            # dense_projected: (B, 128), text_projected: {col: (B, 128)}
            if text_projected:
                text_proj_list = [text_projected[col] for col in sorted(text_projected.keys())]
                text_proj_concat = torch.cat(text_proj_list, dim=1)
                final_dense = torch.cat([dense_projected, text_proj_concat], dim=1)
            else:
                final_dense = dense_projected

            dense_projected_np = final_dense.cpu().numpy()

        return {
            'dense_projected': dense_projected_np,
            'categorical': categorical_data
        }

    def _extract_company_features(self, store: FeatureStore) -> Dict:
        """
        FeatureStore에서 Company 피처 추출 및 projection 적용
        """
        # Numeric 데이터
        if store._num_mat is not None:
            numeric_data = store._num_mat
        else:
            numeric_data = np.zeros((len(store._key_to_row), len(self.schema.company.numeric)), dtype=np.float32)

        # Text 임베딩 (Company는 text가 없을 수 있음)
        text_dict = {}
        if hasattr(self.schema.company, 'text') and self.schema.company.text:
            if store._txt_mat is not None and len(store._txt_mat) > 0:
                for col in self.schema.company.text:
                    if col in store._txt_mat:
                        text_dict[col] = store._txt_mat[col]
                    else:
                        text_dict[col] = np.zeros((len(store._key_to_row), 768), dtype=np.float32)
            else:
                for col in self.schema.company.text:
                    text_dict[col] = np.zeros((len(store._key_to_row), 768), dtype=np.float32)

        # Categorical 데이터
        if store._cat_mat is not None:
            categorical_data = store._cat_mat
        else:
            categorical_data = np.zeros((len(store._key_to_row), len(self.schema.company.categorical)), dtype=np.int64)

        # Projection 적용
        numeric_tensor = torch.from_numpy(numeric_data).float().to(self.device)

        with torch.no_grad():
            if text_dict:
                text_dict_tensor = {
                    col: torch.from_numpy(emb).float().to(self.device)
                    for col, emb in text_dict.items()
                }
                dense_projected, text_projected = self.preprocessor.projectors['company'](
                    numeric_tensor, text_dict_tensor
                )

                if text_projected:
                    text_proj_list = [text_projected[col] for col in sorted(text_projected.keys())]
                    text_proj_concat = torch.cat(text_proj_list, dim=1)
                    final_dense = torch.cat([dense_projected, text_proj_concat], dim=1)
                else:
                    final_dense = dense_projected
            else:
                # Text가 없는 경우 numeric만 projection
                final_dense = self.preprocessor.projectors['company'].num_proj(numeric_tensor)

            dense_projected_np = final_dense.cpu().numpy()

        return {
            'dense_projected': dense_projected_np,
            'categorical': categorical_data
        }

    def _create_notice_input(self, notice_data: Dict) -> Dict[str, torch.Tensor]:
        """
        전처리된 Notice 데이터를 모델 입력 형식으로 변환
        """
        dense_tensor = torch.from_numpy(notice_data['dense_projected']).float().to(self.device)
        cat_tensor = torch.from_numpy(notice_data['categorical']).long().to(self.device)
        kjt = create_kjt_from_batch_gpu(cat_tensor, self.schema.notice.categorical, self.device)

        return {
            "dense": dense_tensor,
            "kjt": kjt
        }

    def _create_company_input(self, company_data: Dict) -> Dict[str, torch.Tensor]:
        """
        전처리된 Company 데이터를 모델 입력 형식으로 변환
        """
        dense_tensor = torch.from_numpy(company_data['dense_projected']).float().to(self.device)
        cat_tensor = torch.from_numpy(company_data['categorical']).long().to(self.device)
        kjt = create_kjt_from_batch_gpu(cat_tensor, self.schema.company.categorical, self.device)

        return {
            "dense": dense_tensor,
            "kjt": kjt
        }


# 편의 함수들
def get_notice_embedding(
    checkpoint_path: str,
    db_engine: Engine,
    bidntceno: str,
    bidntceord: str = "00",
    device: str = "cuda"
) -> Optional[np.ndarray]:
    """
    단일 공고의 임베딩 벡터 추출 (원샷 함수)

    여러 번 호출할 경우 EmbeddingService 인스턴스를 직접 사용하는 것이 효율적입니다.

    Args:
        checkpoint_path: 모델 체크포인트 경로
        db_engine: 데이터베이스 엔진
        bidntceno: 공고번호
        bidntceord: 공고차수
        device: 디바이스

    Returns:
        np.ndarray: 임베딩 벡터 또는 None
    """
    service = EmbeddingService(
        checkpoint_path=checkpoint_path,
        db_engine=db_engine,
        device=device
    )
    return service.get_notice_embedding(bidntceno, bidntceord)


def get_company_embedding(
    checkpoint_path: str,
    db_engine: Engine,
    bizno: str,
    device: str = "cuda"
) -> Optional[np.ndarray]:
    """
    단일 업체의 임베딩 벡터 추출 (원샷 함수)

    여러 번 호출할 경우 EmbeddingService 인스턴스를 직접 사용하는 것이 효율적입니다.

    Args:
        checkpoint_path: 모델 체크포인트 경로
        db_engine: 데이터베이스 엔진
        bizno: 사업자등록번호
        device: 디바이스

    Returns:
        np.ndarray: 임베딩 벡터 또는 None
    """
    service = EmbeddingService(
        checkpoint_path=checkpoint_path,
        db_engine=db_engine,
        device=device
    )
    return service.get_company_embedding(bizno)




