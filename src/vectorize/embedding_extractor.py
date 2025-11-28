"""
Embedding Extractor: 학습된 Two-Tower 모델로부터 임베딩 추출

주요 기능:
1. 모델 체크포인트 로드
2. DB로부터 전체 데이터 로드
3. 배치 단위로 임베딩 생성
4. Notice/Company 임베딩 분리 추출
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import json

from sqlalchemy.engine import Engine
from preprocess.torchrec.schema import TorchRecSchema
from preprocess.torchrec.feature_store import build_feature_store
from preprocess.torchrec.feature_preprocessor import FeaturePreprocessor
from src.towers.two_tower_train_task import create_two_tower_train_task


class EmbeddingExtractor:
    """
    Two-Tower 모델로부터 임베딩을 추출하는 클래스
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로
            config_path: 모델 설정 파일 경로 (None이면 checkpoint와 같은 디렉토리의 config.json)
            device: 사용할 디바이스 ("cuda" 또는 "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 설정 파일 로드
        if config_path is None:
            config_path = self.checkpoint_path.parent / "config.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        print(f"✓ 설정 파일 로드: {config_path}")
        print(f"✓ 디바이스: {self.device}")

        # 모델 초기화 (아직 로드하지 않음)
        self.model = None
        self.schema = None

    def load_model(self, schema: TorchRecSchema):
        """
        모델 로드

        Args:
            schema: TorchRec 스키마 (피처 정의)
        """
        self.schema = schema

        print("\n모델 초기화 중...")

        # TrainTask 생성
        from src.towers.two_tower_train_task import create_two_tower_train_task

        self.model = create_two_tower_train_task(
            notice_categorical_keys=schema.notice.categorical,
            company_categorical_keys=schema.company.categorical,
            metadata_path=self.config.get("metadata_path", "meta/metadata.csv"),
            categorical_embedding_dim=self.config["categorical_embedding_dim"],
            notice_dense_input_dim=self.config.get("notice_dense_input_dim", 256),
            company_dense_input_dim=self.config.get("company_dense_input_dim", 128),
            tower_hidden_dims=self.config.get("tower_hidden_dims", [512, 256]),
            final_embedding_dim=self.config["final_embedding_dim"],
            dropout_rate=self.config.get("dropout_rate", 0.1),
            temperature=self.config.get("temperature", 1.0),
            loss_type=self.config.get("loss_type", "cross_entropy"),
            device=self.device
        )

        # 체크포인트 로드
        print(f"체크포인트 로드: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ 모델 로드 완료")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Embedding dim: {self.config['final_embedding_dim']}")

    def extract_notice_embeddings(
        self,
        notice_store: Dict,
        batch_size: int = 1024,
        verbose: bool = True
    ) -> Tuple[List[Tuple], np.ndarray]:
        """
        전체 Notice 임베딩 추출

        Args:
            notice_store: Notice feature store (from build_feature_store)
            batch_size: 배치 크기
            verbose: 진행 상황 출력 여부

        Returns:
            Tuple of (notice_ids, embeddings)
            - notice_ids: List of (bidntceno, bidntceord) tuples
            - embeddings: np.ndarray of shape (N, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        notice_ids = notice_store['ids']
        dense_projected = notice_store['dense_projected']
        categorical = notice_store['categorical']

        num_samples = len(notice_ids)
        embedding_dim = self.config['final_embedding_dim']

        embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)

        num_batches = (num_samples + batch_size - 1) // batch_size

        iterator = tqdm(range(num_batches), desc="Extracting Notice embeddings") if verbose else range(num_batches)

        with torch.no_grad():
            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                # 배치 데이터 준비
                batch_dense = torch.from_numpy(dense_projected[start_idx:end_idx]).float().to(self.device)
                batch_cat = torch.from_numpy(categorical[start_idx:end_idx]).long().to(self.device)

                # KJT 생성
                from src.towers.kjt_utils import create_kjt_from_batch_gpu
                batch_kjt = create_kjt_from_batch_gpu(batch_cat, self.schema.notice.categorical, self.device)

                # Notice 입력 구성
                notice_input = {
                    "dense": batch_dense,
                    "kjt": batch_kjt
                }

                # Notice Tower만 forward
                batch_embeddings = self.model.two_tower_model.notice_tower(notice_input)

                # CPU로 이동하여 저장
                embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()

        if verbose:
            print(f"✓ Notice 임베딩 추출 완료: {num_samples} samples")

        return notice_ids, embeddings

    def extract_company_embeddings(
        self,
        company_store: Dict,
        batch_size: int = 1024,
        verbose: bool = True
    ) -> Tuple[List, np.ndarray]:
        """
        전체 Company 임베딩 추출

        Args:
            company_store: Company feature store (from build_feature_store)
            batch_size: 배치 크기
            verbose: 진행 상황 출력 여부

        Returns:
            Tuple of (company_ids, embeddings)
            - company_ids: List of company IDs
            - embeddings: np.ndarray of shape (N, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        company_ids = company_store['ids']
        dense_projected = company_store['dense_projected']
        categorical = company_store['categorical']

        num_samples = len(company_ids)
        embedding_dim = self.config['final_embedding_dim']

        embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)

        num_batches = (num_samples + batch_size - 1) // batch_size

        iterator = tqdm(range(num_batches), desc="Extracting Company embeddings") if verbose else range(num_batches)

        with torch.no_grad():
            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                # 배치 데이터 준비
                batch_dense = torch.from_numpy(dense_projected[start_idx:end_idx]).float().to(self.device)
                batch_cat = torch.from_numpy(categorical[start_idx:end_idx]).long().to(self.device)

                # KJT 생성
                from src.towers.kjt_utils import create_kjt_from_batch_gpu
                batch_kjt = create_kjt_from_batch_gpu(batch_cat, self.schema.company.categorical, self.device)

                # Company 입력 구성
                company_input = {
                    "dense": batch_dense,
                    "kjt": batch_kjt
                }

                # Company Tower만 forward
                batch_embeddings = self.model.two_tower_model.company_tower(company_input)

                # CPU로 이동하여 저장
                embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()

        if verbose:
            print(f"✓ Company 임베딩 추출 완료: {num_samples} samples")

        return company_ids, embeddings

    def extract_all_embeddings(
        self,
        db_engine: Engine,
        schema: TorchRecSchema,
        batch_size: int = 1024,
        feature_chunksize: int = 5000,
        verbose: bool = True
    ) -> Tuple[Tuple[List, np.ndarray], Tuple[List, np.ndarray]]:
        """
        전체 Notice와 Company 임베딩 추출 (원스톱 함수)

        Args:
            db_engine: DB 엔진
            schema: TorchRec 스키마
            batch_size: 임베딩 추출 배치 크기
            feature_chunksize: 피처 로딩 청크 크기
            verbose: 진행 상황 출력 여부

        Returns:
            Tuple of ((notice_ids, notice_embeddings), (company_ids, company_embeddings))
        """
        # 1. 모델 로드
        if self.model is None:
            self.load_model(schema)

        # 2. 피처 전처리 및 로드
        if verbose:
            print("\n" + "=" * 80)
            print("피처 전처리 중...")
            print("=" * 80)

        preprocessor = FeaturePreprocessor(
            schema=schema,
            device=self.device,
            num_proj_dim=128,
            text_proj_dim=128,
            batch_size=1024
        )

        preprocessed_stores = preprocessor.preprocess_all(
            db_engine=db_engine,
            feature_chunksize=feature_chunksize,
            show_progress=verbose
        )

        notice_store = preprocessed_stores['notice']
        company_store = preprocessed_stores['company']

        if verbose:
            print(f"\n✓ 피처 로딩 완료")
            print(f"  - Notice: {len(notice_store['ids']):,} samples")
            print(f"  - Company: {len(company_store['ids']):,} samples")

        # 3. 임베딩 추출
        if verbose:
            print("\n" + "=" * 80)
            print("임베딩 추출 중...")
            print("=" * 80)

        notice_ids, notice_embeddings = self.extract_notice_embeddings(
            notice_store, batch_size=batch_size, verbose=verbose
        )

        company_ids, company_embeddings = self.extract_company_embeddings(
            company_store, batch_size=batch_size, verbose=verbose
        )

        if verbose:
            print(f"\n✓ 전체 임베딩 추출 완료")
            print(f"  - Notice: {notice_embeddings.shape}")
            print(f"  - Company: {company_embeddings.shape}")

        return (notice_ids, notice_embeddings), (company_ids, company_embeddings)
