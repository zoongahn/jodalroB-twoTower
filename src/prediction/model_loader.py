"""
Model Loader: 학습된 Two-Tower 모델을 로드하고 초기화

주요 기능:
1. 체크포인트 로드
2. Config 복원
3. Schema 초기화
4. Preprocessor 초기화
"""

import torch
import json
from pathlib import Path
from typing import Optional, Dict

from sqlalchemy.engine import Engine
from preprocess.torchrec.schema import TorchRecSchema, build_torchrec_schema_from_meta
from preprocess.torchrec.feature_preprocessor import FeaturePreprocessor
from src.towers.tower.notice_tower import NoticeTower
from src.towers.tower.company_tower import CompanyTower


class ModelLoader:
    """
    Two-Tower 모델 로더

    체크포인트와 설정을 로드하여 예측 가능한 상태로 모델을 준비합니다.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로 (예: output/models/20251023_055430/best_model.pt)
            config_path: 설정 파일 경로 (None이면 checkpoint와 같은 디렉토리의 config.json)
            device: 사용할 디바이스 ("cuda" 또는 "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 설정 파일 로드
        if config_path is None:
            config_path = self.checkpoint_path.parent / "config.json"
        else:
            config_path = Path(config_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {self.checkpoint_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        print(f"✓ 설정 파일 로드: {config_path}")
        print(f"✓ 디바이스: {self.device}")

        # 초기화할 객체들
        self.model = None
        self.schema = None
        self.preprocessor = None

    def load_model(self, schema: Optional[TorchRecSchema] = None) -> None:
        """
        모델 로드

        Args:
            schema: TorchRec 스키마 (None이면 config에서 자동 생성)
        """
        # Schema 로드 또는 생성
        if schema is None:
            print("\n스키마 구축 중...")
            schema_config = {
                "pair_notice_id_cols": ["bidntceno", "bidntceord"],
                "pair_company_id_cols": ["bizno"],
                "metadata_path": self.config.get("metadata_path", "meta/metadata.csv")
            }
            self.schema = build_torchrec_schema_from_meta(**schema_config)
            print(f"✓ Notice 피처: {len(self.schema.notice.categorical)}개 범주형, {len(self.schema.notice.numeric)}개 수치형")
            print(f"✓ Company 피처: {len(self.schema.company.categorical)}개 범주형, {len(self.schema.company.numeric)}개 수치형")
        else:
            self.schema = schema

        # 모델 생성 (train.py와 동일한 구조)
        print("\n모델 초기화 중...")

        notice_tower = NoticeTower(
            metadata_path=self.config.get("metadata_path", "meta/metadata.csv"),
            categorical_embedding_dim=self.config["categorical_embedding_dim"],
            dense_input_dim=self.config.get("notice_dense_input_dim", 256),
            tower_hidden_dims=self.config.get("tower_hidden_dims", [512, 256]),
            final_embedding_dim=self.config["final_embedding_dim"],
            dropout_rate=self.config.get("dropout_rate", 0.1),
            device=self.device,
            use_fp16=False,
        )

        company_tower = CompanyTower(
            metadata_path=self.config.get("metadata_path", "meta/metadata.csv"),
            categorical_embedding_dim=self.config["categorical_embedding_dim"],
            dense_input_dim=self.config.get("company_dense_input_dim", 128),
            tower_hidden_dims=self.config.get("tower_hidden_dims", [512, 256]),
            final_embedding_dim=self.config["final_embedding_dim"],
            dropout_rate=self.config.get("dropout_rate", 0.1),
            device=self.device,
            use_fp16=False,
        )

        # TwoTowerWrapper (train.py와 동일)
        class TwoTowerWrapper(torch.nn.Module):
            def __init__(self, notice_tower, company_tower):
                super().__init__()
                self.notice_tower = notice_tower
                self.company_tower = company_tower

            def forward(self, batch):
                notice_emb = self.notice_tower(batch["notice"])
                company_emb = self.company_tower(batch["company"])
                return notice_emb, company_emb

            def get_notice_embeddings(self, notice_input):
                return self.notice_tower(notice_input)

            def get_company_embeddings(self, company_input):
                return self.company_tower(company_input)

        self.model = TwoTowerWrapper(notice_tower, company_tower).to(self.device)

        # 체크포인트 로드
        print(f"체크포인트 로드: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # State dict 키 변환 (구버전 호환성)
        # torch.compile 모델: '_orig_mod.' prefix 제거
        # 구버전 TwoTowerTrainTask: 'two_tower_model.' prefix 제거
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # torch.compile로 저장된 경우
            if new_key.startswith('_orig_mod.'):
                new_key = new_key.replace('_orig_mod.', '')
            # 구버전 TwoTowerTrainTask로 저장된 경우
            if new_key.startswith('two_tower_model.'):
                new_key = new_key.replace('two_tower_model.', '')
            new_state_dict[new_key] = value

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        print(f"✓ 모델 로드 완료")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Embedding dim: {self.config['final_embedding_dim']}")

        # Metrics 출력 (있는 경우)
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            train_loss = metrics.get('train_loss', 'N/A')
            val_loss = metrics.get('val_loss', 'N/A')
            val_acc = metrics.get('val_acc', 'N/A')

            print(f"  - Train Loss: {train_loss if train_loss == 'N/A' else f'{train_loss:.4f}'}")
            print(f"  - Val Loss: {val_loss if val_loss == 'N/A' else f'{val_loss:.4f}'}")
            print(f"  - Val Accuracy: {val_acc if val_acc == 'N/A' else f'{val_acc:.4f}'}")

    def init_preprocessor(self) -> FeaturePreprocessor:
        """
        Preprocessor 초기화 또는 로드

        Returns:
            초기화된 FeaturePreprocessor
        """
        if self.schema is None:
            raise RuntimeError("스키마가 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        print("\nPreprocessor 초기화 중...")

        # 체크포인트와 같은 디렉토리에서 preprocessor.pt 찾기
        checkpoint_dir = Path(self.checkpoint_path).parent
        preprocessor_path = checkpoint_dir / 'preprocessor.pt'

        if preprocessor_path.exists():
            print(f"✓ 저장된 Preprocessor 로드: {preprocessor_path}")
            self.preprocessor = torch.load(preprocessor_path, map_location=self.device, weights_only=False)
            print("✓ Preprocessor 로드 완료")
        else:
            print("⚠️  저장된 Preprocessor 없음, 새로 초기화합니다")
            self.preprocessor = FeaturePreprocessor(
                schema=self.schema,
                device=self.device,
                num_proj_dim=128,
                text_proj_dim=128,
                batch_size=1024
            )
            print("✓ Preprocessor 초기화 완료")

        return self.preprocessor

    def get_model(self):
        """로드된 모델 반환"""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        return self.model

    def get_schema(self) -> TorchRecSchema:
        """로드된 스키마 반환"""
        if self.schema is None:
            raise RuntimeError("스키마가 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        return self.schema

    def get_preprocessor(self) -> FeaturePreprocessor:
        """Preprocessor 반환 (없으면 초기화)"""
        if self.preprocessor is None:
            return self.init_preprocessor()
        return self.preprocessor

    def get_config(self) -> Dict:
        """설정 딕셔너리 반환"""
        return self.config
