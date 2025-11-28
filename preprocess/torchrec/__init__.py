"""
TorchRec Preprocessing Module

런타임 전처리 모듈 (학습/예측 시 실행)

step1.notice_preprocessed (사전 전처리된 데이터) → TorchRec 형식 변환

Modules:
- schema.py: 스키마 정의 (TorchRecSchema, SideSchema)
- feature_store.py: DB/Parquet에서 피처 로드
- feature_projector.py: 차원 축소 (numeric, text → 128D)
- feature_preprocessor.py: 고수준 전처리 오케스트레이션
- torchrec_inputs.py: TorchRec 형식 변환 (Dense + KJT)
- metadata_parser.py: 메타데이터 파싱
"""

from .schema import TorchRecSchema, SideSchema, build_torchrec_schema_from_meta
from .feature_store import build_feature_store, FeatureStore, build_feature_store_with_condition, load_feature_store_from_parquet
from .feature_projector import FeatureProjector
from .feature_preprocessor import FeaturePreprocessor
from .torchrec_inputs import get_tower_input
from .metadata_parser import MetadataParser

__all__ = [
    # Schema
    "TorchRecSchema",
    "SideSchema",
    "build_torchrec_schema_from_meta",

    # Feature Store
    "build_feature_store",
    "FeatureStore",
    "build_feature_store_with_condition",
    "load_feature_store_from_parquet",

    # Projector & Preprocessor
    "FeatureProjector",
    "FeaturePreprocessor",

    # TorchRec Inputs
    "get_tower_input",

    # Metadata
    "MetadataParser",
]
