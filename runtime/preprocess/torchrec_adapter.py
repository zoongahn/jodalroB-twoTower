"""
TorchRec Adapter for runtime preprocessing

This adapter wraps the complex TorchRec preprocessing pipeline
for simple runtime usage.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python path에 추가 (임시)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 원본 preprocess 모듈 import
from preprocess.torchrec.schema import TorchRecSchema, build_torchrec_schema_from_meta
from preprocess.torchrec.feature_preprocessor import FeaturePreprocessor
from preprocess.torchrec.feature_projector import FeatureProjector
# KJT utils는 towers에서 import
from src.towers.kjt_utils import create_kjt_from_batch_gpu


class TorchRecAdapter:
    """
    Runtime adapter for TorchRec preprocessing

    This class simplifies the complex preprocessing pipeline
    for runtime prediction usage.
    """

    def __init__(
        self,
        schema: Optional[TorchRecSchema] = None,
        metadata_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize the adapter

        Args:
            schema: Pre-built TorchRecSchema (if available)
            metadata_path: Path to metadata CSV file
            device: Device to use for preprocessing
        """
        self.device = device

        # Schema 초기화
        if schema is not None:
            self.schema = schema
        elif metadata_path is not None:
            self.schema = build_torchrec_schema_from_meta(metadata_path)
        else:
            # 기본 metadata 경로 사용
            default_meta = project_root / "meta" / "metadata.csv"
            if default_meta.exists():
                self.schema = build_torchrec_schema_from_meta(str(default_meta))
            else:
                raise ValueError("No schema or metadata_path provided")

        # Preprocessor 초기화
        self.preprocessor = FeaturePreprocessor(self.schema)

        # Projector 초기화
        self.projector = FeatureProjector(self.schema)

    def preprocess_notice(
        self,
        notice_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Preprocess notice data for model input

        Args:
            notice_data: Raw notice data from database

        Returns:
            Preprocessed notice features
        """
        # Feature preprocessing
        processed = self.preprocessor.process_notice(notice_data)

        # Feature projection
        projected = self.projector.project_notice(processed)

        return projected

    def preprocess_company(
        self,
        company_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Preprocess company data for model input

        Args:
            company_data: Raw company data from database

        Returns:
            Preprocessed company features
        """
        # Feature preprocessing
        processed = self.preprocessor.process_company(company_data)

        # Feature projection
        projected = self.projector.project_company(processed)

        return projected

    def create_kjt_inputs(
        self,
        features: Dict[str, Any],
        side: str = "notice"
    ):
        """
        Create TorchRec KJT inputs from preprocessed features

        Args:
            features: Preprocessed features dict with 'categorical' and 'dense_projected'
            side: "notice" or "company"

        Returns:
            Dict with 'dense' tensor and 'kjt' KeyedJaggedTensor
        """
        import torch

        # Dense features
        dense_tensor = torch.from_numpy(features['dense_projected']).float().to(self.device)

        # Categorical features to KJT
        cat_tensor = torch.from_numpy(features['categorical']).long().to(self.device)

        # Get categorical columns based on side
        if side == "notice":
            cat_cols = self.schema.notice.categorical
        else:
            cat_cols = self.schema.company.categorical

        kjt = create_kjt_from_batch_gpu(cat_tensor, cat_cols, self.device)

        return {
            "dense": dense_tensor,
            "kjt": kjt
        }


# Convenience functions for backward compatibility
def preprocess_for_runtime(
    notice_data: Dict[str, Any],
    company_data: Dict[str, Any],
    schema: Optional[TorchRecSchema] = None
) -> tuple:
    """
    Simple preprocessing function for runtime

    Args:
        notice_data: Raw notice data
        company_data: Raw company data
        schema: Optional pre-built schema

    Returns:
        Tuple of (notice_features, company_features)
    """
    adapter = TorchRecAdapter(schema=schema)

    notice_features = adapter.preprocess_notice(notice_data)
    company_features = adapter.preprocess_company(company_data)

    return notice_features, company_features