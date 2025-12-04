"""
Prediction module for Two-Tower model.

Provides model loading and prediction capabilities for production inference.
"""

from src.prediction.model_loader import ModelLoader
from src.prediction.predictor import TwoTowerPredictor
from src.prediction.embedding_service import (
    EmbeddingService,
    get_notice_embedding,
    get_company_embedding
)

__all__ = [
    'ModelLoader',
    'TwoTowerPredictor',
    'EmbeddingService',
    'get_notice_embedding',
    'get_company_embedding'
]
