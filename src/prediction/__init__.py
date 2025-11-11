"""
Prediction module for Two-Tower model.

Provides model loading and prediction capabilities for production inference.
"""

from src.prediction.model_loader import ModelLoader
from src.prediction.predictor import TwoTowerPredictor

__all__ = ['ModelLoader', 'TwoTowerPredictor']
