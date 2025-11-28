"""
STEP1: Two-Tower Model Runtime Package

This package provides the runtime components for the Two-Tower recommendation system.
It is designed to be used by the jodalroB-platform for serving predictions.
"""

from .service import initialize, run_step1_prediction

__version__ = '1.0.0'
__all__ = ['initialize', 'run_step1_prediction']