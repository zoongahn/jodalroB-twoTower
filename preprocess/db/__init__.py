"""
DB Preprocessing Module

public.notice → step1.notice_preprocessed 전처리 모듈

사전 DB 전처리 (런타임 전처리와 구분):
- 원본 데이터(public.notice)를 읽어서
- 수치형, 범주형, 텍스트 전처리를 수행하고
- step1.notice_preprocessed 테이블에 저장

Modules:
- numeric.py: 수치형 데이터 전처리 (결측치, 클리핑, 스케일링)
- categorical.py: 범주형 데이터 인코딩 (Label Encoding + NULL/RARE/UNKNOWN 처리)
- text.py: 텍스트 임베딩 (KoELECTRA)
- pipeline.py: 전체 파이프라인 오케스트레이션
"""

from .pipeline import DataPreprocessor, preprocess_pipeline
from .numeric import NumericPreprocessor, preprocess_numeric_data
from .categorical import CategoricalPreprocessor, preprocess_categorical_data
from .text import TextPreprocessor, preprocess_text_data

__all__ = [
    "DataPreprocessor",
    "preprocess_pipeline",
    "NumericPreprocessor",
    "preprocess_numeric_data",
    "CategoricalPreprocessor",
    "preprocess_categorical_data",
    "TextPreprocessor",
    "preprocess_text_data",
]
