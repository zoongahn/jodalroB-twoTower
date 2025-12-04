"""
Vectorize Module: 임베딩 추출 및 벡터 DB 관리

주요 구성 요소:
- EmbeddingExtractor: 모델로부터 임베딩 추출
- EmbeddingStore: Faiss 기반 벡터 DB
"""

from .embedding_extractor import EmbeddingExtractor
from .embedding_store import EmbeddingStore

__all__ = [
    'EmbeddingExtractor',
    'EmbeddingStore',
]
