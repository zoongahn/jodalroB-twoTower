"""
NoticeTower - EmbeddingBagCollection 기반
"""

from typing import List, Dict, Optional
import torch
from torchrec.modules.embedding_configs import PoolingType

from src.towers.tower.base_tower import BaseTower


class NoticeTower(BaseTower):
    """
    공고(Notice) 타워 - BaseTower를 상속

    EmbeddingBagCollection 기반
    """

    def __init__(
        self,
        metadata_path: str = "meta/metadata.csv",
        categorical_embedding_dim: int = 32,
        dense_input_dim: int = 256,
        tower_hidden_dims: Optional[List[int]] = None,
        final_embedding_dim: int = 128,
        dropout_rate: float = 0.2,
        pooling_mode: PoolingType = PoolingType.MEAN,
        device: Optional[torch.device] = None,
        use_fp16: bool = False,
    ):
        super().__init__(
            table_name="notice",  # Notice 테이블 고정
            metadata_path=metadata_path,
            categorical_embedding_dim=categorical_embedding_dim,
            dense_input_dim=dense_input_dim,
            tower_hidden_dims=tower_hidden_dims,
            final_embedding_dim=final_embedding_dim,
            dropout_rate=dropout_rate,
            pooling_mode=pooling_mode,
            device=device,
            use_fp16=use_fp16,
        )
