from typing import List, Dict, Optional
import torch

from src.towers.tower.base_tower import BaseTower


class CompanyTower(BaseTower):
    """
    업체(Company) 타워 - BaseTower를 상속
    """
    def __init__(
        self,
        categorical_keys: List[str],
        metadata_path: str = "meta/metadata.csv",
        categorical_embedding_dim: int = 64,
        dense_input_dim: int = 256,
        tower_hidden_dims: Optional[List[int]] = None,
        final_embedding_dim: int = 128,
        dropout_rate: float = 0.2,
        device: Optional[torch.device] = "cuda:0",
    ):
        super().__init__(
            categorical_keys=categorical_keys,
            metadata_path=metadata_path,
            table_name="company",  # Company 테이블 고정
            categorical_embedding_dim=categorical_embedding_dim,
            dense_input_dim=dense_input_dim,
            tower_hidden_dims=tower_hidden_dims,
            final_embedding_dim=final_embedding_dim,
            dropout_rate=dropout_rate,
            device=device,
        )