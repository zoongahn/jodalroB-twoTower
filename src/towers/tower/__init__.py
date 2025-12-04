"""
Tower 패키지 - EmbeddingBagCollection 기반
"""

from typing import List, Dict
import torch
from torchrec.modules.embedding_configs import PoolingType

from src.towers.tower.notice_tower import NoticeTower
from src.towers.tower.company_tower import CompanyTower
from src.towers.tower.base_tower import BaseTower


# Factory 함수들
def create_notice_tower(
    metadata_path: str = "meta/metadata.csv",
    categorical_embedding_dim: int = 32,
    dense_input_dim: int = 256,
    tower_hidden_dims: List[int] = None,
    final_embedding_dim: int = 128,
    dropout_rate: float = 0.2,
    pooling_mode: PoolingType = PoolingType.MEAN,
    device: torch.device = None,
    use_fp16: bool = False,
    **kwargs
) -> NoticeTower:
    """Notice Tower 생성 헬퍼 함수"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return NoticeTower(
        metadata_path=metadata_path,
        categorical_embedding_dim=categorical_embedding_dim,
        dense_input_dim=dense_input_dim,
        tower_hidden_dims=tower_hidden_dims,
        final_embedding_dim=final_embedding_dim,
        dropout_rate=dropout_rate,
        pooling_mode=pooling_mode,
        device=device,
        use_fp16=use_fp16,
        **kwargs
    )


def create_company_tower(
    metadata_path: str = "meta/metadata.csv",
    categorical_embedding_dim: int = 32,
    dense_input_dim: int = 256,
    tower_hidden_dims: List[int] = None,
    final_embedding_dim: int = 128,
    dropout_rate: float = 0.2,
    pooling_mode: PoolingType = PoolingType.MEAN,
    device: torch.device = None,
    use_fp16: bool = False,
    **kwargs
) -> CompanyTower:
    """Company Tower 생성 헬퍼 함수"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return CompanyTower(
        metadata_path=metadata_path,
        categorical_embedding_dim=categorical_embedding_dim,
        dense_input_dim=dense_input_dim,
        tower_hidden_dims=tower_hidden_dims,
        final_embedding_dim=final_embedding_dim,
        dropout_rate=dropout_rate,
        pooling_mode=pooling_mode,
        device=device,
        use_fp16=use_fp16,
        **kwargs
    )


__all__ = [
    "BaseTower",
    "NoticeTower",
    "CompanyTower",
    "create_notice_tower",
    "create_company_tower",
]
