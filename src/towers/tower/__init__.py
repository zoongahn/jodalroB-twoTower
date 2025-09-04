from typing import List, Dict
import torch

from src.towers.tower.notice_tower import NoticeTower
from src.towers.tower.company_tower import CompanyTower


# Factory 함수들
def create_notice_tower(
    categorical_keys: List[str], 
    vocab_sizes: Dict[str, int],
    **kwargs
) -> NoticeTower:
    """Notice Tower 생성 헬퍼 함수"""
    return NoticeTower(
        categorical_keys=categorical_keys,
        categorical_vocab_sizes=vocab_sizes,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    )


def create_company_tower(
    categorical_keys: List[str], 
    vocab_sizes: Dict[str, int],
    **kwargs
) -> CompanyTower:
    """Company Tower 생성 헬퍼 함수"""
    return CompanyTower(
        categorical_keys=categorical_keys,
        categorical_vocab_sizes=vocab_sizes,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    )