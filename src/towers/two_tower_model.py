import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from src.towers.tower.notice_tower import NoticeTower
from src.towers.tower.company_tower import CompanyTower


class TwoTowerModel(nn.Module):
    """
    Two-Tower 모델 - Notice Tower와 Company Tower를 결합
    """
    def __init__(
        self,
        notice_tower_config: Dict,
        company_tower_config: Dict,
        final_embedding_dim: int = 128,
        device: Optional[torch.device] = "cuda:0",
    ):
        """
        Args:
            notice_tower_config: Notice Tower 설정 dict
            company_tower_config: Company Tower 설정 dict  
            final_embedding_dim: 최종 임베딩 차원 (두 타워 동일해야 함)
            device: 디바이스
        """
        super().__init__()
        
        self.device = device
        self.final_embedding_dim = final_embedding_dim
        
        # 두 타워 생성
        self.notice_tower = NoticeTower(**notice_tower_config)
        self.company_tower = CompanyTower(**company_tower_config)
        
        # 최종 임베딩 차원이 동일한지 확인
        assert notice_tower_config.get("final_embedding_dim", 128) == final_embedding_dim
        assert company_tower_config.get("final_embedding_dim", 128) == final_embedding_dim
        
        self.to(torch.device(self.device))
        
    def forward(
        self, 
        notice_input: Dict[str, torch.Tensor], 
        company_input: Dict[str, torch.Tensor],
        return_similarity: bool = False,
        temperature: float = 1.0
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            notice_input: Notice Tower 입력 {"dense": Tensor, "kjt": KJT}
            company_input: Company Tower 입력 {"dense": Tensor, "kjt": KJT}  
            return_similarity: True면 유사도 행렬도 함께 반환
            temperature: 유사도 계산시 temperature scaling
        
        Returns:
            return_similarity=False: (notice_emb, company_emb)
            return_similarity=True: {
                "notice_emb": Tensor [B, D],
                "company_emb": Tensor [B, D], 
                "similarity_matrix": Tensor [B, B]
            }
        """
        
        # 1) 각 타워에서 임베딩 계산
        notice_embeddings = self.notice_tower(notice_input)    # [B, final_embedding_dim]
        company_embeddings = self.company_tower(company_input)  # [B, final_embedding_dim]
        
        # 2) 배치 크기 검증
        batch_size_notice = notice_embeddings.size(0)
        batch_size_company = company_embeddings.size(0)
        
        if batch_size_notice != batch_size_company:
            raise ValueError(
                f"Notice와 Company 배치 크기가 다릅니다: "
                f"{batch_size_notice} vs {batch_size_company}"
            )
        
        # 3) 임베딩이 정규화되었는지 확인 (BaseTower에서 L2 정규화 수행)
        notice_norms = torch.norm(notice_embeddings, p=2, dim=1)
        company_norms = torch.norm(company_embeddings, p=2, dim=1)
        
        if not torch.allclose(notice_norms, torch.ones_like(notice_norms), atol=1e-4):
            print("Warning: Notice embeddings are not L2 normalized")
        if not torch.allclose(company_norms, torch.ones_like(company_norms), atol=1e-4):
            print("Warning: Company embeddings are not L2 normalized")
        
        # 4) 유사도 계산 여부에 따라 반환
        if return_similarity:
            # 코사인 유사도 행렬 계산 (L2 정규화된 벡터이므로)
            similarity_matrix = torch.mm(notice_embeddings, company_embeddings.t()) / temperature
            
            return {
                "notice_embeddings": notice_embeddings,
                "company_embeddings": company_embeddings,
                "similarity_matrix": similarity_matrix
            }
        else:
            return notice_embeddings, company_embeddings
    
    def get_notice_embeddings(self, notice_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Notice만 단독으로 임베딩 계산 (추론시 사용)"""
        return self.notice_tower(notice_input)
    
    def get_company_embeddings(self, company_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Company만 단독으로 임베딩 계산 (인덱스 구축시 사용)"""
        return self.company_tower(company_input)
    
    def compute_similarity(
        self, 
        notice_emb: torch.Tensor, 
        company_emb: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """두 임베딩 간 유사도 행렬 계산"""
        return torch.mm(notice_emb, company_emb.t()) / temperature



# TwoTower 생성을 위한 헬퍼 함수
def create_two_tower_model(
    notice_categorical_keys: List[str],
    company_categorical_keys: List[str],
    metadata_path: str = "meta/metadata.csv",
    categorical_embedding_dim: int = 64,
    notice_dense_input_dim: int = 256,
    company_dense_input_dim: int = 128,
    tower_hidden_dims: Optional[List[int]] = None,
    final_embedding_dim: int = 128,
    dropout_rate: float = 0.2,
    device: Optional[torch.device] = "cuda:0",
) -> TwoTowerModel:
    """TwoTower 모델 생성 헬퍼"""
    
    if tower_hidden_dims is None:
        tower_hidden_dims = [256, 128]
    
    notice_config = {
        "categorical_keys": notice_categorical_keys,
        "metadata_path": metadata_path,
        "categorical_embedding_dim": categorical_embedding_dim,
        "dense_input_dim": notice_dense_input_dim,
        "tower_hidden_dims": tower_hidden_dims,
        "final_embedding_dim": final_embedding_dim,
        "dropout_rate": dropout_rate,
        "device": device,
    }
    
    company_config = {
        "categorical_keys": company_categorical_keys,
        "metadata_path": metadata_path,
        "categorical_embedding_dim": categorical_embedding_dim,
        "dense_input_dim": company_dense_input_dim,
        "tower_hidden_dims": tower_hidden_dims,
        "final_embedding_dim": final_embedding_dim,
        "dropout_rate": dropout_rate,
        "device": device,
    }
    
    return TwoTowerModel(
        notice_tower_config=notice_config,
        company_tower_config=company_config,
        final_embedding_dim=final_embedding_dim,
        device=device,
    )