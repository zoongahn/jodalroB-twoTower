import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# SafeCategoricalEmbedder 임포트 (경로는 실제 구조에 맞게 조정)
from src.towers.cat_embed import CategoricalEmbedder


class BaseTower(nn.Module):
    """
    Two-Tower 모델의 기본 타워 클래스
    """
    def __init__(
        self,
        categorical_keys: List[str],
        metadata_path: str = "meta/metadata.csv",
        table_name: str = "notice",
        categorical_embedding_dim: int = 64,
        dense_input_dim: int = 256,
        tower_hidden_dims: Optional[List[int]] = None,
        final_embedding_dim: int = 128,
        dropout_rate: float = 0.2,
        device: Optional[torch.device] = "cuda:0",
    ):
        """
        Base Tower 모델
        
        Args:
            categorical_keys: 범주형 피처 키 리스트
            metadata_path: 메타데이터 CSV 파일 경로
            table_name: 테이블명 (notice 또는 company)
            categorical_embedding_dim: 범주형 임베딩 차원
            dense_input_dim: dense 입력 차원 (수치형+텍스트 결합 후)
            tower_hidden_dims: MLP 히든 레이어 차원들
            final_embedding_dim: 최종 출력 임베딩 차원
            dropout_rate: 드롭아웃 비율
            device: 디바이스
        """
        super().__init__()
        
        if tower_hidden_dims is None:
            tower_hidden_dims = [256, 128]
        
        self.categorical_keys = categorical_keys
        self.device = device
                
        # 1) SafeCategoricalEmbedder 사용
        self.categorical_embedder = CategoricalEmbedder(
            keys=categorical_keys,
            metadata_path=metadata_path,
            table_name=table_name,
            embedding_dim=categorical_embedding_dim,
            device=str(self.device),
        ).to(torch.device(self.device))
        
        # 2) Dense 프로젝션
        self.dense_projection = nn.Linear(dense_input_dim, tower_hidden_dims[0])
        
        # 3) MLP 구성
        self._build_mlp(
            categorical_embedding_dim=categorical_embedding_dim,
            tower_hidden_dims=tower_hidden_dims,
            final_embedding_dim=final_embedding_dim,
            dropout_rate=dropout_rate
        )
        
        self.to(torch.device("cuda:0"))
    
    def _build_mlp(
        self, 
        categorical_embedding_dim: int, 
        tower_hidden_dims: List[int],
        final_embedding_dim: int,
        dropout_rate: float
    ):
        """MLP 레이어 구성"""
        # 결합된 피처 차원 계산
        categorical_total_dim = len(self.categorical_keys) * categorical_embedding_dim
        combined_input_dim = tower_hidden_dims[0] + categorical_total_dim
        
        mlp_layers = []
        input_dim = combined_input_dim
        
        # 히든 레이어들
        for hidden_dim in tower_hidden_dims[1:]:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate),
            ])
            input_dim = hidden_dim
        
        # 최종 출력 레이어
        mlp_layers.append(nn.Linear(input_dim, final_embedding_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, tower_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tower_input: {
                "dense": torch.Tensor [B, dense_dim],
                "kjt": KeyedJaggedTensor,
                "text": Dict[str, torch.Tensor] (optional)
            }
        
        Returns:
            torch.Tensor [B, final_embedding_dim]: 최종 타워 임베딩
        """
        
        # 모든 파라미터와 버퍼를 cuda:0으로 이동
        for name, param in self.named_parameters():
            if param.device != torch.device("cuda:0"):
                print(f"WARNING: param {name} is on {param.device}")
        
        # 버퍼도 이동 (BatchNorm의 running_mean, running_var 등)
        for name, buffer in self.named_buffers():
            if buffer.device != torch.device("cuda:0"):
                print(f"WARNING: buffer {name} is on {buffer.device}")
                
        # 어떤 파라미터가 requires_grad=False인지 확인
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"{name}: requires_grad={param.requires_grad}")
        
        dense = tower_input["dense"].to(self.device)
        kjt = tower_input["kjt"].to(self.device)
        
        # 1) Dense 피처 프로젝션
        dense_projected = self.dense_projection(dense)
        
        # 2) 범주형 임베딩
        categorical_combined = self.categorical_embedder(kjt, return_dict=False)
        
        # 3) 결합
        combined_features = torch.cat([dense_projected, categorical_combined], dim=1)
        
        # 4) MLP를 통과하여 최종 임베딩 생성
        tower_embedding = self.mlp(combined_features)
        
        # L2 정규화
        tower_embedding = torch.nn.functional.normalize(tower_embedding, p=2, dim=1)
        
        return tower_embedding