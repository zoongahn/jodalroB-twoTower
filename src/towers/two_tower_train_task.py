import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

from src.towers.two_tower_model import TwoTowerModel


class TwoTowerTrainTask(nn.Module):
    """
    Two-Tower 학습 태스크 클래스
    - Positive pair 배치 처리
    - In-batch negative sampling
    - Cross entropy loss 계산
    """
    def __init__(
        self,
        two_tower_model: TwoTowerModel,
        temperature: float = 1.0,
        loss_type: str = "cross_entropy",
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            two_tower_model: 학습할 TwoTowerModel 인스턴스
            temperature: 유사도 계산시 temperature scaling
            loss_type: 손실 함수 타입 ("cross_entropy", "cosine_embedding")
            label_smoothing: Label smoothing 정도 (0.0 ~ 1.0)
        """
        super().__init__()
        
        self.two_tower_model = two_tower_model
        self.temperature = temperature
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        
        if loss_type not in ["cross_entropy", "cosine_embedding"]:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    def forward(
        self, 
        batch: Dict[str, Dict[str, torch.Tensor]],
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            batch: {
                "notice": {"dense": Tensor, "kjt": KJT},
                "company": {"dense": Tensor, "kjt": KJT}
            }
            return_metrics: True면 추가 메트릭도 반환
            
        Returns:
            return_metrics=False: loss tensor
            return_metrics=True: {"loss": tensor, "accuracy": tensor, "similarities": tensor}
        """
        notice_input = batch["notice"]
        company_input = batch["company"]
        
        # 1) 배치 크기 검증
        notice_batch_size = notice_input["dense"].size(0)
        company_batch_size = company_input["dense"].size(0)
        
        if notice_batch_size != company_batch_size:
            raise ValueError(
                f"Notice와 Company 배치 크기가 다릅니다: {notice_batch_size} vs {company_batch_size}"
            )
        
        batch_size = notice_batch_size
        
        # 2) 임베딩 계산
        notice_embeddings, company_embeddings = self.two_tower_model(
            notice_input, company_input
        )
        
        # 3) 유사도 행렬 계산 (In-batch negative sampling)
        similarity_matrix = self._compute_similarity_matrix(
            notice_embeddings, company_embeddings
        )  # [B, B]
        
        # 4) 손실 계산
        loss = self._compute_loss(similarity_matrix, batch_size)
        
        if return_metrics:
            metrics = self._compute_metrics(similarity_matrix, batch_size)
            return {
                "loss": loss,
                **metrics,
                "similarity_matrix": similarity_matrix.detach()
            }
        else:
            return loss
    
    def _compute_similarity_matrix(
        self, 
        notice_embeddings: torch.Tensor, 
        company_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """유사도 행렬 계산"""
        # 코사인 유사도 (L2 정규화된 벡터이므로)
        similarity_matrix = torch.mm(notice_embeddings, company_embeddings.t())
        
        # Temperature scaling 적용
        if self.temperature != 1.0:
            similarity_matrix = similarity_matrix / self.temperature
            
        return similarity_matrix
    
    def _compute_loss(self, similarity_matrix: torch.Tensor, batch_size: int) -> torch.Tensor:
        """손실 계산"""
        # similarity_matrix[j, i] = sθ(uj, vi)
        if self.loss_type == "cross_entropy":
            # 대각선이 positive pair (정답 레이블)
            labels = torch.arange(batch_size, device=similarity_matrix.device)
            
            # Cross entropy loss (각 공고가 자신과 매칭되는 업체를 선택)
            loss = F.cross_entropy(
                similarity_matrix, 
                labels, 
                label_smoothing=self.label_smoothing
            )
            
        elif self.loss_type == "cosine_embedding":
            # Cosine embedding loss 구현
            batch_size = similarity_matrix.size(0)
            device = similarity_matrix.device
            
            # Positive pairs (대각선)
            positive_similarities = torch.diag(similarity_matrix)
            positive_targets = torch.ones(batch_size, device=device)
            
            # Negative pairs (비대각선)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            negative_similarities = similarity_matrix[mask]
            negative_targets = -torch.ones(len(negative_similarities), device=device)
            
            # 전체 손실
            all_similarities = torch.cat([positive_similarities, negative_similarities])
            all_targets = torch.cat([positive_targets, negative_targets])
            
            loss = F.cosine_embedding_loss(
                all_similarities.unsqueeze(1), 
                torch.ones_like(all_similarities.unsqueeze(1)), 
                all_targets
            )
        
        return loss
    
    def _compute_metrics(self, similarity_matrix: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """평가 메트릭 계산"""
        # Top-1 정확도 (각 공고가 자신의 업체를 1순위로 선택하는 비율)
        predicted_indices = torch.argmax(similarity_matrix, dim=1)
        correct_indices = torch.arange(batch_size, device=similarity_matrix.device)
        accuracy = (predicted_indices == correct_indices).float().mean()
        
        # 대각선 vs 평균 유사도
        diagonal_similarities = torch.diag(similarity_matrix)
        off_diagonal_mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        off_diagonal_similarities = similarity_matrix[off_diagonal_mask]
        
        return {
            "accuracy": accuracy,
            "positive_similarity_mean": diagonal_similarities.mean(),
            "negative_similarity_mean": off_diagonal_similarities.mean(),
            "similarity_gap": diagonal_similarities.mean() - off_diagonal_similarities.mean()
        }
    
    def predict_batch(
        self, 
        batch: Dict[str, Dict[str, torch.Tensor]], 
        top_k: int = 10
    ) -> Dict[str, torch.Tensor]:
        """배치에 대한 Top-K 예측 (추론용)"""
        self.eval()
        with torch.no_grad():
            notice_input = batch["notice"]
            company_input = batch["company"]
            
            notice_embeddings, company_embeddings = self.two_tower_model(
                notice_input, company_input
            )
            
            similarity_matrix = self._compute_similarity_matrix(
                notice_embeddings, company_embeddings
            )
            
            # Top-K 선택
            top_similarities, top_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
            
            return {
                "top_similarities": top_similarities,  # [B, K]
                "top_indices": top_indices,            # [B, K]
                "all_similarities": similarity_matrix  # [B, B]
            }


# Helper 함수들
def create_two_tower_train_task(
    notice_categorical_keys,
    company_categorical_keys,
    metadata_path: str = "meta/metadata.csv",
    categorical_embedding_dim: int = 64,
    notice_dense_input_dim: int = 256,
    company_dense_input_dim: int = 128,
    tower_hidden_dims = None,
    final_embedding_dim: int = 128,
    dropout_rate: float = 0.2,
    temperature: float = 1.0,
    loss_type: str = "cross_entropy",
    device = "cuda:0"
) -> TwoTowerTrainTask:
    """TwoTowerTrainTask 생성 헬퍼 함수"""
    from src.towers.two_tower_model import create_two_tower_model
    
    # TwoTowerModel 생성
    two_tower_model = create_two_tower_model(
        notice_categorical_keys=notice_categorical_keys,
        company_categorical_keys=company_categorical_keys,
        metadata_path=metadata_path,
        categorical_embedding_dim=categorical_embedding_dim,
        notice_dense_input_dim=notice_dense_input_dim,
        company_dense_input_dim=company_dense_input_dim,
        tower_hidden_dims=tower_hidden_dims,
        final_embedding_dim=final_embedding_dim,
        dropout_rate=dropout_rate,
        device=device
    )
    
    # TrainTask 생성
    train_task = TwoTowerTrainTask(
        two_tower_model=two_tower_model,
        temperature=temperature,
        loss_type=loss_type
    )
    
    return train_task