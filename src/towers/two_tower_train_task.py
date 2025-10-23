import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

from src.towers.two_tower_model import TwoTowerModel
from src.towers.contrastive_head import diag_consistency_report


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

        # 2) 모델 forward: 임베딩 + Contrastive Head를 통한 로짓 계산
        result = self.two_tower_model(
            notice_input, company_input, return_logits=True
        )

        if len(result) == 3:
            notice_embeddings, company_embeddings, logits = result
        else:
            # Fallback: 구형 시그니처 대비
            notice_embeddings, company_embeddings = result
            logits = self.two_tower_model.contrastive_head(notice_embeddings, company_embeddings)

        # 3) Positive pair 정합성 검증 (최초 2회만 - 디버깅용)
        if not hasattr(self, '_pair_check_count'):
            self._pair_check_count = 0

        if self._pair_check_count < 2:
            print(f"\n🔍 [Step {self._pair_check_count}] Positive Pair Alignment Check")
            z_gap = diag_consistency_report(
                notice_embeddings, company_embeddings,
                log_prefix=f"[diag@step{self._pair_check_count}]"
            )
            self._pair_check_count += 1

        # 4) 손실 계산 (in-batch negative sampling)
        loss = self._compute_loss_with_logits(logits, batch_size)
        
        if return_metrics:
            metrics = self._compute_metrics(logits, batch_size)
            return {
                "loss": loss,
                **metrics,
                "similarity_matrix": logits.detach()
            }
        else:
            return loss

    def _compute_loss_with_logits(self, logits: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Contrastive Head로부터 얻은 로짓으로 손실 계산

        Args:
            logits: Temperature-scaled similarity matrix [B, B]
            batch_size: 배치 크기

        Returns:
            loss: 양방향 Cross Entropy Loss
        """
        # In-batch negative sampling: 대각선이 positive pair
        labels = torch.arange(batch_size, device=logits.device)

        # 양방향 Cross Entropy Loss
        loss_i2t = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss_t2i = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)

        # 평균으로 결합
        loss = 0.5 * (loss_i2t + loss_t2i)

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

            # 양방향 Cross Entropy Loss (더 강한 학습 신호)
            loss_notice_to_company = F.cross_entropy(
                similarity_matrix,
                labels,
                label_smoothing=self.label_smoothing
            )
            loss_company_to_notice = F.cross_entropy(
                similarity_matrix.t(),
                labels,
                label_smoothing=self.label_smoothing
            )

            # 평균으로 결합
            loss = 0.5 * (loss_notice_to_company + loss_company_to_notice)
            
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


# 추가 메서드들을 TwoTowerTrainTask 클래스에 동적으로 추가
def _verify_positive_pair_alignment(self, similarity_matrix: torch.Tensor):
    """Positive pair 정합성 검증 - 대각선이 진짜 정답인지 확인"""
    B = similarity_matrix.size(0)

    # 각 notice가 가장 유사한 company 선택 (row-wise)
    row_top1 = similarity_matrix.argmax(dim=1)
    # 각 company가 가장 유사한 notice 선택 (col-wise)
    col_top1 = similarity_matrix.argmax(dim=0)

    # 대각선 인덱스 (정답이어야 할 위치)
    diag_idx = torch.arange(B, device=similarity_matrix.device)

    # 정확도 계산
    row_hit = (row_top1 == diag_idx).float().mean().item()  # notice → company
    col_hit = (col_top1 == diag_idx).float().mean().item()  # company → notice

    print(f"🔍 [Positive Pair Alignment Check]")
    print(f"   📊 Notice→Company Top-1 정확도: {row_hit:.3f}")
    print(f"   📊 Company→Notice Top-1 정확도: {col_hit:.3f}")

    # 대각선 vs 최대값 분석
    diag_values = similarity_matrix.diag()
    max_values = similarity_matrix.max(dim=1)[0]
    diag_is_max = (diag_values == max_values).float().mean().item()

    print(f"   🎯 대각선이 row-max인 비율: {diag_is_max:.3f}")

    # 경고 판정
    if row_hit < 0.05 and col_hit < 0.05:
        print("   🚨 CRITICAL: Positive pair 정합성 실패!")
        print("   → DataLoader에서 notice/company 순서가 어긋남")
        print("   → 동일한 샘플러/인덱스로 페어를 구성하세요")
    elif row_hit < 0.3 or col_hit < 0.3:
        print("   ⚠️  WARNING: Positive pair 정합성 부족")
        print("   → 배치 내 positive 비율 확인 필요")
    else:
        print("   ✅ Positive pair 정합성 양호")
    print()

# 메서드를 클래스에 바인딩
TwoTowerTrainTask._verify_positive_pair_alignment = _verify_positive_pair_alignment