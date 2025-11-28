"""
TwoTowerModelV2 - EmbeddingBagCollection 기반

기존 TwoTowerModel과 인터페이스 호환성 유지하면서 성능 개선
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from torchrec.modules.embedding_configs import PoolingType

from src.towers.tower.notice_tower import NoticeTower
from src.towers.tower.company_tower import CompanyTower


class TwoTowerModel(nn.Module):
    """
    Two-Tower Model - EmbeddingBagCollection 기반

    기존 TwoTowerModel과 인터페이스 호환성 유지
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
        """
        Args:
            metadata_path: metadata.csv 경로
            categorical_embedding_dim: 범주형 임베딩 차원
            dense_input_dim: dense 입력 차원
            tower_hidden_dims: MLP 히든 레이어 차원
            final_embedding_dim: 최종 출력 임베딩 차원
            dropout_rate: 드롭아웃 비율
            pooling_mode: EmbeddingBag pooling 방식
            device: GPU device (DDP 시 local_rank 기반)
            use_fp16: FP16 연산 사용 여부
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device

        # Notice Tower
        self.notice_tower = NoticeTower(
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

        # Company Tower
        self.company_tower = CompanyTower(
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

        print(f"\n{'='*80}")
        print(f"TwoTowerModel 초기화 완료 (EmbeddingBagCollection 기반)")
        print(f"{'='*80}")

    def forward(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            batch: {
                "notice": {"dense": Tensor, "kjt": KeyedJaggedTensor},
                "company": {"dense": Tensor, "kjt": KeyedJaggedTensor}
            }

        Returns:
            {
                "notice_embeddings": Tensor [B, final_embedding_dim],
                "company_embeddings": Tensor [B, final_embedding_dim]
            }
        """
        notice_embeddings = self.notice_tower(batch["notice"])
        company_embeddings = self.company_tower(batch["company"])

        return {
            "notice_embeddings": notice_embeddings,
            "company_embeddings": company_embeddings
        }

    def get_notice_embeddings(self, notice_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Notice 임베딩만 추출 (inference/벡터화)"""
        return self.notice_tower(notice_input)

    def get_company_embeddings(self, company_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Company 임베딩만 추출 (inference/벡터화)"""
        return self.company_tower(company_input)


if __name__ == "__main__":
    print("=== TwoTowerModel 테스트 ===\n")

    # 1. 모델 생성
    print("1. 모델 생성 중...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TwoTowerModel(
        metadata_path="meta/metadata.csv",
        categorical_embedding_dim=32,
        dense_input_dim=256,
        tower_hidden_dims=[256, 128],
        final_embedding_dim=128,
        dropout_rate=0.2,
        pooling_mode=PoolingType.MEAN,
        device=device,
        use_fp16=False  # 테스트 시 FP32
    )

    # 2. 더미 배치 생성
    print("\n2. 더미 배치 생성...")
    B = 4

    # Notice KJT
    from src.towers.kjt_utils import create_kjt_from_batch_gpu

    notice_cat_dummy = torch.randint(
        0, 100, (B, len(model.notice_tower.categorical_keys)), device=device
    )
    notice_kjt = create_kjt_from_batch_gpu(
        notice_cat_dummy, model.notice_tower.categorical_keys, device
    )

    # Company KJT
    company_cat_dummy = torch.randint(
        0, 100, (B, len(model.company_tower.categorical_keys)), device=device
    )
    company_kjt = create_kjt_from_batch_gpu(
        company_cat_dummy, model.company_tower.categorical_keys, device
    )

    batch = {
        "notice": {
            "dense": torch.randn(B, 256, device=device),
            "kjt": notice_kjt
        },
        "company": {
            "dense": torch.randn(B, 256, device=device),
            "kjt": company_kjt
        }
    }

    print(f"   Notice dense: {batch['notice']['dense'].shape}")
    print(f"   Notice KJT keys: {len(notice_kjt.keys())}")
    print(f"   Company dense: {batch['company']['dense'].shape}")
    print(f"   Company KJT keys: {len(company_kjt.keys())}")

    # 3. Forward pass
    print("\n3. Forward pass 테스트...")
    with torch.no_grad():
        outputs = model(batch)

    print(f"   Notice embeddings: {outputs['notice_embeddings'].shape}")
    print(f"   Company embeddings: {outputs['company_embeddings'].shape}")

    # L2 norm 확인 (모두 1이어야 함)
    notice_norms = torch.norm(outputs['notice_embeddings'], p=2, dim=1)
    company_norms = torch.norm(outputs['company_embeddings'], p=2, dim=1)

    print(f"\n   Notice L2 norms: {notice_norms}")
    print(f"   Company L2 norms: {company_norms}")

    assert torch.allclose(notice_norms, torch.ones_like(notice_norms), atol=1e-5)
    assert torch.allclose(company_norms, torch.ones_like(company_norms), atol=1e-5)
    print("   ✅ L2 정규화 검증 완료")

    # 4. 파라미터 수 확인
    print("\n4. 모델 파라미터 분석...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   총 파라미터: {total_params:,}")
    print(f"   학습 가능 파라미터: {trainable_params:,}")

    print("\n✅ TwoTowerModel 테스트 완료!")
