"""
BaseTowerV2 - EmbeddingBagCollection 기반

기존 BaseTower 대비 개선:
- CatEmbed 사용 (GPU kernel 최적화)
- Metadata 기반 자동 config 생성
- FP16 연산 지원
- stride=B 명시로 배치 차원 보장
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from torchrec import KeyedJaggedTensor
from torchrec.modules.embedding_configs import PoolingType

from src.towers.cat_embed import CatEmbed
from src.towers.embedding_config import (
    create_notice_embedding_configs,
    create_company_embedding_configs
)


class BaseTower(nn.Module):
    """
    BaseTower V2 - EmbeddingBagCollection 기반

    기존 BaseTower와 인터페이스 호환성 유지하면서 성능 개선
    """

    def __init__(
        self,
        table_name: str,
        metadata_path: str = "meta/metadata.csv",
        categorical_embedding_dim: int = 32,
        dense_input_dim: int = 256,
        tower_hidden_dims: Optional[List[int]] = None,
        final_embedding_dim: int = 128,
        dropout_rate: float = 0.2,
        pooling_mode: PoolingType = PoolingType.MEAN,
        device: Optional[torch.device] = None,
        use_fp16: bool = False,  # FP16 학습 지원
    ):
        """
        Args:
            table_name: "notice" or "company"
            metadata_path: metadata.csv 경로
            categorical_embedding_dim: 범주형 임베딩 차원
            dense_input_dim: dense 입력 차원
            tower_hidden_dims: MLP 히든 레이어 차원
            final_embedding_dim: 최종 출력 임베딩 차원
            dropout_rate: 드롭아웃 비율
            pooling_mode: EmbeddingBag pooling 방식
            device: GPU device (None이면 cuda:0)
            use_fp16: True면 FP16 연산 (AMP와 함께 사용)
        """
        super().__init__()

        if tower_hidden_dims is None:
            tower_hidden_dims = [256, 128]

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.table_name = table_name
        self.device = device
        self.use_fp16 = use_fp16

        # 1) EmbeddingBagConfig 생성 (metadata 기반)
        if table_name == "notice":
            embedding_configs = create_notice_embedding_configs(
                metadata_path=metadata_path,
                embedding_dim=categorical_embedding_dim,
                pooling_mode=pooling_mode,
                add_unk_token=True
            )
        elif table_name == "company":
            embedding_configs = create_company_embedding_configs(
                metadata_path=metadata_path,
                embedding_dim=categorical_embedding_dim,
                pooling_mode=pooling_mode,
                add_unk_token=True
            )
        else:
            raise ValueError(f"Unknown table_name: {table_name}")

        # 2) CatEmbed (EmbeddingBagCollection 기반)
        self.categorical_embedder = CatEmbed(
            embedding_configs=embedding_configs,
            device=device
        )

        # Feature names 저장 (디버깅용)
        self.categorical_keys = self.categorical_embedder.feature_names

        # 3) Dense 프로젝션
        self.dense_projection = nn.Linear(dense_input_dim, tower_hidden_dims[0]).to(device)

        # 4) MLP 구성
        categorical_total_dim = self.categorical_embedder.get_output_dim()
        self._build_mlp(
            categorical_total_dim=categorical_total_dim,
            tower_hidden_dims=tower_hidden_dims,
            final_embedding_dim=final_embedding_dim,
            dropout_rate=dropout_rate
        )

        # 모델 전체를 device로 이동
        self.to(device)

        print(f"\n✅ {table_name.capitalize()} TowerV2 초기화 완료:")
        print(f"   - 범주형 피처 수: {len(self.categorical_keys)}")
        print(f"   - 범주형 출력 차원: {categorical_total_dim}")
        print(f"   - Dense 입력 차원: {dense_input_dim}")
        print(f"   - 최종 출력 차원: {final_embedding_dim}")
        print(f"   - 디바이스: {device}")
        print(f"   - FP16 모드: {use_fp16}")

    def _check_tensor(self, name: str, x: torch.Tensor):
        """🔍 NaN/Inf 체크 헬퍼"""
        if x is None:
            return

        t = x.detach()
        if not torch.is_floating_point(t):
            return

        has_nan = torch.isnan(t).any()
        has_inf = torch.isinf(t).any()

        if has_nan or has_inf:
            # 가능하면 finite 값 기준으로 대략적인 통계도 같이 로그
            try:
                finite_mask = torch.isfinite(t)
                if finite_mask.any():
                    finite_vals = t[finite_mask]
                    t_min = float(finite_vals.min())
                    t_max = float(finite_vals.max())
                    t_mean = float(finite_vals.mean())
                    stats = f"min={t_min}, max={t_max}, mean={t_mean}"
                else:
                    stats = "no finite values"
            except Exception:
                stats = "stats_unavailable"

            raise RuntimeError(
                f"[{self.__class__.__name__}] NaN/Inf detected in '{name}'. "
                f"(has_nan={bool(has_nan)}, has_inf={bool(has_inf)}, {stats})"
            )

    def _build_mlp(
        self,
        categorical_total_dim: int,
        tower_hidden_dims: List[int],
        final_embedding_dim: int,
        dropout_rate: float
    ):
        """MLP 레이어 구성"""
        combined_input_dim = tower_hidden_dims[0] + categorical_total_dim

        mlp_layers = []
        input_dim = combined_input_dim

        # 히든 레이어들
        for hidden_dim in tower_hidden_dims[1:]:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # 🔁 BatchNorm1d → LayerNorm (NaN 문제 방지)
                nn.Dropout(dropout_rate),
            ])
            input_dim = hidden_dim

        # 최종 출력 레이어
        mlp_layers.append(nn.Linear(input_dim, final_embedding_dim))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, tower_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Args:
            tower_input: {
                "dense": Tensor [B, dense_dim] (FP16 or FP32),
                "kjt": KeyedJaggedTensor (GPU)
            }

        Returns:
            Tensor [B, final_embedding_dim]: L2 normalized embedding
        """
        dense = tower_input["dense"]
        kjt = tower_input["kjt"]

        # 디바이스 이동 (필요 시)
        if dense.device != self.device:
            dense = dense.to(self.device)
        if hasattr(kjt, 'to') and kjt.device != self.device:
            kjt = kjt.to(self.device)

        # 🔍 입력 dense 체크
        self._check_tensor("input_dense", dense)

        # 1) Dense 프로젝션
        dense_projected = self.dense_projection(dense)
        self._check_tensor("dense_projected", dense_projected)

        # 2) 범주형 임베딩 (EmbeddingBagCollection)
        categorical_combined = self.categorical_embedder(kjt)  # [B, cat_total_dim]
        self._check_tensor("categorical_combined", categorical_combined)

        # 3) 결합
        combined_features = torch.cat([dense_projected, categorical_combined], dim=1)
        self._check_tensor("combined_features", combined_features)

        # 4) MLP
        tower_embedding = self.mlp(combined_features)
        self._check_tensor("tower_embedding_before_norm", tower_embedding)

        # 5) L2 정규화
        tower_embedding = torch.nn.functional.normalize(tower_embedding, p=2, dim=1)
        self._check_tensor("tower_embedding_after_norm", tower_embedding)

        return tower_embedding
