"""
CatEmbedV2 - TorchRec EmbeddingBagCollection 기반 범주형 임베딩

기존 CatEmbed와의 차이점:
1. EmbeddingBagCollection 사용 → GPU kernel 최적화
2. KeyedJaggedTensor 직접 입력
3. Pooling이 내부에서 자동 처리
4. Python loop 제거
"""

import torch
import torch.nn as nn
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor, KeyedTensor
from typing import List, Optional


class CatEmbed(nn.Module):
    """
    TorchRec EmbeddingBagCollection 기반 범주형 임베딩 레이어

    기존 CatEmbed 대비 개선:
    - ✅ GPU kernel로 embedding lookup + pooling 한 번에 처리
    - ✅ Python loop 제거
    - ✅ KJT 생성 오버헤드 감소
    - ✅ 배치 단위 최적화
    """

    def __init__(
        self,
        embedding_configs: List,  # List[EmbeddingBagConfig]
        device: Optional[torch.device] = None
    ):
        """
        Args:
            embedding_configs: EmbeddingBagConfig 리스트
            device: 디바이스 (기본값: cuda:0 if available)
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.embedding_configs = embedding_configs

        # EmbeddingBagCollection 생성
        self.ebc = EmbeddingBagCollection(
            tables=embedding_configs,
            device=device
        )

        # 출력 차원 계산 (모든 임베딩 차원의 합)
        self.output_dim = sum(cfg.embedding_dim for cfg in embedding_configs)

        # Feature 순서 저장 (concat 시 필요)
        # CRITICAL: pooled_features의 key는 feature_names를 사용!
        self.feature_names = [cfg.feature_names[0] for cfg in embedding_configs]
        self.embedding_configs = embedding_configs

        print(f"CatEmbedV2 초기화 완료:")
        print(f"  - 피처 개수: {len(embedding_configs)}")
        print(f"  - 출력 차원: {self.output_dim}")
        print(f"  - 디바이스: {device}")

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        """
        Forward pass - KeyedJaggedTensor → Pooled Embeddings

        Args:
            kjt: KeyedJaggedTensor (sparse categorical features)
                - keys: feature names
                - values: category indices
                - lengths: number of values per sample per feature

        Returns:
            pooled_embeddings: [B, output_dim] tensor
                모든 범주형 피처의 임베딩을 concat한 결과
        """
        # EmbeddingBagCollection forward
        # 반환값: KeyedTensor (key → embedding tensor 매핑)
        pooled_features: KeyedTensor = self.ebc(kjt)

        # KeyedTensor → Tensor 변환 (feature 순서대로 concat)
        values_list = []
        for name in self.feature_names:
            if name in pooled_features.keys():
                values_list.append(pooled_features[name])
            else:
                # 해당 feature가 없는 경우 (불가능하지만 안전장치)
                batch_size = kjt.stride()
                zeros = torch.zeros(
                    batch_size,
                    self._get_embedding_dim(name),
                    device=self.device
                )
                values_list.append(zeros)

        # 모든 임베딩 concat
        return torch.cat(values_list, dim=1)  # [B, sum(embedding_dims)]

    def _get_embedding_dim(self, feature_name: str) -> int:
        """특정 feature의 embedding_dim 반환"""
        for cfg in self.embedding_configs:
            if cfg.name == feature_name:
                return cfg.embedding_dim
        return 0

    def get_output_dim(self) -> int:
        """출력 차원 반환"""
        return self.output_dim

    def extra_repr(self) -> str:
        """모델 정보 문자열"""
        return f"num_features={len(self.feature_names)}, output_dim={self.output_dim}"


if __name__ == "__main__":
    # 테스트 코드
    from src.towers.embedding_config import create_notice_embedding_configs
    from torchrec.modules.embedding_configs import PoolingType

    print("=== CatEmbedV2 테스트 ===\n")

    # 1. EmbeddingBagConfig 생성
    print("1. EmbeddingBagConfig 생성...")
    configs = create_notice_embedding_configs(
        embedding_dim=32,
        pooling_mode=PoolingType.MEAN,
        add_unk_token=True
    )
    print(f"   생성된 config 개수: {len(configs)}\n")

    # 2. CatEmbedV2 모델 생성
    print("2. CatEmbedV2 모델 생성...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CatEmbedV2(
        embedding_configs=configs,
        device=device
    )
    print(f"   출력 차원: {model.get_output_dim()}\n")

    # 3. 더미 KJT 생성
    print("3. 더미 KeyedJaggedTensor 생성...")
    batch_size = 4
    device = model.device

    # 각 feature별로 랜덤 인덱스 생성
    # CRITICAL: KJT keys는 feature_names를 사용해야 함!
    keys = [cfg.feature_names[0] for cfg in configs]  # feature_names 사용
    values = []
    lengths = []

    for cfg in configs:
        # 각 샘플당 1개 값 (길이 1)
        batch_values = torch.randint(
            0, cfg.num_embeddings,
            (batch_size,),
            device=device
        )
        values.append(batch_values)
        lengths.append(torch.ones(batch_size, dtype=torch.int32, device=device))

    # KJT 생성
    all_values = torch.cat(values, dim=0)
    all_lengths = torch.cat(lengths, dim=0)

    kjt = KeyedJaggedTensor(
        keys=keys,
        values=all_values,
        lengths=all_lengths
    )

    print(f"   배치 크기: {batch_size}")
    print(f"   Keys: {len(keys)} 개")
    print(f"   Keys 예시 (처음 3개): {keys[:3]}")
    print(f"   Values shape: {all_values.shape}")
    print(f"   Lengths shape: {all_lengths.shape}\n")

    # 4. Forward pass
    print("4. Forward pass 테스트...")
    with torch.no_grad():
        output = model(kjt)

    print(f"   출력 shape: {output.shape}")
    print(f"   예상 shape: ({batch_size}, {model.get_output_dim()})")
    print(f"   Shape 일치: {output.shape == (batch_size, model.get_output_dim())}")
    print(f"   출력 범위: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"   디바이스: {output.device}\n")

    # 5. 파라미터 수 확인
    print("5. 모델 파라미터 분석...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   총 파라미터 수: {total_params:,}")

    # EmbeddingBagCollection의 각 테이블별 파라미터
    expected_params = sum(cfg.num_embeddings * cfg.embedding_dim for cfg in configs)
    print(f"   예상 파라미터 수: {expected_params:,}")
    print(f"   일치 여부: {total_params == expected_params}")

    print("\n✅ CatEmbedV2 테스트 완료!")
