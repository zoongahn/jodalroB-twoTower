#!/usr/bin/env python3
"""
EmbeddingBagCollection Sanity Test
출력이 0인 문제 원인 파악
"""

import torch
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_1_minimal_ebc():
    """최소 단위 EBC 테스트 - 정상 동작 확인"""
    print("=" * 80)
    print("Test 1: 최소 단위 EBC (단일 키)")
    print("=" * 80)

    # 단일 키 EBC
    cfg = EmbeddingBagConfig(
        name="test_ebc",
        embedding_dim=8,
        num_embeddings=10,
        feature_names=["foo"],
        pooling=PoolingType.MEAN
    )

    ebc = EmbeddingBagCollection(tables=[cfg], device="cuda")

    # Weight 수동 초기화
    print(f"\n1. Weight 초기화 전:")
    print(f"   평균: {ebc.embedding_bags['test_ebc'].weight.abs().mean().item():.6f}")
    print(f"   표준편차: {ebc.embedding_bags['test_ebc'].weight.std().item():.6f}")

    with torch.no_grad():
        ebc.embedding_bags["test_ebc"].weight.normal_(0, 0.1)

    print(f"\n2. Weight 초기화 후 (normal_(0, 0.1)):")
    print(f"   평균: {ebc.embedding_bags['test_ebc'].weight.abs().mean().item():.6f}")
    print(f"   표준편차: {ebc.embedding_bags['test_ebc'].weight.std().item():.6f}")
    print(f"   requires_grad: {ebc.embedding_bags['test_ebc'].weight.requires_grad}")

    # KJT 생성 (배치 2, 각 샘플 1개 값)
    values = torch.tensor([1, 3], device="cuda", dtype=torch.long)
    lengths = torch.tensor([1, 1], device="cuda", dtype=torch.int32)

    kjt = KeyedJaggedTensor(
        keys=["foo"],
        values=values,
        lengths=lengths
    )

    print(f"\n3. KJT 생성:")
    print(f"   keys: {kjt.keys()}")
    print(f"   values: {values} (device: {values.device})")
    print(f"   lengths: {lengths}")

    # Forward pass
    out = ebc(kjt)

    print(f"\n4. Forward 결과:")
    print(f"   출력 keys: {list(out.keys())}")

    # KeyedTensor는 feature_name으로 접근
    output_tensor = out["foo"]  # feature_name 사용

    print(f"   출력 shape: {output_tensor.shape}")
    print(f"   출력 값 합계: {output_tensor.abs().sum().item():.6f}")
    print(f"   출력 평균: {output_tensor.mean().item():.6f}")
    print(f"   출력: {output_tensor}")

    if output_tensor.abs().sum().item() > 0:
        print("\n✅ Test 1 PASS: EBC 자체는 정상 동작")
    else:
        print("\n❌ Test 1 FAIL: EBC 자체에 문제 있음")

    return output_tensor.abs().sum().item() > 0


def test_2_catembedv2_weights():
    """CatEmbedV2의 weight 초기화 확인"""
    print("\n" + "=" * 80)
    print("Test 2: CatEmbedV2 Weight 초기화")
    print("=" * 80)

    from src.towers.cat_embed_v2 import CatEmbedV2
    from src.towers.embedding_config import create_notice_embedding_configs

    # Notice configs (처음 3개만)
    all_configs = create_notice_embedding_configs(
        embedding_dim=32,
        pooling_mode=PoolingType.MEAN,
        add_unk_token=True
    )
    configs = all_configs[:3]  # 처음 3개만 테스트

    print(f"\n1. Config 정보:")
    for cfg in configs:
        print(f"   - name: {cfg.name}")
        print(f"     feature_names: {cfg.feature_names}")
        print(f"     num_embeddings: {cfg.num_embeddings}")
        print(f"     embedding_dim: {cfg.embedding_dim}")

    # CatEmbedV2 생성
    model = CatEmbedV2(embedding_configs=configs, device=torch.device("cuda"))

    print(f"\n2. Weight 통계:")
    for name, param in model.named_parameters():
        print(f"   {name}:")
        print(f"     - shape: {param.shape}")
        print(f"     - requires_grad: {param.requires_grad}")
        print(f"     - mean: {param.mean().item():.6f}")
        print(f"     - std: {param.std().item():.6f}")
        print(f"     - abs_mean: {param.abs().mean().item():.6f}")

    # Weight가 모두 0인지 확인
    total_abs_mean = sum(p.abs().mean().item() for p in model.parameters())
    print(f"\n3. 전체 파라미터 절대값 평균 합: {total_abs_mean:.6f}")

    if total_abs_mean > 0:
        print("\n✅ Test 2 PASS: Weight가 정상 초기화됨")
    else:
        print("\n❌ Test 2 FAIL: Weight가 모두 0! 초기화 문제")

    return total_abs_mean > 0


def test_3_kjt_key_matching():
    """KJT keys와 EmbeddingBagConfig feature_names 매칭 테스트"""
    print("\n" + "=" * 80)
    print("Test 3: KJT Keys ↔ EmbeddingBagConfig 매칭")
    print("=" * 80)

    from src.towers.embedding_config import create_notice_embedding_configs

    configs = create_notice_embedding_configs(embedding_dim=32)[:3]

    print(f"\n1. EmbeddingBagConfig 정보:")
    for cfg in configs:
        print(f"   Config name: {cfg.name}")
        print(f"   Feature names: {cfg.feature_names}")
        print(f"   → KJT에서 사용할 key: {cfg.feature_names[0]}")

    # KJT keys 생성 (테스트 코드와 동일)
    kjt_keys = [cfg.feature_names[0] for cfg in configs]

    print(f"\n2. KJT keys: {kjt_keys}")

    # 매칭 확인
    all_match = True
    for cfg in configs:
        expected_key = cfg.feature_names[0]
        if expected_key not in kjt_keys:
            print(f"   ❌ MISMATCH: {expected_key} not in KJT keys")
            all_match = False

    if all_match:
        print("\n✅ Test 3 PASS: 모든 키가 매칭됨")
    else:
        print("\n❌ Test 3 FAIL: 키 불일치 발견")

    return all_match


def test_4_kjt_lengths_validation():
    """KJT lengths/values 정합성 검증"""
    print("\n" + "=" * 80)
    print("Test 4: KJT Lengths/Values 정합성")
    print("=" * 80)

    from src.towers.embedding_config import create_notice_embedding_configs

    configs = create_notice_embedding_configs(embedding_dim=32)[:3]
    batch_size = 4
    device = torch.device("cuda")

    # KJT 생성 (테스트 코드 방식)
    keys = [cfg.feature_names[0] for cfg in configs]
    values_list = []
    lengths_list = []

    for cfg in configs:
        batch_values = torch.randint(0, cfg.num_embeddings, (batch_size,), device=device)
        values_list.append(batch_values)
        lengths_list.append(torch.ones(batch_size, dtype=torch.int32, device=device))

    all_values = torch.cat(values_list, dim=0)
    all_lengths = torch.cat(lengths_list, dim=0)

    print(f"\n1. KJT 구성:")
    print(f"   Keys: {len(keys)}개")
    print(f"   Batch size: {batch_size}")
    print(f"   Values shape: {all_values.shape}")
    print(f"   Lengths shape: {all_lengths.shape}")

    # 정합성 체크
    expected_total = len(keys) * batch_size
    print(f"\n2. 정합성 체크:")
    print(f"   예상 총 값 개수: {expected_total}")
    print(f"   실제 values 개수: {all_values.shape[0]}")
    print(f"   실제 lengths 개수: {all_lengths.shape[0]}")
    print(f"   Lengths 합계: {all_lengths.sum().item()}")

    # KJT 생성
    kjt = KeyedJaggedTensor(
        keys=keys,
        values=all_values,
        lengths=all_lengths
    )

    print(f"\n3. KJT 정보:")
    print(f"   stride: {kjt.stride()}")  # batch size
    print(f"   keys: {kjt.keys()}")

    is_valid = (
        all_values.shape[0] == expected_total and
        all_lengths.shape[0] == expected_total and
        all_lengths.sum().item() == expected_total
    )

    if is_valid:
        print("\n✅ Test 4 PASS: KJT 정합성 정상")
    else:
        print("\n❌ Test 4 FAIL: KJT 정합성 문제")

    return is_valid


def test_5_full_catembedv2_forward():
    """CatEmbedV2 전체 forward 테스트 (수동 weight 초기화)"""
    print("\n" + "=" * 80)
    print("Test 5: CatEmbedV2 Full Forward (Weight 수동 초기화)")
    print("=" * 80)

    from src.towers.cat_embed_v2 import CatEmbedV2
    from src.towers.embedding_config import create_notice_embedding_configs

    configs = create_notice_embedding_configs(embedding_dim=32)[:3]
    device = torch.device("cuda")

    # 모델 생성
    model = CatEmbedV2(embedding_configs=configs, device=device)

    # ✅ CRITICAL: Weight 수동 초기화
    print("\n1. Weight 수동 초기화...")
    for name, param in model.named_parameters():
        with torch.no_grad():
            param.normal_(0, 0.1)
        print(f"   {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

    # KJT 생성
    batch_size = 4
    keys = [cfg.feature_names[0] for cfg in configs]
    values_list = []
    lengths_list = []

    for cfg in configs:
        batch_values = torch.randint(0, cfg.num_embeddings, (batch_size,), device=device)
        values_list.append(batch_values)
        lengths_list.append(torch.ones(batch_size, dtype=torch.int32, device=device))

    all_values = torch.cat(values_list, dim=0)
    all_lengths = torch.cat(lengths_list, dim=0)

    kjt = KeyedJaggedTensor(keys=keys, values=all_values, lengths=all_lengths)

    print(f"\n2. Forward pass...")
    with torch.no_grad():
        output = model(kjt)

    print(f"\n3. 출력 결과:")
    print(f"   Shape: {output.shape}")
    print(f"   평균: {output.mean().item():.6f}")
    print(f"   표준편차: {output.std().item():.6f}")
    print(f"   절대값 평균: {output.abs().mean().item():.6f}")
    print(f"   최소값: {output.min().item():.6f}")
    print(f"   최대값: {output.max().item():.6f}")

    if output.abs().sum().item() > 0:
        print("\n✅ Test 5 PASS: Forward 정상 (출력 > 0)")
    else:
        print("\n❌ Test 5 FAIL: Forward 결과가 모두 0!")

    return output.abs().sum().item() > 0


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EmbeddingBagCollection 정확성 점검")
    print("=" * 80 + "\n")

    results = {}

    # Test 1: 최소 EBC
    results["minimal_ebc"] = test_1_minimal_ebc()

    # Test 2: CatEmbedV2 weight
    results["catembedv2_weights"] = test_2_catembedv2_weights()

    # Test 3: Key matching
    results["key_matching"] = test_3_kjt_key_matching()

    # Test 4: KJT lengths
    results["kjt_lengths"] = test_4_kjt_lengths_validation()

    # Test 5: Full forward
    results["full_forward"] = test_5_full_catembedv2_forward()

    # 요약
    print("\n" + "=" * 80)
    print("테스트 요약")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 모든 테스트 통과! EBC 정상 동작")
    else:
        print("⚠️  일부 테스트 실패 - 위 로그 확인 필요")
    print("=" * 80)
