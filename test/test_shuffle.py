#!/usr/bin/env python3
"""
Shuffle 동작 검증 테스트
- Chunk-level shuffle 검증
- Within-chunk shuffle 검증
- Epoch별 재현성 검증
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.database_connector import DatabaseConnector
from preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders


def collect_pair_ids(loader, max_batches=10):
    """DataLoader에서 pair ID 수집

    Returns:
        List of (bidntceno, bidntceord, bizno) tuples
    """
    pair_ids = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch_size = batch['notice']['dense'].shape[0]

        # Dataset에서 직접 pair 정보 조회
        dataset = loader.dataset
        start_idx = batch_idx * loader.batch_size

        for i in range(batch_size):
            idx = start_idx + i
            if idx >= len(dataset):
                break

            item = dataset[idx]
            pair_info = item.get('pair_info', {})

            pair_id = (
                pair_info.get('bidntceno'),
                pair_info.get('bidntceord'),
                pair_info.get('bizno')
            )
            pair_ids.append(pair_id)

    return pair_ids


def test_chunk_level_shuffle(pair_limit=100000, chunk_size=1000, num_epochs=3):
    """Chunk-level shuffle 검증

    검증 항목:
    1. Epoch마다 chunk 순서가 바뀌는지
    2. 같은 epoch를 재실행하면 같은 순서인지 (재현성)
    """
    print("=" * 80)
    print("TEST 1: Chunk-Level Shuffle 검증")
    print("=" * 80)

    # DB 연결
    db = DatabaseConnector()
    engine = db.engine

    # 스키마 구축
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # DataLoader 생성
    print(f"\n📊 설정:")
    print(f"   - Pair limit: {pair_limit:,}")
    print(f"   - Chunk size: {chunk_size:,}")
    print(f"   - Num epochs: {num_epochs}")
    print(f"   - Shuffle: True")
    print(f"   - Streaming: True")

    train_loader, _ = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=256,
        test_split=0.0,
        shuffle=True,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        chunk_size=chunk_size,
        feature_chunksize=10000,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=pair_limit,
    )

    dataset = train_loader.dataset
    num_chunks = dataset.num_chunks
    print(f"\n📦 총 chunk 수: {num_chunks}")

    # Epoch별 chunk order 수집
    epoch_chunk_orders = {}

    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)
        chunk_order = dataset.chunk_order.copy()
        epoch_chunk_orders[epoch] = chunk_order

        print(f"\n🔀 Epoch {epoch}:")
        print(f"   - Chunk order (first 20): {list(chunk_order[:20])}")

    # 검증 1: Epoch마다 다른 순서인지
    print("\n" + "=" * 80)
    print("검증 1: Epoch마다 chunk 순서가 다른가?")
    print("=" * 80)

    all_different = True
    for i in range(num_epochs - 1):
        for j in range(i + 1, num_epochs):
            order_i = epoch_chunk_orders[i]
            order_j = epoch_chunk_orders[j]

            if np.array_equal(order_i, order_j):
                print(f"❌ FAIL: Epoch {i}와 Epoch {j}의 chunk 순서가 동일함!")
                all_different = False
            else:
                # 얼마나 다른지 계산
                diff_ratio = np.mean(order_i != order_j)
                print(f"✅ PASS: Epoch {i} vs {j} - {diff_ratio*100:.1f}% 다름")

    if all_different:
        print("\n✅ 결과: 모든 epoch의 chunk 순서가 다릅니다!")
    else:
        print("\n❌ 결과: 일부 epoch의 chunk 순서가 동일합니다!")

    # 검증 2: 재현성 - 같은 epoch를 다시 설정하면 같은 순서인지
    print("\n" + "=" * 80)
    print("검증 2: 같은 epoch를 재설정하면 같은 순서인가? (재현성)")
    print("=" * 80)

    reproducible = True
    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)
        chunk_order_1 = dataset.chunk_order.copy()

        # 다른 epoch로 바꾸고 다시 돌아옴
        dataset.set_epoch((epoch + 1) % num_epochs)
        dataset.set_epoch(epoch)
        chunk_order_2 = dataset.chunk_order.copy()

        if np.array_equal(chunk_order_1, chunk_order_2):
            print(f"✅ PASS: Epoch {epoch} - 재현 가능")
        else:
            print(f"❌ FAIL: Epoch {epoch} - 재현 불가능!")
            reproducible = False

    if reproducible:
        print("\n✅ 결과: 모든 epoch에서 재현 가능합니다!")
    else:
        print("\n❌ 결과: 일부 epoch에서 재현 불가능합니다!")

    return all_different and reproducible


def test_within_chunk_shuffle(pair_limit=100000, chunk_size=1000, num_epochs=3):
    """Within-chunk shuffle 검증

    검증 항목:
    1. 같은 chunk 내 pair 순서가 epoch마다 바뀌는지
    2. 첫 번째 chunk의 첫 10개 pair가 epoch마다 다른지
    """
    print("\n" + "=" * 80)
    print("TEST 2: Within-Chunk Shuffle 검증")
    print("=" * 80)

    # DB 연결
    db = DatabaseConnector()
    engine = db.engine

    # 스키마 구축
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # DataLoader 생성
    train_loader, _ = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=256,
        test_split=0.0,
        shuffle=True,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        chunk_size=chunk_size,
        feature_chunksize=10000,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=pair_limit,
    )

    dataset = train_loader.dataset

    # Epoch별 첫 번째 배치의 pair 수집
    epoch_first_pairs = {}

    print(f"\n📋 각 epoch의 첫 10개 pair 수집 중...")

    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)

        # 첫 10개 pair 수집
        first_pairs = []
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            pair_info = item.get('pair_info', {})
            pair_id = (
                pair_info.get('bidntceno'),
                pair_info.get('bidntceord'),
                pair_info.get('bizno')
            )
            first_pairs.append(pair_id)

        epoch_first_pairs[epoch] = first_pairs

        print(f"\n🔀 Epoch {epoch} - 첫 10개 pair:")
        for idx, pair in enumerate(first_pairs):
            print(f"   [{idx}] {pair[0]}-{pair[1]} / {pair[2]}")

    # 검증: Epoch마다 첫 10개 pair가 다른지
    print("\n" + "=" * 80)
    print("검증: Epoch마다 첫 10개 pair 순서가 다른가?")
    print("=" * 80)

    all_different = True
    for i in range(num_epochs - 1):
        for j in range(i + 1, num_epochs):
            pairs_i = epoch_first_pairs[i]
            pairs_j = epoch_first_pairs[j]

            # 순서 비교
            if pairs_i == pairs_j:
                print(f"❌ FAIL: Epoch {i}와 Epoch {j}의 첫 10개 pair 순서가 동일함!")
                all_different = False
            else:
                # 얼마나 다른지 계산
                same_count = sum(1 for a, b in zip(pairs_i, pairs_j) if a == b)
                diff_count = len(pairs_i) - same_count
                print(f"✅ PASS: Epoch {i} vs {j} - {diff_count}/10개 다름 ({diff_count/10*100:.0f}%)")

    if all_different:
        print("\n✅ 결과: 모든 epoch의 첫 10개 pair 순서가 다릅니다!")
    else:
        print("\n❌ 결과: 일부 epoch의 첫 10개 pair 순서가 동일합니다!")

    return all_different


def test_chunk_physical_mapping(pair_limit=100000, chunk_size=1000):
    """Logical chunk → Physical chunk 매핑 검증

    검증 항목:
    1. Logical chunk 0이 실제로 다른 physical chunk를 가리키는지
    """
    print("\n" + "=" * 80)
    print("TEST 3: Logical → Physical Chunk 매핑 검증")
    print("=" * 80)

    # DB 연결
    db = DatabaseConnector()
    engine = db.engine

    # 스키마 구축
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # DataLoader 생성
    train_loader, _ = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=256,
        test_split=0.0,
        shuffle=True,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        chunk_size=chunk_size,
        feature_chunksize=10000,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=pair_limit,
    )

    dataset = train_loader.dataset

    print(f"\n📊 Logical chunk 0 → Physical chunk 매핑:")
    print(f"   (DataLoader는 항상 logical chunk 0부터 순차 접근)")

    epoch_mappings = {}

    for epoch in range(3):
        dataset.set_epoch(epoch)

        # Logical chunk 0 = chunk_order[0]
        physical_chunk_0 = dataset.chunk_order[0]
        epoch_mappings[epoch] = physical_chunk_0

        print(f"\n   Epoch {epoch}: Logical 0 → Physical {physical_chunk_0}")

    # 검증: Epoch마다 다른 physical chunk를 가리키는지
    print("\n" + "=" * 80)
    print("검증: Epoch마다 다른 physical chunk를 로딩하는가?")
    print("=" * 80)

    all_different = True
    unique_mappings = set(epoch_mappings.values())

    if len(unique_mappings) == len(epoch_mappings):
        print(f"✅ PASS: 모든 epoch에서 다른 physical chunk 사용 ({unique_mappings})")
    else:
        print(f"❌ FAIL: 일부 epoch에서 같은 physical chunk 재사용!")
        all_different = False

    return all_different


def test_no_shuffle_baseline(pair_limit=100000, chunk_size=1000):
    """Shuffle=False 기준선 테스트

    검증 항목:
    1. shuffle=False일 때 epoch마다 같은 순서인지
    """
    print("\n" + "=" * 80)
    print("TEST 4: Shuffle=False 기준선 검증")
    print("=" * 80)

    # DB 연결
    db = DatabaseConnector()
    engine = db.engine

    # 스키마 구축
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # DataLoader 생성 (shuffle=False)
    train_loader, _ = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=256,
        test_split=0.0,
        shuffle=False,  # CRITICAL: shuffle OFF
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        chunk_size=chunk_size,
        feature_chunksize=10000,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=pair_limit,
    )

    dataset = train_loader.dataset

    # Epoch별 첫 10개 pair 수집
    epoch_first_pairs = {}

    print(f"\n📋 각 epoch의 첫 10개 pair 수집 중 (shuffle=False)...")

    for epoch in range(3):
        dataset.set_epoch(epoch)

        # 첫 10개 pair 수집
        first_pairs = []
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            pair_info = item.get('pair_info', {})
            pair_id = (
                pair_info.get('bidntceno'),
                pair_info.get('bidntceord'),
                pair_info.get('bizno')
            )
            first_pairs.append(pair_id)

        epoch_first_pairs[epoch] = first_pairs

        print(f"\n   Epoch {epoch} - 첫 10개 pair:")
        for idx, pair in enumerate(first_pairs[:5]):  # 처음 5개만
            print(f"      [{idx}] {pair[0]}-{pair[1]} / {pair[2]}")

    # 검증: 모든 epoch에서 같은 순서인지
    print("\n" + "=" * 80)
    print("검증: shuffle=False일 때 모든 epoch에서 같은 순서인가?")
    print("=" * 80)

    all_same = True
    baseline = epoch_first_pairs[0]

    for epoch in range(1, 3):
        if epoch_first_pairs[epoch] == baseline:
            print(f"✅ PASS: Epoch {epoch}은 Epoch 0과 동일한 순서")
        else:
            print(f"❌ FAIL: Epoch {epoch}은 Epoch 0과 다른 순서!")
            all_same = False

    if all_same:
        print("\n✅ 결과: shuffle=False일 때 모든 epoch에서 같은 순서입니다!")
    else:
        print("\n❌ 결과: shuffle=False인데 순서가 바뀌었습니다!")

    return all_same


def test_dataloader_iteration_shuffle(pair_limit=10000, chunk_size=500, num_epochs=2):
    """DataLoader iteration 중 shuffle 검증

    검증 항목:
    1. Epoch마다 DataLoader에서 받는 batch 순서가 다른지
    2. 실제 학습 루프처럼 iteration하면서 pair 순서 추적
    3. Chunk boundary를 넘나들며 shuffle이 유지되는지
    """
    print("\n" + "=" * 80)
    print("TEST 5: DataLoader Iteration Shuffle 검증 (실제 학습 시뮬레이션)")
    print("=" * 80)

    # DB 연결
    db = DatabaseConnector()
    engine = db.engine

    # 스키마 구축
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # DataLoader 생성
    print(f"\n📊 설정:")
    print(f"   - Pair limit: {pair_limit:,}")
    print(f"   - Chunk size: {chunk_size:,}")
    print(f"   - Batch size: 128")
    print(f"   - Num epochs: {num_epochs}")
    print(f"   - Shuffle: True")

    train_loader, _ = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=128,
        test_split=0.0,
        shuffle=True,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        chunk_size=chunk_size,
        feature_chunksize=10000,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=pair_limit,
    )

    dataset = train_loader.dataset

    # Epoch별 batch 추적
    epoch_batch_samples = {}

    print(f"\n📋 각 epoch에서 DataLoader iteration 중...")

    for epoch in range(num_epochs):
        # CRITICAL: set_epoch 호출 (실제 학습과 동일)
        dataset.set_epoch(epoch)

        batch_samples = []

        # DataLoader에서 처음 5개 배치만 수집
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:
                break

            # 배치에서 샘플 추출 (dataset 직접 접근)
            batch_size = batch['notice']['dense'].shape[0]
            start_idx = batch_idx * train_loader.batch_size

            for i in range(min(3, batch_size)):  # 각 배치에서 처음 3개만
                idx = start_idx + i
                if idx >= len(dataset):
                    break

                item = dataset[idx]
                pair_info = item.get('pair_info', {})
                pair_id = (
                    pair_info.get('bidntceno'),
                    pair_info.get('bidntceord'),
                    pair_info.get('bizno')
                )
                batch_samples.append((batch_idx, i, pair_id))

        epoch_batch_samples[epoch] = batch_samples

        print(f"\n🔀 Epoch {epoch} - 처음 5 배치의 처음 3개 샘플:")
        for batch_idx, local_idx, pair_id in batch_samples[:9]:  # 처음 9개만 출력
            print(f"   Batch[{batch_idx}][{local_idx}]: {pair_id[0]}-{pair_id[1]} / {pair_id[2]}")

    # 검증: Epoch마다 다른 순서인지
    print("\n" + "=" * 80)
    print("검증: Epoch마다 배치 내 샘플 순서가 다른가?")
    print("=" * 80)

    all_different = True
    for i in range(num_epochs - 1):
        samples_i = [pair for _, _, pair in epoch_batch_samples[i]]
        samples_j = [pair for _, _, pair in epoch_batch_samples[i + 1]]

        # 순서 비교
        if samples_i == samples_j:
            print(f"❌ FAIL: Epoch {i}와 Epoch {i+1}의 배치 샘플 순서가 동일함!")
            all_different = False
        else:
            # 얼마나 다른지 계산
            same_count = sum(1 for a, b in zip(samples_i, samples_j) if a == b)
            diff_count = len(samples_i) - same_count
            print(f"✅ PASS: Epoch {i} vs {i+1} - {diff_count}/{len(samples_i)}개 다름 ({diff_count/len(samples_i)*100:.0f}%)")

    if all_different:
        print("\n✅ 결과: 모든 epoch에서 배치 샘플 순서가 다릅니다!")
    else:
        print("\n❌ 결과: 일부 epoch에서 배치 샘플 순서가 동일합니다!")

    return all_different


def test_chunk_boundary_crossing(pair_limit=5000, chunk_size=1000, num_epochs=2):
    """Chunk boundary를 넘나들며 shuffle 검증

    검증 항목:
    1. Chunk 0의 마지막 샘플 → Chunk 1의 첫 샘플로 넘어갈 때 순서가 바뀌는지
    2. Epoch마다 chunk boundary에서의 pair가 다른지
    """
    print("\n" + "=" * 80)
    print("TEST 6: Chunk Boundary Crossing Shuffle 검증")
    print("=" * 80)

    # DB 연결
    db = DatabaseConnector()
    engine = db.engine

    # 스키마 구축
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # DataLoader 생성
    print(f"\n📊 설정:")
    print(f"   - Pair limit: {pair_limit:,}")
    print(f"   - Chunk size: {chunk_size:,}")
    print(f"   - Expected chunks: {pair_limit // chunk_size}")

    train_loader, _ = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=128,
        test_split=0.0,
        shuffle=True,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        chunk_size=chunk_size,
        feature_chunksize=10000,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=pair_limit,
    )

    dataset = train_loader.dataset

    # Epoch별 chunk boundary 샘플 수집
    epoch_boundary_samples = {}

    print(f"\n📋 각 epoch에서 chunk boundary 샘플 수집 중...")

    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)

        boundary_samples = []

        # 첫 2개 chunk의 경계 샘플 수집
        for chunk_idx in range(min(2, dataset.num_chunks)):
            # Chunk 끝 3개
            chunk_end_start = (chunk_idx + 1) * chunk_size - 3
            for offset in range(3):
                idx = chunk_end_start + offset
                if idx >= len(dataset):
                    break

                item = dataset[idx]
                pair_info = item.get('pair_info', {})
                pair_id = (
                    pair_info.get('bidntceno'),
                    pair_info.get('bidntceord'),
                    pair_info.get('bizno')
                )
                boundary_samples.append((f"Chunk{chunk_idx}_end", idx, pair_id))

            # 다음 Chunk 시작 3개
            if chunk_idx + 1 < dataset.num_chunks:
                chunk_start = (chunk_idx + 1) * chunk_size
                for offset in range(3):
                    idx = chunk_start + offset
                    if idx >= len(dataset):
                        break

                    item = dataset[idx]
                    pair_info = item.get('pair_info', {})
                    pair_id = (
                        pair_info.get('bidntceno'),
                        pair_info.get('bidntceord'),
                        pair_info.get('bizno')
                    )
                    boundary_samples.append((f"Chunk{chunk_idx+1}_start", idx, pair_id))

        epoch_boundary_samples[epoch] = boundary_samples

        print(f"\n🔀 Epoch {epoch} - Chunk boundary 샘플:")
        for label, idx, pair_id in boundary_samples[:12]:  # 처음 12개만 출력
            print(f"   {label}[{idx}]: {pair_id[0]}-{pair_id[1]} / {pair_id[2]}")

    # 검증: Epoch마다 chunk boundary 샘플이 다른지
    print("\n" + "=" * 80)
    print("검증: Epoch마다 chunk boundary 샘플이 다른가?")
    print("=" * 80)

    all_different = True
    for i in range(num_epochs - 1):
        samples_i = [pair for _, _, pair in epoch_boundary_samples[i]]
        samples_j = [pair for _, _, pair in epoch_boundary_samples[i + 1]]

        if samples_i == samples_j:
            print(f"❌ FAIL: Epoch {i}와 Epoch {i+1}의 chunk boundary 샘플이 동일함!")
            all_different = False
        else:
            same_count = sum(1 for a, b in zip(samples_i, samples_j) if a == b)
            diff_count = len(samples_i) - same_count
            print(f"✅ PASS: Epoch {i} vs {i+1} - {diff_count}/{len(samples_i)}개 다름 ({diff_count/len(samples_i)*100:.0f}%)")

    if all_different:
        print("\n✅ 결과: Chunk boundary에서도 shuffle이 올바르게 작동합니다!")
    else:
        print("\n❌ 결과: Chunk boundary에서 shuffle이 작동하지 않습니다!")

    return all_different


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 80)
    print("🧪 2-Level Shuffle 검증 테스트 시작")
    print("=" * 80)

    # 테스트 설정
    pair_limit = 100000
    chunk_size = 1000
    num_epochs = 3

    print(f"\n⚙️ 테스트 설정:")
    print(f"   - Pair limit: {pair_limit:,}")
    print(f"   - Chunk size: {chunk_size:,}")
    print(f"   - Num epochs: {num_epochs}")
    print(f"   - Expected num chunks: ~{pair_limit // chunk_size}")

    # 테스트 실행
    results = {}

    try:
        results['chunk_level'] = test_chunk_level_shuffle(pair_limit, chunk_size, num_epochs)
    except Exception as e:
        print(f"\n❌ TEST 1 실패: {e}")
        import traceback
        traceback.print_exc()
        results['chunk_level'] = False

    try:
        results['within_chunk'] = test_within_chunk_shuffle(pair_limit, chunk_size, num_epochs)
    except Exception as e:
        print(f"\n❌ TEST 2 실패: {e}")
        import traceback
        traceback.print_exc()
        results['within_chunk'] = False

    try:
        results['mapping'] = test_chunk_physical_mapping(pair_limit, chunk_size)
    except Exception as e:
        print(f"\n❌ TEST 3 실패: {e}")
        import traceback
        traceback.print_exc()
        results['mapping'] = False

    try:
        results['baseline'] = test_no_shuffle_baseline(pair_limit, chunk_size)
    except Exception as e:
        print(f"\n❌ TEST 4 실패: {e}")
        import traceback
        traceback.print_exc()
        results['baseline'] = False

    # 새로운 테스트: 실제 DataLoader iteration
    try:
        results['dataloader_iteration'] = test_dataloader_iteration_shuffle(
            pair_limit=10000, chunk_size=500, num_epochs=2
        )
    except Exception as e:
        print(f"\n❌ TEST 5 실패: {e}")
        import traceback
        traceback.print_exc()
        results['dataloader_iteration'] = False

    # 새로운 테스트: Chunk boundary crossing
    try:
        results['chunk_boundary'] = test_chunk_boundary_crossing(
            pair_limit=5000, chunk_size=1000, num_epochs=2
        )
    except Exception as e:
        print(f"\n❌ TEST 6 실패: {e}")
        import traceback
        traceback.print_exc()
        results['chunk_boundary'] = False

    # 최종 결과
    print("\n" + "=" * 80)
    print("🎯 최종 테스트 결과")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print(f"\n📊 통과율: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.0f}%)")

    if passed_tests == total_tests:
        print("\n🎉 모든 테스트 통과! 2-Level Shuffle이 올바르게 구현되었습니다!")
        return 0
    else:
        print("\n⚠️ 일부 테스트 실패. 코드를 다시 확인하세요.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
