#!/usr/bin/env python3
"""
메모리 최적화 테스트: 청크 로딩 vs 전체 로딩 비교
"""
import psutil
import os
import gc

from preprocess.torchrec.schema import build_torchrec_schema_from_meta
from preprocess.torchrec.feature_store import load_feature_store_from_parquet


def get_memory_usage():
    """현재 프로세스의 메모리 사용량 (GB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


def test_chunked_loading():
    """청크 로딩 테스트"""
    print("\n" + "=" * 80)
    print("테스트 1: 청크 로딩 (메모리 최적화)")
    print("=" * 80)

    gc.collect()
    mem_before = get_memory_usage()
    print(f"시작 메모리: {mem_before:.2f} GB")

    schema = build_torchrec_schema_from_meta(
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv"
    )

    # Notice (큰 파일) - 청크 로딩
    notice_store = load_feature_store_from_parquet(
        "data/parquet/notice.parquet",
        schema.notice,
        show_progress=True,
        use_chunked=True,
        chunksize=100_000
    )

    mem_after_notice = get_memory_usage()
    print(f"Notice 로드 후 메모리: {mem_after_notice:.2f} GB (증가: {mem_after_notice - mem_before:.2f} GB)")

    # Company - 청크 로딩
    company_store = load_feature_store_from_parquet(
        "data/parquet/company.parquet",
        schema.company,
        show_progress=True,
        use_chunked=True,
        chunksize=50_000
    )

    mem_after_company = get_memory_usage()
    print(f"Company 로드 후 메모리: {mem_after_company:.2f} GB (증가: {mem_after_company - mem_after_notice:.2f} GB)")

    print(f"\n총 메모리 사용량: {mem_after_company:.2f} GB")
    print(f"총 증가량: {mem_after_company - mem_before:.2f} GB")

    return notice_store, company_store


def test_full_loading():
    """전체 로딩 테스트 (기존 방식)"""
    print("\n" + "=" * 80)
    print("테스트 2: 전체 로딩 (기존 방식)")
    print("=" * 80)
    print("⚠️  메모리 부족으로 인해 스킵할 수 있습니다.")

    gc.collect()
    mem_before = get_memory_usage()
    print(f"시작 메모리: {mem_before:.2f} GB")

    schema = build_torchrec_schema_from_meta(
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv"
    )

    try:
        # Notice (큰 파일) - 전체 로딩
        notice_store = load_feature_store_from_parquet(
            "data/parquet/notice.parquet",
            schema.notice,
            show_progress=True,
            use_chunked=False  # 전체 로딩
        )

        mem_after_notice = get_memory_usage()
        print(f"Notice 로드 후 메모리: {mem_after_notice:.2f} GB (증가: {mem_after_notice - mem_before:.2f} GB)")

        # Company - 전체 로딩
        company_store = load_feature_store_from_parquet(
            "data/parquet/company.parquet",
            schema.company,
            show_progress=True,
            use_chunked=False
        )

        mem_after_company = get_memory_usage()
        print(f"Company 로드 후 메모리: {mem_after_company:.2f} GB (증가: {mem_after_company - mem_after_notice:.2f} GB)")

        print(f"\n총 메모리 사용량: {mem_after_company:.2f} GB")
        print(f"총 증가량: {mem_after_company - mem_before:.2f} GB")

        return notice_store, company_store

    except MemoryError as e:
        print(f"❌ 메모리 부족: {e}")
        return None, None


def test_full_dataset_loading():
    """실제 학습 시나리오: Notice + Company + Pairs 모두 로드"""
    import pandas as pd

    print("\n" + "=" * 80)
    print("테스트 3: 전체 데이터셋 로딩 (Notice + Company + Pairs)")
    print("=" * 80)

    gc.collect()
    mem_start = get_memory_usage()
    print(f"시작 메모리: {mem_start:.2f} GB")

    schema = build_torchrec_schema_from_meta(
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv"
    )

    # 1. Notice 로드
    print("\n[1/3] Notice 피처 로딩...")
    notice_store = load_feature_store_from_parquet(
        "data/parquet/notice.parquet",
        schema.notice,
        show_progress=True,
        use_chunked=True,
        chunksize=100_000
    )
    mem_after_notice = get_memory_usage()
    print(f"   메모리: {mem_after_notice:.2f} GB (Notice 증가: +{mem_after_notice - mem_start:.2f} GB)")

    # 2. Company 로드
    print("\n[2/3] Company 피처 로딩...")
    company_store = load_feature_store_from_parquet(
        "data/parquet/company.parquet",
        schema.company,
        show_progress=True,
        use_chunked=True,
        chunksize=50_000
    )
    mem_after_company = get_memory_usage()
    print(f"   메모리: {mem_after_company:.2f} GB (Company 증가: +{mem_after_company - mem_after_notice:.2f} GB)")

    # 3. Pairs 로드 (청크 단위)
    print("\n[3/3] Pairs 데이터 로딩 (청크 단위)...")
    pairs_start = get_memory_usage()

    import pyarrow.parquet as pq
    from tqdm import tqdm

    parquet_file = pq.ParquetFile("data/parquet/pairs.parquet")
    total_rows = parquet_file.metadata.num_rows

    pairs_chunks = []
    with tqdm(total=total_rows, desc="     Loading Pairs", unit="rows", leave=False) as pbar:
        for batch in parquet_file.iter_batches(batch_size=1_000_000):  # 100만 개씩
            chunk = batch.to_pandas()

            # 메모리 최적화: categorical 타입으로 변환
            chunk['bidntceno'] = chunk['bidntceno'].astype('category')
            chunk['bidntceord'] = pd.to_numeric(chunk['bidntceord'], errors='coerce').astype('int16')
            chunk['bizno'] = chunk['bizno'].astype('category')

            pairs_chunks.append(chunk)
            pbar.update(len(chunk))

            # 메모리 해제
            del chunk

    # 청크 병합
    print("     병합 중...")
    pairs_df = pd.concat(pairs_chunks, ignore_index=True)
    del pairs_chunks
    gc.collect()

    mem_after_pairs = get_memory_usage()
    pairs_mem = pairs_df.memory_usage(deep=True).sum() / 1024**3
    print(f"   ✓ Pairs 로드 완료: {len(pairs_df):,} rows")
    print(f"   Pairs DataFrame 메모리: {pairs_mem:.2f} GB")
    print(f"   메모리: {mem_after_pairs:.2f} GB (Pairs 증가: +{mem_after_pairs - mem_after_company:.2f} GB)")

    # 최종 통계
    print("\n" + "=" * 80)
    print("전체 데이터셋 로딩 완료")
    print("=" * 80)
    print(f"Notice features: {len(notice_store['ids']):,}")
    print(f"Company features: {len(company_store['ids']):,}")
    print(f"Pairs: {len(pairs_df):,}")
    print(f"\n총 메모리 사용량: {mem_after_pairs:.2f} GB")
    print(f"총 증가량: {mem_after_pairs - mem_start:.2f} GB")
    print(f"사용 가능한 메모리: {psutil.virtual_memory().available / 1024**3:.2f} GB")

    # 상세 메모리 분석
    print("\n상세 메모리 분석:")
    print(f"  - Notice: ~{mem_after_notice - mem_start:.2f} GB")
    print(f"  - Company: ~{mem_after_company - mem_after_notice:.2f} GB")
    print(f"  - Pairs: ~{mem_after_pairs - mem_after_company:.2f} GB")

    return notice_store, company_store, pairs_df


if __name__ == "__main__":
    import sys

    print("메모리 최적화 테스트 시작")
    print(f"시스템 총 메모리: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    print(f"사용 가능한 메모리: {psutil.virtual_memory().available / 1024**3:.2f} GB")

    # 실제 학습 시나리오 테스트 (Notice + Company + Pairs)
    if "--full-dataset" in sys.argv or len(sys.argv) == 1:
        notice, company, pairs = test_full_dataset_loading()

        # 메모리 정리
        del notice, company, pairs
        gc.collect()

        print(f"\n메모리 정리 후: {get_memory_usage():.2f} GB")

    # 개별 청크 로딩 테스트
    elif "--chunked-only" in sys.argv:
        notice_chunked, company_chunked = test_chunked_loading()
        del notice_chunked, company_chunked
        gc.collect()

    # 전체 로딩 비교 테스트 (메모리 많이 사용)
    elif "--test-full" in sys.argv:
        test_full_loading()

    print("\n✅ 테스트 완료!")
