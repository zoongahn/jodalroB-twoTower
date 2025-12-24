"""
Pair Loader V2 - EmbeddingBagCollection용 최적화 DataLoader

V1의 모든 기능 + GPU 최적화:
- ID-only DataLoader (H2D 최소화)
- Streaming 모드 (대용량 데이터)
- Shuffle (epoch별 셔플)
- 선택적 feature 로딩
- GPU gather (train loop에서)
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Tuple, Optional
from sqlalchemy.engine import Engine
from pathlib import Path

from preprocess.torchrec.schema import TorchRecSchema
from preprocess.torchrec.feature_store import build_feature_store, build_feature_store_with_condition, load_feature_store_from_parquet, load_feature_store_from_parquet_filtered
from preprocess.torchrec.feature_preprocessor import FeaturePreprocessor


class PairDatasetV2(Dataset):
    """
    Pair Dataset V2 - 비스트리밍 모드 (전체 메모리 로드)

    ID-only 반환 → GPU gather는 train loop에서
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        notice_store: Dict,
        company_store: Dict,
        schema: TorchRecSchema,
        shuffle: bool = False,
        shuffle_seed: int = 42,
    ):
        """
        Args:
            pairs_df: pair 데이터 (bidntceno, bidntceord, bizno)
            notice_store: Notice feature store (preprocessed)
            company_store: Company feature store (preprocessed)
            schema: TorchRec 스키마
            shuffle: Epoch별 셔플 여부
            shuffle_seed: 셔플 시드
        """
        self.schema = schema
        self.notice_store = notice_store
        self.company_store = company_store
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.current_epoch = 0

        # ID 매핑 생성
        self._build_id_mappings(notice_store, company_store)

        # Pairs를 텐서로 변환 (한 번만)
        self._build_pair_tensors(pairs_df)

        print(f"✅ PairDatasetV2 초기화 완료:")
        print(f"   - Pairs: {len(self.notice_idx_tensor):,}개")
        print(f"   - Notice features: {len(self.notice_id_to_idx):,}개")
        print(f"   - Company features: {len(self.company_id_to_idx):,}개")
        print(f"   - Shuffle: {shuffle}")

    def _build_id_mappings(self, notice_store: Dict, company_store: Dict):
        """ID → 인덱스 매핑"""
        # Notice: tuple of str
        notice_ids = notice_store['ids']
        self.notice_id_to_idx = {
            (str(a), str(b)): idx for idx, (a, b) in enumerate(notice_ids)
        }

        # Company: str
        company_ids = company_store['ids']
        if company_ids:
            if isinstance(company_ids[0], (tuple, list)):
                self.company_id_to_idx = {
                    str(id_tuple[0]): idx for idx, id_tuple in enumerate(company_ids)
                }
            else:
                self.company_id_to_idx = {
                    str(company_id): idx for idx, company_id in enumerate(company_ids)
                }

    def _build_pair_tensors(self, pairs_df: pd.DataFrame):
        """Pairs를 텐서로 변환 (벡터화 최적화)"""
        from tqdm import tqdm

        print(f"   Pair 텐서 변환 중 ({len(pairs_df):,} pairs)...")

        # Category dtype 처리
        bidntceno_values = pairs_df['bidntceno'].astype(str).values
        bidntceord_values = pairs_df['bidntceord'].astype(str).values
        bizno_values = pairs_df['bizno'].astype(str).values

        # 청크 단위로 처리 (메모리 효율적)
        chunk_size = 1_000_000  # 100만 개씩
        total_pairs = len(pairs_df)

        notice_indices = []
        company_indices = []
        skipped = 0

        with tqdm(total=total_pairs, desc="   Mapping", unit="pairs", leave=False) as pbar:
            for start_idx in range(0, total_pairs, chunk_size):
                end_idx = min(start_idx + chunk_size, total_pairs)

                # 청크 처리
                chunk_notice_indices = []
                chunk_company_indices = []

                for i in range(start_idx, end_idx):
                    notice_key = (bidntceno_values[i], bidntceord_values[i])
                    company_key = bizno_values[i]

                    # 매핑
                    notice_idx = self.notice_id_to_idx.get(notice_key)
                    company_idx = self.company_id_to_idx.get(company_key)

                    if notice_idx is not None and company_idx is not None:
                        chunk_notice_indices.append(notice_idx)
                        chunk_company_indices.append(company_idx)
                    else:
                        skipped += 1
                        if skipped <= 5:
                            missing = notice_key if notice_idx is None else company_key
                            print(f"⚠️ Pair 매핑 실패 (스킵): {missing}")

                notice_indices.extend(chunk_notice_indices)
                company_indices.extend(chunk_company_indices)

                pbar.update(end_idx - start_idx)

        # 매칭률 출력
        match_rate = len(notice_indices) / len(pairs_df) * 100 if len(pairs_df) > 0 else 0
        print(f"   📊 Pair 매칭률: {len(notice_indices):,}/{len(pairs_df):,} ({match_rate:.1f}%)")
        if skipped > 0:
            print(f"   ⚠️ 스킵된 pair: {skipped:,}개")

        # LongTensor로 변환 (CPU)
        self.notice_idx_tensor = torch.as_tensor(notice_indices, dtype=torch.long)
        self.company_idx_tensor = torch.as_tensor(company_indices, dtype=torch.long)

        # 메모리 해제
        del notice_indices, company_indices
        import gc
        gc.collect()

        # Shuffle 인덱스 초기화
        self.indices = torch.arange(len(self.notice_idx_tensor))

    def set_epoch(self, epoch: int):
        """Epoch 설정 (shuffle용)"""
        self.current_epoch = epoch
        if self.shuffle:
            # Epoch별로 다른 순서
            generator = torch.Generator()
            generator.manual_seed(self.shuffle_seed + epoch)
            self.indices = torch.randperm(len(self.notice_idx_tensor), generator=generator)

    def __len__(self) -> int:
        return len(self.notice_idx_tensor)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """
        텐서 스칼라 반환 → default_collate가 자동으로 [B] 텐서 생성

        Returns:
            (notice_idx, company_idx): 각각 Python int (텐서 스칼라)
        """
        # Shuffle 적용
        actual_idx = self.indices[idx].item()

        return (
            self.notice_idx_tensor[actual_idx].item(),
            self.company_idx_tensor[actual_idx].item()
        )


class StreamingPairDatasetV2(IterableDataset):
    """
    Streaming Pair Dataset V2 - chunk 단위 로딩

    대용량 데이터셋용 (메모리 제약)
    """

    def __init__(
        self,
        db_engine: Engine,
        schema: TorchRecSchema,
        notice_store: Dict,
        company_store: Dict,
        chunk_size: int = 10000,
        offset: int = 0,
        limit: Optional[int] = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
    ):
        """
        Args:
            db_engine: DB 엔진
            schema: TorchRec 스키마
            notice_store: Notice feature store (preprocessed)
            company_store: Company feature store (preprocessed)
            chunk_size: Chunk 크기
            offset: 시작 offset
            limit: 총 pair 제한
            shuffle: Chunk 내 셔플 여부
            shuffle_seed: 셔플 시드
        """
        self.db_engine = db_engine
        self.schema = schema
        self.notice_store = notice_store
        self.company_store = company_store
        self.chunk_size = chunk_size
        self.offset = offset
        self.limit = limit
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.current_epoch = 0

        # ID 매핑
        self._build_id_mappings(notice_store, company_store)

        print(f"✅ StreamingPairDatasetV2 초기화 완료:")
        print(f"   - Chunk size: {chunk_size:,}")
        print(f"   - Offset: {offset:,}")
        print(f"   - Limit: {limit if limit else '전체'}")
        print(f"   - Shuffle: {shuffle}")

    def _build_id_mappings(self, notice_store: Dict, company_store: Dict):
        """ID → 인덱스 매핑"""
        notice_ids = notice_store['ids']
        self.notice_id_to_idx = {
            (str(a), str(b)): idx for idx, (a, b) in enumerate(notice_ids)
        }

        company_ids = company_store['ids']
        if company_ids:
            if isinstance(company_ids[0], (tuple, list)):
                self.company_id_to_idx = {
                    str(id_tuple[0]): idx for idx, id_tuple in enumerate(company_ids)
                }
            else:
                self.company_id_to_idx = {
                    str(company_id): idx for idx, company_id in enumerate(company_ids)
                }

    def set_epoch(self, epoch: int):
        """Epoch 설정 (shuffle seed 변경용)"""
        self.current_epoch = epoch

    def __iter__(self):
        """Iterator: chunk 단위로 pair 로딩"""
        current_offset = self.offset
        total_yielded = 0

        while True:
            # Limit 체크
            if self.limit and total_yielded >= self.limit:
                break

            # Chunk 크기 조정 (limit 고려)
            actual_chunk_size = self.chunk_size
            if self.limit:
                remaining = self.limit - total_yielded
                actual_chunk_size = min(self.chunk_size, remaining)

            # DB에서 chunk 로딩
            query = f"""
            SELECT bidntceno, bidntceord, bizno
            FROM {self.schema.pair.table}
            ORDER BY id
            LIMIT {actual_chunk_size} OFFSET {current_offset}
            """

            chunk_df = pd.read_sql(query, self.db_engine)

            if len(chunk_df) == 0:
                break

            # Chunk를 텐서로 변환
            notice_indices = []
            company_indices = []

            for _, row in chunk_df.iterrows():
                notice_key = (str(row['bidntceno']), str(row['bidntceord']))
                company_key = str(row['bizno'])

                try:
                    notice_idx = self.notice_id_to_idx[notice_key]
                    company_idx = self.company_id_to_idx[company_key]

                    notice_indices.append(notice_idx)
                    company_indices.append(company_idx)

                except KeyError:
                    continue

            # Shuffle (chunk 내부)
            if self.shuffle:
                indices = list(range(len(notice_indices)))
                rng = np.random.RandomState(self.shuffle_seed + self.current_epoch + current_offset)
                rng.shuffle(indices)
                notice_indices = [notice_indices[i] for i in indices]
                company_indices = [company_indices[i] for i in indices]

            # Yield
            for notice_idx, company_idx in zip(notice_indices, company_indices):
                yield (notice_idx, company_idx)
                total_yielded += 1

            # 다음 chunk
            current_offset += actual_chunk_size

            # Chunk가 작으면 종료
            if len(chunk_df) < actual_chunk_size:
                break


def create_pair_dataloaders(
    db_engine: Engine,
    schema: TorchRecSchema,
    batch_size: int = 256,
    pair_limit: Optional[int] = None,
    test_split: float = 0.1,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    streaming: bool = False,
    chunk_size: int = 10000,
    feature_chunksize: int = 10000,
    feature_limit: Optional[int] = None,
    device: torch.device = None,
    test_mode: bool = False,
    use_parquet: bool = False,
    parquet_dir: str = "data/parquet",
    preprocessor: Optional[FeaturePreprocessor] = None,  # Resume 시 기존 preprocessor 사용
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Pair DataLoader V2 생성 (EmbeddingBagCollection용)

    V1의 모든 기능 포함:
    - streaming: chunk 단위 로딩
    - shuffle: epoch별 셔플
    - num_workers, pin_memory, prefetch_factor
    - 선택적 feature 로딩
    - test_mode: pair_limit만큼만 feature 로딩
    - use_parquet: Parquet 파일로부터 로딩

    Args:
        db_engine: DB 엔진
        schema: TorchRec 스키마
        batch_size: 배치 크기
        pair_limit: pair 개수 제한
        test_split: 테스트 비율
        shuffle: 셔플 여부
        shuffle_seed: 셔플 시드
        num_workers: DataLoader workers (streaming=False면 0 권장)
        pin_memory: Pin memory 사용
        prefetch_factor: Prefetch factor
        persistent_workers: Persistent workers
        streaming: 스트리밍 모드
        chunk_size: 스트리밍 chunk 크기
        feature_chunksize: Feature 로딩 chunk 크기
        feature_limit: Feature 로딩 제한
        device: GPU device
        test_mode: True면 pair_limit에 해당하는 feature만 로딩
        use_parquet: True면 DB 대신 Parquet 파일 사용
        parquet_dir: Parquet 파일 디렉토리
        preprocessor: Resume 시 기존 학습에서 사용한 FeaturePreprocessor (None이면 새로 생성)

    Returns:
        (train_loader, test_loader, metadata)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test mode: pair_limit에 해당하는 feature만 선택적으로 로딩
    if test_mode and pair_limit:
        print(f"🧪 Test Mode 활성화: pair_limit={pair_limit}, streaming={streaming}")
        return _create_test_mode_dataloaders(
            db_engine=db_engine,
            schema=schema,
            batch_size=batch_size,
            pair_limit=pair_limit,
            test_split=test_split,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            streaming=streaming,
            chunk_size=chunk_size,
            device=device,
            use_parquet=use_parquet,
            parquet_dir=parquet_dir,
            preprocessor=preprocessor,
        )

    print("\n" + "="*80)
    print("PairDataLoaderV2 생성 (EmbeddingBagCollection용)")
    print("="*80)
    print(f"모드: {'Streaming' if streaming else '전체 로드'}")
    print(f"데이터 소스: {'Parquet' if use_parquet else 'Database'}")
    print(f"Shuffle: {shuffle}")
    print(f"Batch size: {batch_size}")
    print(f"Pair limit: {pair_limit if pair_limit else '전체'}")

    # ========================================================================
    # 1. Feature Store 로딩 (전체 또는 선택적)
    # ========================================================================
    print("\n1. Feature Store 로딩...")

    # 메모리 모니터링 시작
    import psutil
    import os as os_module
    import gc
    process = psutil.Process(os_module.getpid())

    def get_memory_gb():
        return process.memory_info().rss / 1024**3

    mem_start = get_memory_gb()
    print(f"   시작 메모리: {mem_start:.2f} GB")

    if use_parquet:
        # Parquet에서 로딩
        print(f"   Parquet 파일에서 로딩: {parquet_dir}")
        parquet_path = Path(parquet_dir)

        # Notice store (청크 단위)
        print(f"\n   [1/3] Notice 피처 로딩...")
        notice_store = load_feature_store_from_parquet(
            parquet_path / "notice.parquet",
            schema.notice,
            show_progress=True,
            use_chunked=True,
            chunksize=100_000
        )
        mem_after_notice = get_memory_gb()
        print(f"   ✓ Notice features: {len(notice_store['ids']):,}")
        print(f"   메모리: {mem_after_notice:.2f} GB (+{mem_after_notice - mem_start:.2f} GB)")

        # Company store (청크 단위)
        print(f"\n   [2/3] Company 피처 로딩...")
        company_store = load_feature_store_from_parquet(
            parquet_path / "company.parquet",
            schema.company,
            show_progress=True,
            use_chunked=True,
            chunksize=50_000
        )
        mem_after_company = get_memory_gb()
        print(f"   ✓ Company features: {len(company_store['ids']):,}")
        print(f"   메모리: {mem_after_company:.2f} GB (+{mem_after_company - mem_after_notice:.2f} GB)")

        # Pair 로딩 (청크 단위)
        if not streaming:
            print(f"\n   [3/3] Pairs 데이터 로딩...")
            import pyarrow.parquet as pq
            from tqdm import tqdm

            parquet_file = pq.ParquetFile(parquet_path / "pairs.parquet")
            total_rows = parquet_file.metadata.num_rows

            # pair_limit 적용
            if pair_limit and pair_limit < total_rows:
                total_rows = pair_limit

            pairs_chunks = []
            loaded_rows = 0

            with tqdm(total=total_rows, desc="   Loading", unit="rows", leave=False) as pbar:
                for batch in parquet_file.iter_batches(batch_size=1_000_000):
                    chunk = batch.to_pandas()

                    # pair_limit 체크
                    if pair_limit and loaded_rows + len(chunk) > pair_limit:
                        chunk = chunk.head(pair_limit - loaded_rows)

                    # 메모리 최적화: categorical 타입으로 변환
                    # 주의: bidntceord는 '000' 같은 leading zero를 유지해야 하므로 category로 변환
                    chunk['bidntceno'] = chunk['bidntceno'].astype('category')
                    chunk['bidntceord'] = chunk['bidntceord'].astype('category')
                    chunk['bizno'] = chunk['bizno'].astype('category')

                    pairs_chunks.append(chunk)
                    loaded_rows += len(chunk)
                    pbar.update(len(chunk))

                    del chunk

                    # pair_limit 도달 시 중단
                    if pair_limit and loaded_rows >= pair_limit:
                        break

            # 청크 병합
            pairs_df = pd.concat(pairs_chunks, ignore_index=True)
            del pairs_chunks
            gc.collect()

            mem_after_pairs = get_memory_gb()
            pairs_mem = pairs_df.memory_usage(deep=True).sum() / 1024**3
            print(f"   ✓ 로딩된 pair 수: {len(pairs_df):,}")
            print(f"   Pairs 메모리: {pairs_mem:.2f} GB")
            print(f"   메모리: {mem_after_pairs:.2f} GB (+{mem_after_pairs - mem_after_company:.2f} GB)")

            print(f"\n   총 메모리 사용량: {mem_after_pairs:.2f} GB (증가: +{mem_after_pairs - mem_start:.2f} GB)")
        else:
            print(f"   ⚠️ Streaming 모드는 Parquet에서 아직 지원되지 않습니다.")
            raise NotImplementedError("Streaming mode with Parquet is not yet supported")

    elif pair_limit and not streaming:
        # 선택적 로딩: pair_limit만큼만 로드 후 필요한 feature만
        print(f"   선택적 Feature 로딩 모드 (pair_limit={pair_limit:,})")

        # 먼저 pair 로딩
        pair_query = f"""
        SELECT bidntceno, bidntceord, bizno
        FROM {schema.pair.table}
        ORDER BY id
        LIMIT {pair_limit}
        """
        pairs_df = pd.read_sql(pair_query, db_engine)
        print(f"   로딩된 pair 수: {len(pairs_df):,}")

        # 필요한 ID만 추출
        notice_ids = set(zip(pairs_df['bidntceno'], pairs_df['bidntceord']))
        company_ids = set(pairs_df['bizno'].astype(str))

        print(f"   Notice ID 수: {len(notice_ids):,}")
        print(f"   Company ID 수: {len(company_ids):,}")

        # SQL WHERE 조건 생성
        notice_conditions = []
        for bidntceno, bidntceord in list(notice_ids):
            notice_conditions.append(
                f"(bidntceno='{bidntceno}' AND bidntceord::integer={int(bidntceord)})"
            )
        notice_where = " OR ".join(notice_conditions)

        company_quoted_ids = [f"'{cid}'" for cid in company_ids]
        company_where = "bizno IN (" + ",".join(company_quoted_ids) + ")"

        # Feature store 로딩
        notice_store = build_feature_store_with_condition(
            db_engine, schema.notice, where_condition=notice_where, show_progress=True
        )
        company_store = build_feature_store_with_condition(
            db_engine, schema.company, where_condition=company_where, show_progress=True
        )

        print(f"   Notice features: {len(notice_store['ids']):,}")
        print(f"   Company features: {len(company_store['ids']):,}")

    else:
        # 전체 로딩
        print(f"   전체 Feature 로딩")
        notice_store = build_feature_store(
            db_engine, schema.notice,
            chunksize=feature_chunksize,
            limit=feature_limit,
            show_progress=True
        )
        company_store = build_feature_store(
            db_engine, schema.company,
            chunksize=feature_chunksize,
            limit=feature_limit,
            show_progress=True
        )

        print(f"   Notice features: {len(notice_store['ids']):,}")
        print(f"   Company features: {len(company_store['ids']):,}")

        # Pair 로딩 (선택적 로딩이 아니면 여기서)
        if not streaming:
            pair_query = f"""
            SELECT bidntceno, bidntceord, bizno
            FROM {schema.pair.table}
            ORDER BY id
            """
            if pair_limit:
                pair_query += f" LIMIT {pair_limit}"

            print(f"   Pair 로딩 중 (메모리 최적화: 청크 단위 로딩 + categorical 변환)...")

            # 청크 단위로 읽으면서 즉시 변환 (OOM 방지)
            chunk_size = 10_000_000  # 1천만 개씩
            chunks = []

            for chunk_df in pd.read_sql(pair_query, db_engine, chunksize=chunk_size):
                # 즉시 메모리 최적화
                chunk_df['bidntceno'] = chunk_df['bidntceno'].astype('category')
                chunk_df['bidntceord'] = pd.to_numeric(chunk_df['bidntceord'], errors='coerce').astype('int16')
                chunk_df['bizno'] = chunk_df['bizno'].astype('category')
                chunks.append(chunk_df)
                print(f"      청크 로딩: {len(chunk_df):,}개 (누적: {sum(len(c) for c in chunks):,}개)")

            # 최종 병합
            pairs_df = pd.concat(chunks, ignore_index=True)

            mem_usage = pairs_df.memory_usage(deep=True).sum() / 1024**3
            print(f"   ✅ 로딩 완료: {len(pairs_df):,}개 pair (메모리: {mem_usage:.2f} GB)")

    # ========================================================================
    # 2. Feature 전처리 (projection)
    # ========================================================================
    print("\n2. Feature 전처리 (projection)...")
    if preprocessor is None:
        print("   새로운 FeaturePreprocessor 생성")
        preprocessor = FeaturePreprocessor(
            schema=schema,
            device=device,
            num_proj_dim=128,
            text_proj_dim=128,
            batch_size=1024
        )
    else:
        print("   ✅ 기존 FeaturePreprocessor 사용 (Resume 모드)")
        # device 이동 확인
        preprocessor.device = device
        for tower_name, proj in preprocessor.projectors.items():
            proj.to(device)

    preprocessed_stores = preprocessor.preprocess_stores(notice_store, company_store)

    print(f"   Notice dense_projected: {preprocessed_stores['notice']['dense_projected'].shape}")
    print(f"   Company dense_projected: {preprocessed_stores['company']['dense_projected'].shape}")

    # ========================================================================
    # 3. Dataset 생성
    # ========================================================================
    print("\n3. Dataset 생성...")

    if streaming:
        # Streaming 모드
        print(f"   Streaming 모드 (chunk_size={chunk_size:,})")

        # 전체 pair 개수 확인 (pair_limit이 없는 경우)
        total_pairs_in_db = None
        if not pair_limit:
            count_query = f"SELECT COUNT(*) FROM {schema.pair.table}"
            total_pairs_in_db = pd.read_sql(count_query, db_engine).iloc[0, 0]
            print(f"   전체 pair 수: {total_pairs_in_db:,}")

        # Train dataset
        effective_total = pair_limit if pair_limit else total_pairs_in_db
        train_limit = int(effective_total * (1 - test_split)) if effective_total else None

        train_dataset = StreamingPairDatasetV2(
            db_engine=db_engine,
            schema=schema,
            notice_store=preprocessed_stores['notice'],
            company_store=preprocessed_stores['company'],
            chunk_size=chunk_size,
            offset=0,
            limit=train_limit,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
        )

        # Test dataset
        test_dataset = None
        if test_split > 0 and train_limit:
            test_offset = train_limit
            test_limit = int(effective_total * test_split) if effective_total else None

            test_dataset = StreamingPairDatasetV2(
                db_engine=db_engine,
                schema=schema,
                notice_store=preprocessed_stores['notice'],
                company_store=preprocessed_stores['company'],
                chunk_size=chunk_size,
                offset=test_offset,
                limit=test_limit,
                shuffle=False,
                shuffle_seed=shuffle_seed,
            )
            print(f"   Train limit: {train_limit:,}, Test offset: {test_offset:,}, Test limit: {test_limit:,}")

    else:
        # 비스트리밍 모드 (전체 메모리)
        print(f"   비스트리밍 모드 (전체 메모리 로드)")

        # Train/Test 분할
        if test_split > 0:
            from sklearn.model_selection import train_test_split
            train_pairs, test_pairs = train_test_split(
                pairs_df, test_size=test_split, random_state=shuffle_seed
            )
        else:
            train_pairs = pairs_df
            test_pairs = None

        print(f"   Train pairs: {len(train_pairs):,}")
        if test_pairs is not None:
            print(f"   Test pairs: {len(test_pairs):,}")

        # Train dataset
        train_dataset = PairDatasetV2(
            pairs_df=train_pairs,
            notice_store=preprocessed_stores['notice'],
            company_store=preprocessed_stores['company'],
            schema=schema,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
        )

        # Test dataset
        test_dataset = None
        if test_pairs is not None:
            test_dataset = PairDatasetV2(
                pairs_df=test_pairs,
                notice_store=preprocessed_stores['notice'],
                company_store=preprocessed_stores['company'],
                schema=schema,
                shuffle=False,
                shuffle_seed=shuffle_seed,
            )

    # ========================================================================
    # 4. DataLoader 생성
    # ========================================================================
    print("\n4. DataLoader 생성...")

    # Prefetch factor 설정
    if prefetch_factor is None:
        prefetch_factor = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False if streaming else False,  # Dataset 내부에서 shuffle 처리
        collate_fn=None,  # default_collate
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers and num_workers > 0,
        )

    # ========================================================================
    # 5. Metadata 반환
    # ========================================================================
    metadata = {
        "notice_store": preprocessed_stores['notice'],
        "company_store": preprocessed_stores['company'],
        "schema": schema,
        "preprocessor": preprocessor,  # 전처리기 추가
    }

    # Streaming 모드면 total_pairs 추가
    if streaming:
        metadata["total_pairs"] = effective_total

    print(f"\n✅ DataLoader 생성 완료:")
    print(f"   Train batches: {len(train_loader) if not streaming else 'streaming'}")
    if test_loader is not None:
        print(f"   Test batches: {len(test_loader) if not streaming else 'streaming'}")
    print("="*80 + "\n")

    return train_loader, test_loader, metadata


if __name__ == "__main__":
    """테스트"""
    from database.database_connector import DatabaseConnector
    from preprocess.torchrec.schema import build_torchrec_schema_from_meta

    print("=== PairLoaderV2 테스트 ===\n")

    db = DatabaseConnector()
    engine = db.engine

    config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**config)

    # 비스트리밍 테스트
    print("\n[테스트 1] 비스트리밍 모드")
    train_loader, test_loader, metadata = create_pair_dataloaders_v2(
        db_engine=engine,
        schema=schema,
        batch_size=64,
        pair_limit=1000,
        test_split=0.2,
        shuffle=True,
        streaming=False,
        num_workers=0,
        pin_memory=True,
    )

    print("\n배치 샘플:")
    for batch_idx, batch in enumerate(train_loader):
        notice_idx, company_idx = batch
        print(f"  Batch {batch_idx}: notice_idx={notice_idx.shape}, company_idx={company_idx.shape}")
        print(f"    notice_idx 예시: {notice_idx[:5]}")
        print(f"    company_idx 예시: {company_idx[:5]}")
        if batch_idx >= 2:
            break

    # Streaming 테스트
    print("\n\n[테스트 2] 스트리밍 모드")
    train_loader_stream, test_loader_stream, metadata_stream = create_pair_dataloaders_v2(
        db_engine=engine,
        schema=schema,
        batch_size=64,
        pair_limit=1000,
        test_split=0.2,
        shuffle=True,
        streaming=True,
        chunk_size=200,
        num_workers=0,
        pin_memory=True,
    )

    print("\n배치 샘플:")
    for batch_idx, batch in enumerate(train_loader_stream):
        notice_idx, company_idx = batch
        print(f"  Batch {batch_idx}: notice_idx={notice_idx.shape}, company_idx={company_idx.shape}")
        if batch_idx >= 2:
            break

    print("\n✅ 모든 테스트 통과!")


def _create_test_mode_dataloaders(
    db_engine: Engine,
    schema: TorchRecSchema,
    batch_size: int,
    pair_limit: int,
    test_split: float,
    shuffle: bool,
    shuffle_seed: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: Optional[int],
    persistent_workers: bool,
    streaming: bool,
    chunk_size: int,
    device: torch.device,
    use_parquet: bool = False,
    parquet_dir: str = "data/parquet",
    preprocessor: Optional[FeaturePreprocessor] = None,  # Resume 시 기존 preprocessor 사용
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Test Mode용 DataLoader 생성 - pair_limit에 해당하는 feature만 선택적 로딩

    핵심: pair → ID 추출 → 해당 ID feature만 로딩 → 100% 매칭 보장
    """
    print("\n" + "="*80)
    print("🧪 Test Mode DataLoader 생성 (선택적 feature 로딩)")
    print("="*80)
    print(f"Pair limit: {pair_limit:,}")
    print(f"Streaming: {streaming}")
    print(f"Chunk size: {chunk_size if streaming else 'N/A'}")
    print(f"데이터 소스: {'Parquet (DuckDB)' if use_parquet else 'Database'}")

    # ========================================================================
    # 1. 제한된 pair 로딩
    # ========================================================================
    print("\n1. 제한된 pair 로딩...")
    if use_parquet:
        import duckdb
        pairs_df = duckdb.query(f"""
            SELECT bidntceno, bidntceord, bizno
            FROM '{parquet_dir}/pairs.parquet'
            LIMIT {pair_limit}
        """).df()
    else:
        pair_query = f"""
        SELECT bidntceno, bidntceord, bizno
        FROM {schema.pair.table}
        ORDER BY id
        LIMIT {pair_limit}
        """
        pairs_df = pd.read_sql(pair_query, db_engine)
    print(f"   로딩된 pair 수: {len(pairs_df):,}")

    # ========================================================================
    # 2. 실제 pair ID 추출
    # ========================================================================
    print("\n2. 필요한 ID 추출...")
    notice_ids = set(zip(pairs_df['bidntceno'], pairs_df['bidntceord']))
    company_ids = set(pairs_df['bizno'].astype(str))

    print(f"   Notice ID 수: {len(notice_ids):,}")
    print(f"   Company ID 수: {len(company_ids):,}")

    # ========================================================================
    # 3. 선택적 feature 로딩
    # ========================================================================
    print("\n3. 선택적 Feature 로딩...")

    if use_parquet:
        # Parquet 모드: 필터링된 로딩
        notice_store = load_feature_store_from_parquet_filtered(
            f"{parquet_dir}/notice.parquet", schema.notice, filter_ids=notice_ids, show_progress=True
        )
        company_store = load_feature_store_from_parquet_filtered(
            f"{parquet_dir}/company.parquet", schema.company, filter_ids=company_ids, show_progress=True
        )
    else:
        # DB 모드: SQL WHERE 조건
        notice_conditions = []
        for bidntceno, bidntceord in list(notice_ids):
            notice_conditions.append(
                f"(bidntceno='{bidntceno}' AND bidntceord::integer={int(bidntceord)})"
            )
        notice_where = " OR ".join(notice_conditions)

        company_quoted_ids = [f"'{cid}'" for cid in company_ids]
        company_where = "bizno IN (" + ",".join(company_quoted_ids) + ")"

        notice_store = build_feature_store_with_condition(
            db_engine, schema.notice, where_condition=notice_where, show_progress=True
        )
        company_store = build_feature_store_with_condition(
            db_engine, schema.company, where_condition=company_where, show_progress=True
        )

    print(f"   Notice features: {len(notice_store['ids']):,}")
    print(f"   Company features: {len(company_store['ids']):,}")

    # ========================================================================
    # 4. Feature 전처리 (projection)
    # ========================================================================
    print("\n4. Feature 전처리...")
    if preprocessor is None:
        print("   새로운 FeaturePreprocessor 생성")
        preprocessor = FeaturePreprocessor(
            schema=schema,
            device=device,
            num_proj_dim=128,
            text_proj_dim=128,
            batch_size=1024
        )
    else:
        print("   ✅ 기존 FeaturePreprocessor 사용 (Resume 모드)")
        # device 이동 확인
        preprocessor.device = device
        for tower_name, proj in preprocessor.projectors.items():
            proj.to(device)

    preprocessed_stores = preprocessor.preprocess_stores(notice_store, company_store)
    print(f"   Notice dense_projected: {preprocessed_stores['notice']['dense_projected'].shape}")
    print(f"   Company dense_projected: {preprocessed_stores['company']['dense_projected'].shape}")

    # ========================================================================
    # 5. Dataset 생성
    # ========================================================================
    print("\n5. Dataset 생성...")

    if streaming:
        # Streaming 모드
        print(f"   Streaming 모드 (chunk_size={chunk_size:,})")

        # Train dataset
        train_limit = int(pair_limit * (1 - test_split)) if test_split > 0 else pair_limit

        train_dataset = StreamingPairDatasetV2(
            db_engine=db_engine,
            schema=schema,
            notice_store=preprocessed_stores['notice'],
            company_store=preprocessed_stores['company'],
            chunk_size=chunk_size,
            offset=0,
            limit=train_limit,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
        )

        # Test dataset
        test_dataset = None
        if test_split > 0:
            test_offset = train_limit
            test_limit = int(pair_limit * test_split)

            test_dataset = StreamingPairDatasetV2(
                db_engine=db_engine,
                schema=schema,
                notice_store=preprocessed_stores['notice'],
                company_store=preprocessed_stores['company'],
                chunk_size=chunk_size,
                offset=test_offset,
                limit=test_limit,
                shuffle=False,
                shuffle_seed=shuffle_seed,
            )

    else:
        # 비스트리밍 모드
        print(f"   비스트리밍 모드")

        # Train/Test 분할
        if test_split > 0:
            from sklearn.model_selection import train_test_split
            train_pairs, test_pairs = train_test_split(
                pairs_df, test_size=test_split, random_state=shuffle_seed
            )
        else:
            train_pairs = pairs_df
            test_pairs = None

        print(f"   Train pairs: {len(train_pairs):,}")
        if test_pairs is not None:
            print(f"   Test pairs: {len(test_pairs):,}")

        # Train dataset
        train_dataset = PairDatasetV2(
            pairs_df=train_pairs,
            notice_store=preprocessed_stores['notice'],
            company_store=preprocessed_stores['company'],
            schema=schema,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
        )

        # Test dataset
        test_dataset = None
        if test_pairs is not None:
            test_dataset = PairDatasetV2(
                pairs_df=test_pairs,
                notice_store=preprocessed_stores['notice'],
                company_store=preprocessed_stores['company'],
                schema=schema,
                shuffle=False,
                shuffle_seed=shuffle_seed,
            )

    # ========================================================================
    # 6. DataLoader 생성
    # ========================================================================
    print("\n6. DataLoader 생성...")

    if prefetch_factor is None:
        prefetch_factor = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset 내부에서 shuffle 처리
        collate_fn=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers and num_workers > 0,
        )

    # ========================================================================
    # 7. Metadata 반환
    # ========================================================================
    metadata = {
        "notice_store": preprocessed_stores['notice'],
        "company_store": preprocessed_stores['company'],
        "schema": schema,
        "preprocessor": preprocessor,  # 전처리기 추가
    }

    # Streaming 모드면 total_pairs 추가
    if streaming:
        metadata["total_pairs"] = pair_limit

    print(f"\n✅ Test Mode DataLoader 생성 완료:")
    print(f"   Train batches: {len(train_loader) if not streaming else 'streaming'}")
    if test_loader is not None:
        print(f"   Test batches: {len(test_loader) if not streaming else 'streaming'}")
    print("="*80 + "\n")

    return train_loader, test_loader, metadata
