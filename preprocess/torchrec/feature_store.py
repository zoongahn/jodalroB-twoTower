from __future__ import annotations
import pandas as pd, numpy as np, time
from typing import Dict, List, Tuple, Optional
from sqlalchemy import text
import torch
from pathlib import Path

from preprocess.torchrec.schema import SideSchema

class FeatureStore:
    def __init__(self, engine, side_schema: SideSchema, chunksize: int = 500_000, limit: int = None, where_condition: str = None):
        self.engine = engine
        self.sch = side_schema
        self.chunksize = chunksize
        self.limit = limit
        self.where_condition = where_condition
        self._key_to_row:Dict[Tuple, int] = {}
        self._num_mat: Optional[np.ndarray] = None
        self._cat_mat: Optional[np.ndarray] = None
        self._txt_mat: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def _make_key_row(row, pk_cols: List[str]) -> Tuple:
        return tuple(row[col] for col in pk_cols)

    def build(self, show_progress: bool = True):
        # 테이블명 처리: schema.table_preprocessed 형식으로 변환
        # 단, 이미 _preprocessed가 붙어있으면 추가하지 않음
        if self.sch.table.endswith('_preprocessed'):
            # 이미 _preprocessed가 있으면 그대로 사용
            full_table = self.sch.table
        else:
            # _preprocessed가 없으면 추가
            table_parts = self.sch.table.split('.')
            if len(table_parts) == 2:
                # schema.table 형식
                schema_name, table_name = table_parts
                full_table = f"{schema_name}.{table_name}_preprocessed"
            else:
                # table만 있는 경우
                full_table = f"{self.sch.table}_preprocessed"

        # 먼저 총 행 수 계산 (진행도 표시용)
        if show_progress:
            if self.limit:
                total_rows = self.limit
            else:
                count_sql = f"SELECT COUNT(*) FROM {full_table}"
                if self.where_condition:
                    count_sql += f" WHERE {self.where_condition}"
                with self.engine.connect() as cx:
                    total_rows = cx.execute(text(count_sql)).scalar()

            from tqdm import tqdm
            import sys
            # 테이블명 축약 (step1.notice -> notice)
            table_display = self.sch.table.split('.')[-1] if '.' in self.sch.table else self.sch.table

            # 터미널이 tty인지 확인
            if sys.stdout.isatty():
                pbar = tqdm(
                    total=total_rows,
                    desc=f"{table_display}",
                    unit="rows",
                    dynamic_ncols=True,
                    leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )
            else:
                # 비-tty 환경 (redirect, pipe 등)에서는 ncols 고정
                pbar = tqdm(
                    total=total_rows,
                    desc=f"{table_display}",
                    unit="rows",
                    ncols=120,
                    leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )

        # 필요한 컬럼만 SELECT
        cols = self.sch.pk_cols + self.sch.numeric + self.sch.categorical
        if self.sch.text:
            for col in self.sch.text:
                # pgvector 타입을 text로 먼저 변환 후 파싱
                vec_select = f"{col}::text AS {col}"
                cols.append(vec_select)

        select_cols = ", ".join([c if "::" in c else c for c in cols])
        sql = f"SELECT {select_cols} FROM {full_table}"
        if self.where_condition:
            sql += f" WHERE {self.where_condition}"
        if self.limit:
            sql += f" LIMIT {self.limit}"

        # 누적 버퍼
        num_buf = []
        cat_buf = []
        txt_buf = {col: [] for col in self.sch.text}

        with self.engine.connect() as cx:
            # PostgreSQL 세션 최적화
            cx.execute(text("SET work_mem = '256MB'"))
            cx.execute(text("SET max_parallel_workers_per_gather = 4"))

            rs = cx.execution_options(stream_results=True).exec_driver_sql(sql)
            df = pd.DataFrame(rs.fetchmany(self.chunksize), columns=rs.keys())
            row_idx = 0
            
            while not df.empty:
                chunk_size = len(df)
                
                # key map
                for _, r in df.iterrows():
                    key = self._make_key_row(r, self.sch.pk_cols)
                    self._key_to_row[key] = row_idx
                    row_idx += 1

                # numeric
                if self.sch.numeric:
                    num_buf.append(df[self.sch.numeric].astype("float64").to_numpy(dtype="float32", copy=False))
                # categorical
                if self.sch.categorical:
                    cat_buf.append(df[self.sch.categorical].astype("int64").to_numpy(dtype="int64", copy=False))
                # text embedding
                vec_dims = self.sch.text_embed_dims
                if vec_dims is None:
                    vec_dims = 768
                    
                if self.sch.text:
                    # for col in self.sch.text:
                    #     lst = df[col].to_list()
                    #     emb = np.stack(
                    #         [np.asarray(v, dtype="float32") if v is not None else np.zeros(vec_dims, dtype=np.float32) for v in lst], axis=0
                    #     )
                    #     txt_buf[col].append(emb)
                    text_results = process_text_embeddings_optimized(df, self.sch.text, vec_dims)
                    for col, emb in text_results.items():
                        txt_buf[col].append(emb)
                
                # 진행도 업데이트
                if show_progress:
                    pbar.update(chunk_size)
                
                # 다음 청크 읽기
                df = pd.DataFrame(rs.fetchmany(self.chunksize), columns=rs.keys())

        if show_progress:
            pbar.close()

        # 스택
        if num_buf: self._num_mat = np.vstack(num_buf)
        if cat_buf: self._cat_mat = np.vstack(cat_buf)
        if txt_buf: self._txt_mat = {col: np.vstack(emb) for col, emb in txt_buf.items() if len(emb) > 0}
        

def process_text_embeddings_optimized(df, text_cols, vec_dims):
    """벡터화된 텍스트 임베딩 처리 (pgvector text format 지원)"""
    result = {}

    for col in text_cols:
        start_time = time.time()

        # 방법 1: pandas 벡터화 + 미리 할당
        embeddings_series = df[col]
        batch_size = len(embeddings_series)

        # 결과 배열 미리 할당
        result_array = np.zeros((batch_size, vec_dims), dtype=np.float32)

        # None이 아닌 것들만 처리
        valid_mask = embeddings_series.notna()
        valid_embeddings = embeddings_series[valid_mask]

        if len(valid_embeddings) > 0:
            # pgvector text format "[0.1,0.2,...]" 파싱
            valid_arrays = []
            for idx, emb in enumerate(valid_embeddings.values):
                if isinstance(emb, str):
                    # pgvector text format: "[0.1,0.2,...]"
                    emb_str = emb.strip('[]')
                    arr = np.fromstring(emb_str, dtype=np.float32, sep=',')

                    # 파싱된 배열의 차원 확인
                    if len(arr) != vec_dims:
                        print(f"WARNING: {col}[{idx}] 차원 불일치 - 기대: {vec_dims}, 실제: {len(arr)}")
                        # 차원 맞추기
                        if len(arr) < vec_dims:
                            arr = np.pad(arr, (0, vec_dims - len(arr)), constant_values=0.0)
                        else:
                            arr = arr[:vec_dims]

                    if np.isnan(arr).any():
                        print(f"WARNING: {col}[{idx}] contains NaN values")
                else:
                    # 이미 배열 형태
                    arr = np.array(emb, dtype=np.float32)
                    if len(arr) != vec_dims:
                        if len(arr) < vec_dims:
                            arr = np.pad(arr, (0, vec_dims - len(arr)), constant_values=0.0)
                        else:
                            arr = arr[:vec_dims]

                valid_arrays.append(arr)

            result_array[valid_mask] = np.array(valid_arrays)

        result[col] = result_array
        # print(f"Text column {col} processed in {time.time() - start_time:.2f}s")
    
    return result        


def build_feature_store(engine, side_schema: SideSchema, chunksize: int = 500_000, limit: int = None, show_progress: bool = False):
    print(f"  🔧 build_feature_store: chunksize={chunksize:,} (table={side_schema.table})")
    store = FeatureStore(engine, side_schema, chunksize, limit, where_condition=None)  # where_condition 추가
    store.build(show_progress)
    
    result = {
        "ids": list(store._key_to_row.keys()),  # ID 목록 추가
        "numeric": store._num_mat,
        "categorical": store._cat_mat,
        "text": store._txt_mat,
    }
    return result

def build_feature_store_with_condition(
    engine,
    side_schema: SideSchema,
    where_condition: str,
    chunksize: int = 500_000,
    show_progress: bool = False
):
    """
    WHERE 조건을 사용하여 선택적으로 피처 로딩

    Args:
        engine: DB 연결
        side_schema: 스키마 정보
        where_condition: SQL WHERE 조건 (예: "WHERE id IN (1,2,3)")
        chunksize: 청크 크기
        show_progress: 진행률 표시 여부
    """
    store = FeatureStore(engine, side_schema, chunksize, limit=None, where_condition=where_condition)
    store.build(show_progress)

    result = {
        "ids": list(store._key_to_row.keys()),  # ID 목록 추가
        "numeric": store._num_mat,
        "categorical": store._cat_mat,
        "text": store._txt_mat,
    }
    return result


def load_feature_store_from_parquet(
    parquet_path: str | Path,
    side_schema: SideSchema,
    show_progress: bool = True,
    use_chunked: bool = True,
    chunksize: int = 100_000
) -> Dict:
    """
    Parquet 파일에서 Feature Store 로딩 (메모리 최적화 버전)

    Args:
        parquet_path: Parquet 파일 경로 (예: "data/parquet/notice.parquet")
        side_schema: 스키마 정보
        show_progress: 진행률 표시 여부
        use_chunked: 청크 단위 로딩 사용 여부 (메모리 효율적)
        chunksize: 청크 크기 (use_chunked=True일 때만 사용)

    Returns:
        dict: {"ids": list, "numeric": ndarray, "categorical": ndarray, "text": dict}
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 파일을 찾을 수 없습니다: {parquet_path}")

    print(f"  🔧 load_feature_store_from_parquet: {parquet_path.name}")
    if use_chunked:
        print(f"     메모리 최적화 모드 활성화 (chunksize={chunksize:,})")

    start_time = time.time()

    if use_chunked:
        # 청크 단위 로딩 (메모리 효율적)
        return _load_feature_store_chunked(
            parquet_path, side_schema, show_progress, chunksize, start_time
        )
    else:
        # 전체 로딩 (기존 방식)
        return _load_feature_store_full(
            parquet_path, side_schema, show_progress, start_time
        )


def _load_feature_store_full(
    parquet_path: Path,
    side_schema: SideSchema,
    show_progress: bool,
    start_time: float
) -> Dict:
    """전체 DataFrame 로딩 방식 (기존 방식, 메모리 많이 사용)"""
    df = pd.read_parquet(parquet_path)

    if show_progress:
        print(f"     ✓ Parquet 로드 완료 ({len(df):,} rows, {time.time() - start_time:.2f}s)")

    # Key-to-row mapping
    key_to_row = {}
    for idx, row in df.iterrows():
        key = tuple(row[col] for col in side_schema.pk_cols)
        key_to_row[key] = idx

    # Numeric features
    num_mat = None
    if side_schema.numeric:
        num_mat = df[side_schema.numeric].astype("float32").to_numpy(dtype="float32", copy=False)

    # Categorical features
    cat_mat = None
    if side_schema.categorical:
        cat_mat = df[side_schema.categorical].astype("int64").to_numpy(dtype="int64", copy=False)

    # Text embeddings - 최적화된 버전
    txt_mat = None
    vec_dims = side_schema.text_embed_dims or 768

    if side_schema.text:
        txt_mat = {}
        for col in side_schema.text:
            # 메모리 효율적 변환: 중간 리스트 생성 없이 직접 stack
            emb_array = np.stack([
                np.array(emb, dtype=np.float32) if emb is not None
                else np.zeros(vec_dims, dtype=np.float32)
                for emb in df[col]
            ])
            txt_mat[col] = emb_array

    result = {
        "ids": list(key_to_row.keys()),
        "numeric": num_mat,
        "categorical": cat_mat,
        "text": txt_mat,
    }

    if show_progress:
        print(f"     ✓ Feature 추출 완료 ({time.time() - start_time:.2f}s)")

    return result


def _load_feature_store_chunked(
    parquet_path: Path,
    side_schema: SideSchema,
    show_progress: bool,
    chunksize: int,
    start_time: float
) -> Dict:
    """청크 단위 로딩 방식 (메모리 효율적)"""
    import pyarrow.parquet as pq
    import gc

    # Parquet 파일 메타데이터 읽기
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows

    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=total_rows, desc=f"     Loading", unit="rows", leave=False)

    # 버퍼 초기화
    key_to_row = {}
    num_buffers = []
    cat_buffers = []
    txt_buffers = {col: [] for col in (side_schema.text or [])}

    vec_dims = side_schema.text_embed_dims or 768
    current_idx = 0

    # 청크 단위로 읽기
    for batch in parquet_file.iter_batches(batch_size=chunksize):
        df_chunk = batch.to_pandas()

        # Key mapping
        for _, row in df_chunk.iterrows():
            key = tuple(row[col] for col in side_schema.pk_cols)
            key_to_row[key] = current_idx
            current_idx += 1

        # Numeric
        if side_schema.numeric:
            num_chunk = df_chunk[side_schema.numeric].astype("float32").to_numpy(dtype="float32", copy=False)
            num_buffers.append(num_chunk)

        # Categorical
        if side_schema.categorical:
            cat_chunk = df_chunk[side_schema.categorical].astype("int64").to_numpy(dtype="int64", copy=False)
            cat_buffers.append(cat_chunk)

        # Text embeddings - 최적화된 변환
        if side_schema.text:
            for col in side_schema.text:
                # 직접 numpy array로 변환 (중간 리스트 없이)
                emb_chunk = np.stack([
                    np.array(emb, dtype=np.float32) if emb is not None
                    else np.zeros(vec_dims, dtype=np.float32)
                    for emb in df_chunk[col]
                ])
                txt_buffers[col].append(emb_chunk)

        if show_progress:
            pbar.update(len(df_chunk))

        # 명시적 메모리 해제
        del df_chunk
        if current_idx % (chunksize * 5) == 0:  # 5개 청크마다 GC
            gc.collect()

    if show_progress:
        pbar.close()
        print(f"     ✓ Parquet 로드 완료 ({total_rows:,} rows, {time.time() - start_time:.2f}s)")

    # 버퍼 병합
    num_mat = np.vstack(num_buffers) if num_buffers else None
    cat_mat = np.vstack(cat_buffers) if cat_buffers else None
    txt_mat = {col: np.vstack(buffers) for col, buffers in txt_buffers.items() if buffers} if side_schema.text else None

    # 메모리 해제
    del num_buffers, cat_buffers, txt_buffers
    gc.collect()

    result = {
        "ids": list(key_to_row.keys()),
        "numeric": num_mat,
        "categorical": cat_mat,
        "text": txt_mat,
    }

    if show_progress:
        print(f"     ✓ Feature 추출 완료 ({time.time() - start_time:.2f}s)")

    return result



def load_feature_store_from_parquet_filtered(
    parquet_path: str | Path,
    side_schema: SideSchema,
    filter_ids: set,
    show_progress: bool = True,
) -> Dict:
    """
    Parquet 파일에서 특정 ID만 필터링하여 Feature Store 로딩

    Args:
        parquet_path: Parquet 파일 경로
        side_schema: 스키마 정보
        filter_ids: 필터링할 ID set (튜플 또는 단일값)
        show_progress: 진행률 표시 여부

    Returns:
        dict: {"ids": list, "numeric": ndarray, "categorical": ndarray, "text": dict}
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 파일을 찾을 수 없습니다: {parquet_path}")

    print(f"  🔧 load_feature_store_from_parquet_filtered: {parquet_path.name}")
    print(f"     필터링할 ID 수: {len(filter_ids):,}")

    start_time = time.time()

    # Parquet 전체 로드 후 필터링 (작은 데이터셋에서 효율적)
    df = pd.read_parquet(parquet_path)

    if show_progress:
        print(f"     ✓ Parquet 로드 완료 ({len(df):,} rows, {time.time() - start_time:.2f}s)")

    # 필터링
    pk_cols = side_schema.pk_cols
    if len(pk_cols) == 1:
        # 단일 키 (예: bizno)
        df = df[df[pk_cols[0]].astype(str).isin(filter_ids)]
    else:
        # 복합 키 (예: bidntceno, bidntceord)
        df = df[df.apply(lambda x: tuple(x[col] for col in pk_cols) in filter_ids, axis=1)]

    if show_progress:
        print(f"     ✓ 필터링 완료 ({len(df):,} rows, {time.time() - start_time:.2f}s)")

    # Key-to-row mapping (재인덱싱 필요)
    df = df.reset_index(drop=True)
    key_to_row = {}
    for idx, row in df.iterrows():
        key = tuple(row[col] for col in pk_cols)
        key_to_row[key] = idx

    # Numeric features
    num_mat = None
    if side_schema.numeric:
        num_mat = df[side_schema.numeric].astype("float32").to_numpy(dtype="float32", copy=False)

    # Categorical features
    cat_mat = None
    if side_schema.categorical:
        cat_mat = df[side_schema.categorical].astype("int64").to_numpy(dtype="int64", copy=False)

    # Text embeddings
    txt_mat = None
    vec_dims = side_schema.text_embed_dims or 768

    if side_schema.text:
        txt_mat = {}
        for col in side_schema.text:
            emb_array = np.stack([
                np.array(emb, dtype=np.float32) if emb is not None
                else np.zeros(vec_dims, dtype=np.float32)
                for emb in df[col]
            ])
            txt_mat[col] = emb_array

    result = {
        "ids": list(key_to_row.keys()),
        "numeric": num_mat,
        "categorical": cat_mat,
        "text": txt_mat,
    }

    if show_progress:
        print(f"     ✓ Feature 추출 완료 ({time.time() - start_time:.2f}s)")

    return result


if __name__ == "__main__":
    from preprocess.torchrec.schema import build_torchrec_schema_from_meta
    from database.database_connector import DatabaseConnector

    db = DatabaseConnector()
    engine = db.engine

    # 스키마 구축 (.env에서 자동으로 설정 읽음)
    schema = build_torchrec_schema_from_meta(
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv",
    )

    notice_schema = schema.notice

    result = build_feature_store(engine, notice_schema, chunksize=1000, limit=10000)

    print(result)