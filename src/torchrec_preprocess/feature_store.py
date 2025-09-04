from __future__ import annotations
import pandas as pd, numpy as np, time
from typing import Dict, List, Tuple, Optional
from sqlalchemy import text
import torch

from src.torchrec_preprocess.schema import SideSchema

class FeatureStore:
    def __init__(self, engine, side_schema: SideSchema, chunksize: int = 200_000, limit: int = None, where_condition: str = None):
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
        # 먼저 총 행 수 계산 (진행도 표시용)
        if show_progress:
            if self.limit:
                total_rows = self.limit
            else:
                count_sql = f"SELECT COUNT(*) FROM {self.sch.table}_preprocessed"
                if self.where_condition:
                    count_sql += f" WHERE {self.where_condition}"
                with self.engine.connect() as cx:
                    total_rows = cx.execute(text(count_sql)).scalar()
            
            from tqdm import tqdm
            pbar = tqdm(total=total_rows, desc=f"Loading {self.sch.table}", unit="rows")
        
        # 필요한 컬럼만 SELECT
        cols = self.sch.pk_cols + self.sch.numeric + self.sch.categorical
        if self.sch.text:
            for col in self.sch.text:
                vec_select = f"{col}::float4[] AS {col}"
                cols.append(vec_select)

        select_cols = ", ".join([c if "::" in c else c for c in cols])
        sql = f"SELECT {select_cols} FROM {self.sch.table}_preprocessed"
        if self.where_condition:
            sql += f" WHERE {self.where_condition}"
        if self.limit:
            sql += f" LIMIT {self.limit}"

        # 누적 버퍼
        num_buf = []
        cat_buf = []
        txt_buf = {col: [] for col in self.sch.text}

        with self.engine.connect() as cx:
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
        if txt_buf: self._txt_mat = {col: np.vstack(emb) for col, emb in txt_buf.items()}
        

def process_text_embeddings_optimized(df, text_cols, vec_dims):
    """벡터화된 텍스트 임베딩 처리"""
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
            # 벡터화된 변환 (훨씬 빠름)
            valid_arrays = np.array([
                np.array(emb, dtype=np.float32) 
                for emb in valid_embeddings.values
            ])
            result_array[valid_mask] = valid_arrays
        
        result[col] = result_array
        # print(f"Text column {col} processed in {time.time() - start_time:.2f}s")
    
    return result        


def build_feature_store(engine, side_schema: SideSchema, chunksize: int = 200_000, limit: int = None, show_progress: bool = False):
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
    chunksize: int = 200_000, 
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



if __name__ == "__main__":
    from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
    from data.database_connector import DatabaseConnector

    db = DatabaseConnector()
    engine = db.engine
    
    schema = build_torchrec_schema_from_meta(
        notice_table="notice",
        company_table="company",
        pair_table="bid_two_tower",
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv",
    )
    
    notice_schema = schema.notice
    
    result = build_feature_store(engine, notice_schema, chunksize=1000, limit=10000)
    
    print(result)