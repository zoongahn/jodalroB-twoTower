# text_vector_updater.py  📌 텍스트 임베딩만 청크로 계산/업데이트 (새 파일 추천)
import pandas as pd
from tqdm import tqdm
import gc, torch, os

from preprocess.text_preprocess import TextPreprocessor
from preprocess.pipeline import TABLE_PK_MAP
from data.database_connector import DatabaseConnector
from data.query_helper import QueryHelper

TEXT_COL = "bidntcenm"
TEXT_EMB_DIM = 768

def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def embed_text_chunk(df_chunk: pd.DataFrame, table_name: str, text_col: str) -> pd.DataFrame:
    pk = TABLE_PK_MAP[table_name]
    for c in pk:
        if c in df_chunk.columns:
            df_chunk[c] = df_chunk[c].astype("string").fillna("")
    cfg = {text_col: {"use": True, "max_length": 128, "l2_normalize": True, "batch_size": 16}}
    tp = TextPreprocessor(config=cfg, table_pk_map=TABLE_PK_MAP)
    tdf = tp.transform(df_chunk, table_name)      # PK + {text_col}_emb###
    emb_cols = [c for c in tdf.columns if str(c).startswith(f"{text_col}_emb")]
    arr = tdf[emb_cols].to_numpy(dtype="float32")
    vec_str = ["[" + ",".join(f"{x:.6g}" for x in row) + "]" for row in arr]
    out = tdf[pk].copy()
    out[text_col] = pd.Series(vec_str, index=tdf.index, dtype="string")
    return out

def update_text_vectors_for_table(table_name: str, chunksize: int = 20_000):
    db = DatabaseConnector()
    qh = QueryHelper(db)
    schema = "public"
    table_pre = f"{table_name}_preprocessd"
    pk_cols = TABLE_PK_MAP[table_name]

    # 1) 벡터 컬럼 보장
    db.ensure_pgvector_and_column(schema=schema, table=table_pre, vec_col=TEXT_COL, dims=TEXT_EMB_DIM)

    # 2) PK+TEXT만 청크로 읽어서 → 임베딩 → temp COPY → UPDATE
    sel = qh.select_pk_and_text(table_name, pk_cols=pk_cols, text_col=TEXT_COL)
    for chunk in tqdm(db.iter_sql_chunks(sel, chunksize=chunksize), desc=f"[TEXT→UPDATE] {table_name}"):
        vec_df = embed_text_chunk(chunk, table_name=table_name, text_col=TEXT_COL)
        db.copy_temp_and_update_vector(schema=schema, table=table_pre,
                                       pk_cols=pk_cols, vec_col=TEXT_COL,
                                       df_vec=vec_df, dims=TEXT_EMB_DIM)
        del chunk, vec_df
        _free()