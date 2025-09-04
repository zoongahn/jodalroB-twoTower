
# pipeline.py (function-to-function, no CSV intermediate)
from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, List, Optional
import pandas as pd
import json
import os
import io
import gc
from psycopg import sql
from tqdm import tqdm
import torch

# Import your in-project preprocessors
from data.database_connector import DatabaseConnector
from data.query_helper import QueryHelper
from preprocess.numeric_preprocess import NumericPreprocessor
from preprocess.categorical_preprocess import CategoricalPreprocessor
from preprocess.text_preprocess import TextPreprocessor
from preprocess.upload_database import DataUploader

# Shared PK mapping (keep consistent with each module)
TABLE_PK_MAP = {
    'notice': ['bidntceno', 'bidntceord'],
    'company': ['bizno'],
}

TEXT_MODEL_NAME = os.getenv("TEXT_EMBEDDING_MODEL")

TEXT_COL = "bidntcenm"   # 텍스트 임베딩 대상 컬럼
TEXT_EMB_DIM = 768       # 사용하는 임베딩 차원



# ===== 설정 ==================================================================
def _read_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 설정 파일 없음: {path} (빈 설정으로 진행)")
        return {}
    except Exception as e:
        print(f"⚠️ 설정 로드 실패: {path} -> {e} (빈 설정으로 진행)")
        return {}
    

def load_preprocess_configs(table_name: str, meta_dir: str = "meta") -> Tuple[Dict, Dict, Dict]:
    """
    /meta/{table_name}_{type}_config.json 규칙으로 설정 로드
    - type ∈ {numeric, categorical, text}
    - 없으면 {} 반환
    """
    t = table_name.strip().lower()
    files = {
        "numeric":     os.path.join(meta_dir, f"{t}_numeric_config.json"),
        "categorical": os.path.join(meta_dir, f"{t}_categorical_config.json"),
        "text":        os.path.join(meta_dir, f"{t}_text_config.json"),
    }
    num_cfg = _read_json(files["numeric"])
    cat_cfg = _read_json(files["categorical"])
    txt_cfg = _read_json(files["text"])
    return num_cfg, cat_cfg, txt_cfg


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_preprocessors(num_cfg, cat_cfg, txt_cfg, model_name: str | None = None):
    num_pp = NumericPreprocessor(config=num_cfg, table_pk_map=TABLE_PK_MAP)
    cat_pp = CategoricalPreprocessor(config=cat_cfg, table_pk_map=TABLE_PK_MAP)
    # 텍스트 전처리기는 한 번만 만들고 여러 청크에서 재사용 (모델 캐시 유지)
    if model_name:
        txt_cfg = {k: {**v, "model_name": model_name} for k, v in txt_cfg.items()}
    txt_pp = TextPreprocessor(config=txt_cfg, table_pk_map=TABLE_PK_MAP)
    return num_pp, cat_pp, txt_pp


def fit_once_full(engine, qh: QueryHelper, table: str, num_pp, cat_pp):
    """수치/범주 fit을 한 번에. 텍스트 컬럼은 제외해서 메모리 절약."""
    sql = qh.get_use_columns_select(table, exclude_text_cols=[TEXT_COL])
    df = pd.read_sql_query(sql, engine)
    # PK는 문자열 보존
    for pk in TABLE_PK_MAP[table]:
        if pk in df.columns:
            df[pk] = df[pk].astype("string").fillna("")
    # fit (텍스트는 fit 필요 없음)
    if hasattr(num_pp, "fit"): num_pp.fit(df)
    if hasattr(cat_pp, "fit"): cat_pp.fit(df)
    del df
    _free()

def _split_features(df: pd.DataFrame, pk: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Return (feature_columns_without_null_flags, null_flag_columns)."""
    pk = list(pk or [])
    cols = list(map(str, df.columns))

    nulls = [c for c in cols if c.endswith("_is_null")]
    feats = [c for c in cols if c not in pk and not c.endswith("_is_null")]
    return feats, nulls


def transform_one_chunk(df_chunk: pd.DataFrame, table: str, num_pp, cat_pp, txt_pp) -> pd.DataFrame:
    """하나의 청크를 num/cat/text 모두 transform 후 PK 기준으로 병합해 반환."""
    pk = TABLE_PK_MAP[table]
    for c in pk:
        if c in df_chunk.columns:
            df_chunk[c] = df_chunk[c].astype("string").fillna("")
    # transform만 호출 (fit 금지)
    num_df = num_pp.transform(df_chunk, table) if num_pp else df_chunk[pk].copy()
    cat_df = cat_pp.transform(df_chunk, table) if cat_pp else df_chunk[pk].copy()
    txt_df = txt_pp.transform(df_chunk, table) if txt_pp else df_chunk[pk].copy()

    # merge
    merged = num_df.merge(cat_df, on=pk, how="left", validate="one_to_one")
    merged = merged.merge(txt_df, on=pk, how="left", validate="one_to_one")
    
    # 컬럼명 문자열화 안전장치
    merged.columns = merged.columns.map(str)
    return merged

def run_pipeline(table_name:str):
    db = DatabaseConnector()
    qh = QueryHelper(db)

    table = table_name  # 필요 시 company도 동일 로직 복제
    num_cfg, cat_cfg, txt_cfg = load_preprocess_configs(table)  # 너의 기존 로더 사용
    text_model = os.getenv("TEXT_EMBEDDING_MODEL")

    print("🚀 수치/범주 fit 1회 수행")
    num_pp, cat_pp, txt_pp = build_preprocessors(num_cfg, cat_cfg, txt_cfg, model_name=text_model)
    fit_once_full(db.engine, qh, table, num_pp, cat_pp)

    print("🚀 청크 단위 transform + 업로드 시작")
    # 모든 사용 컬럼(텍스트 포함) SELECT
    select_sql = qh.get_use_columns_select(table)  # 텍스트 제외 X
    # 진행률 total이 필요하면 count_rows 사용
    # total = db.count_rows(qh.get_use_columns_count(table))

    uploader = DataUploader()
    uploader.if_exists = "replace"
    pk_cols = TABLE_PK_MAP[table]

    for i, chunk in enumerate(tqdm(db.iter_sql_chunks(select_sql, chunksize=50_000),
                                   desc=f"[XFORM/COPY] {table}")):
        out = transform_one_chunk(chunk, table, num_pp, cat_pp, txt_pp)
        uploader.upload_df(out, base_table_name=table, pk_cols=pk_cols)
        if i == 0:
            uploader.if_exists = "append"
        del chunk, out
        _free()

if __name__ == "__main__":
    run_pipeline("company")