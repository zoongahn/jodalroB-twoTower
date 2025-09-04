import os, json
from typing import Any, Iterable, Optional, Dict, List
import numpy as np
import pandas as pd
from sqlalchemy import table

# 충돌 완화: psycopg 구현을 파이썬으로 고정(선택)
os.environ.setdefault("PSYCOPG_IMPL", "python")

import psycopg  # 먼저 import

# TensorFlow 설정을 맨 위에서 한 번만
def setup_tensorflow(use_gpu: bool = True):
    import tensorflow as tf
    if use_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU 사용 가능: {len(gpus)}개")
            except RuntimeError as e:
                print(f"❌ GPU 설정 실패: {e}")
        else:
            print("❌ GPU를 찾을 수 없습니다.")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("💻 CPU만 사용합니다.")
    return tf

# 모듈 레벨에서 한 번만 실행
tf = setup_tensorflow(use_gpu=True)

def fetch_table_batch(conn, table: str, order_by: Optional[List[str]], limit: int, offset: int) -> pd.DataFrame:
    order_sql = f' ORDER BY {", ".join(order_by)} ' if order_by else ""
    sql = f"SELECT * FROM {table}{order_sql} LIMIT %s OFFSET %s"
    with conn.cursor() as cur:
        cur.execute(sql, (limit, offset))
        rows = cur.fetchall()
        if rows:
            # 컬럼명 가져오기
            columns = [desc[0] for desc in cur.description]
            # 튜플을 딕셔너리로 변환
            dict_rows = [dict(zip(columns, row)) for row in rows]
            return pd.DataFrame(dict_rows)
    return pd.DataFrame()

def db_iter_rows(
    dsn: str,
    table: str,
    order_by: Optional[List[str]] = None,
    batch_size: int = 50_000,
):
    # row_factory=dict 제거
    with psycopg.connect(dsn) as conn:
        offset = 0
        while True:
            df = fetch_table_batch(conn, table, order_by, batch_size, offset)
            if df.empty: break
            yield df
            if len(df) < batch_size: break
            offset += batch_size

# === TFRecord 직렬화 유틸(동일) ===
def _floats_feature(xs): return tf.train.Feature(float_list=tf.train.FloatList(value=list(map(float, xs))))
def _ints_feature(xs):   return tf.train.Feature(int64_list=tf.train.Int64List(value=list(map(int, xs))))
def _bytes_feature(v):   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _int_feature(v):     return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
def _float_feature(v):   return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

def _maybe_parse_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    if isinstance(x, (list, tuple, np.ndarray)): return list(x)
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try: import json; return json.loads(s)
            except Exception: return None
    return None

def _to_feature(v):
    ml = _maybe_parse_list(v)
    if ml is not None:
        if len(ml) == 0: return None
        if all(isinstance(x, (int, np.integer, bool)) or (isinstance(x, float) and float(x).is_integer()) for x in ml):
            return _ints_feature([int(x) for x in ml])
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in ml):
            return _floats_feature([float(x) for x in ml])
        return _bytes_feature(json.dumps(ml, ensure_ascii=False).encode("utf-8"))
    if v is None or (isinstance(v, float) and np.isnan(v)): 
        return None
    
    # 순수 숫자 타입만 숫자로 변환
    if isinstance(v, (int, np.integer, bool)): 
        return _int_feature(int(v))
    if isinstance(v, (float, np.floating)):    
        return _float_feature(float(v))
    
    # 문자열은 숫자 변환 시도하지 않고 바로 bytes로
    if isinstance(v, str):
        return _bytes_feature(v.encode("utf-8"))
    
    # 기타
    return _bytes_feature(str(v).encode("utf-8"))

def serialize_row(row: pd.Series) -> bytes:
    feats = {}
    for k, v in row.items():
        ft = _to_feature(v)
        if ft is not None: feats[k] = ft
    return tf.train.Example(features=tf.train.Features(feature=feats)).SerializeToString()

def table_to_tfrecord_psycopg3(
    dsn: str,
    table_name: str,
    batch_size: int = 50_000,
    order_by: Optional[List[str]] = None,
    compression: str = "GZIP",
):
    output_path: str = f"output/tfrecord/{table_name}.tfrecord.gz"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    options = tf.io.TFRecordOptions(compression_type=compression)

    written = 0
    with tf.io.TFRecordWriter(output_path, options=options) as w:
        for df in db_iter_rows(dsn, table_name, order_by=order_by, batch_size=batch_size):
            df.columns = [str(c) for c in df.columns]
            for _, row in df.iterrows():
                w.write(serialize_row(row))
            written += len(df)
            print(f"➡️  wrote {len(df):,} rows (total={written:,})")
    print(f"✅ done: {written:,} rows → {output_path}")
    
# =========================
# Example CLI
# =========================
if __name__ == "__main__":
    # PK가 있다면 정렬 컬럼으로 주는 걸 권장(재현성)
    dsn = "host=localhost port=5432 user=postgres password=0000 dbname=GFCON"
    table_to_tfrecord_psycopg3(dsn,
                               table_name="bid_two_tower"
                               )