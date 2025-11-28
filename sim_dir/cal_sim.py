# -*- coding: cp949 -*-
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.types import BigInteger, Float, String
import sys
import time
import io
from typing import Dict

# ----------------------------------------------------------------------
# 0. 환경 설정 (Server 1)
# ----------------------------------------------------------------------
# A. 파일 경로
NPZ_PATH_NOTICE = '/data/dev/jodalroB-twoTower/data/embeddings/notice.npz'
NPZ_PATH_COMPANY = '/data/dev/jodalroB-twoTower/data/embeddings/company.npz'

# B. DB 접속 정보
DB_HOST = '192.168.0.100'
DB_NAME = 'GFCON_PSQL'
DB_USER = 'postgres'
DB_PASS = '0000'

# C. 소스 테이블 (입찰 기록)
DB_SCHEMA_SOURCE = 'data'
BID_TABLE_NAME = 'bid'
DB_NOTICE_ID_COL = 'bidntceno'        # 공고번호
DB_NOTICE_ORD_COL = 'bidntceord'      # 공고차수 (0,1,2,...)
DB_COMPANY_ID_COL = 'bidprccorpbizrno'  # 업체 사업자번호 등

# D. 타겟 테이블 (결과 저장)
DB_SCHEMA_TARGET = 'public'
BID_SCORES_TABLE_NAME = 'similarity_dh_full'   # 원하는 이름으로 변경 가능
SIMILARITY_SCORE_COL = 'similarity_score'

# E. 배치 설정
BATCH_SIZE = 200000   # 한 번에 처리할 행 수 (메모리 상황에 맞게 조정)


# ----------------------------------------------------------------------
# 1. 헬퍼 함수: 임베딩 데이터 로드 (정규화 포함)
# ----------------------------------------------------------------------
def load_and_normalize_embeddings(path: str) -> Dict[str, np.ndarray]:
    """
    NPZ 파일 로드 후 L2 정규화(Normalize) 수행.
    정규화된 벡터는 내적(Dot Product)만으로 코사인 유사도 계산이 가능함.

    반환값:
        { id(str) : 정규화된 벡터(np.ndarray) }
    """
    try:
        data = np.load(path, allow_pickle=True)
        ids = data['ids']
        vectors = data['embeddings']

        # ID 배열 평탄화
        if ids.ndim > 1:
            ids = ids.flatten()

        print(f"[{path}] 벡터 정규화(L2 Normalize) 수행 중...")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 0으로 나누기 방지
        normalized_vectors = vectors / norms

        # 문자열 키로 통일 (양쪽 strip)
        keys = [str(x).strip() for x in ids]

        emb_map = {k: vec for k, vec in zip(keys, normalized_vectors)}
        print(f"[{path}] 임베딩 로드 완료 - 개수: {len(emb_map):,}개")
        return emb_map

    except Exception as e:
        print(f"오류: NPZ 파일 로드 실패 ({path}): {e}")
        sys.exit(1)


# ----------------------------------------------------------------------
# 2. 고속 DB 저장 함수 (COPY 사용)
# ----------------------------------------------------------------------
def fast_copy_to_db(df: pd.DataFrame, engine, table_name: str, schema: str):
    """
    PostgreSQL COPY 명령어를 사용하여 고속 저장.
    3억 건 이상의 대용량 데이터에 적합.
    """
    csv_io = io.StringIO()
    df.to_csv(csv_io, sep='\t', header=False, index=False)
    csv_io.seek(0)

    conn = engine.raw_connection()
    cur = conn.cursor()
    
    try:
        sql = (
            f"COPY {schema}.{table_name} "
            f"({DB_NOTICE_ID_COL}, {DB_NOTICE_ORD_COL}, "
            f"{DB_COMPANY_ID_COL}, {SIMILARITY_SCORE_COL}) "
            "FROM STDIN WITH CSV DELIMITER '\t'"
        )
        cur.copy_expert(sql=sql, file=csv_io)
        conn.commit()
    except Exception as e:
        print(f"COPY 오류: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ----------------------------------------------------------------------
# 3. 핵심 로직: 배치 단위 고속 계산
# ----------------------------------------------------------------------
def process_batches_optimized(engine, notice_map: Dict[str, np.ndarray], company_map: Dict[str, np.ndarray]):
    print(f"--- 2. DB({DB_HOST})에서 입찰 쌍 로드 및 고속 처리 시작 (Batch: {BATCH_SIZE}) ---")

    # 공고번호 + 공고차수 + 업체ID를 모두 가져오기
    # (더미 업체 '__DEFAULT__' 는 제외)
    query = (
        f"SELECT {DB_NOTICE_ID_COL}, {DB_NOTICE_ORD_COL}, {DB_COMPANY_ID_COL} "
        f"FROM {DB_SCHEMA_SOURCE}.{BID_TABLE_NAME} "
        f"WHERE {DB_COMPANY_ID_COL} <> '__DEFAULT__';"
    )
    
    bid_iterator = pd.read_sql(query, engine, chunksize=BATCH_SIZE)

    total_rows = 0
    total_valid_rows = 0
    start_total_time = time.time()
    
    # 타겟 테이블 초기화: 스키마만 생성 (기존 테이블 덮어쓰기)
    dummy_df = pd.DataFrame(columns=[
        DB_NOTICE_ID_COL, DB_NOTICE_ORD_COL,
        DB_COMPANY_ID_COL, SIMILARITY_SCORE_COL
    ])
    dummy_df.to_sql(
        name=BID_SCORES_TABLE_NAME,
        con=engine,
        schema=DB_SCHEMA_TARGET,
        if_exists='replace',
        index=False,
        dtype={
            # 공고번호 / 업체ID 모두 문자열 타입
            DB_NOTICE_ID_COL: String(32),
            DB_NOTICE_ORD_COL: BigInteger(),
            DB_COMPANY_ID_COL: String(32),
            SIMILARITY_SCORE_COL: Float(),
        }
    )
    print(f"타겟 테이블({DB_SCHEMA_TARGET}.{BID_SCORES_TABLE_NAME}) 초기화 완료. 전체 처리 시작...")

    for i, df_chunk in enumerate(bid_iterator):
        batch_start = time.time()
        chunk_len = len(df_chunk)
        total_rows += chunk_len

        if i == 0:
            print("[DEBUG] 첫 배치 공고ID 샘플:", df_chunk[DB_NOTICE_ID_COL].head().tolist())
            print("[DEBUG] 첫 배치 공고차수 샘플:", df_chunk[DB_NOTICE_ORD_COL].head().tolist())
            print("[DEBUG] 첫 배치 업체ID 샘플:", df_chunk[DB_COMPANY_ID_COL].head().tolist())
            some_notice_keys = list(notice_map.keys())[:5]
            print("[DEBUG] notice_map 샘플 키:", some_notice_keys)

        n_vecs_list = []
        c_vecs_list = []
        valid_rows = []

        n_raw_list = df_chunk[DB_NOTICE_ID_COL].tolist()
        o_raw_list = df_chunk[DB_NOTICE_ORD_COL].tolist()
        c_raw_list = df_chunk[DB_COMPANY_ID_COL].tolist()

        for n_raw, o_raw, c_raw in zip(n_raw_list, o_raw_list, c_raw_list):
            nid_str = str(n_raw).strip()
            ord_str = str(o_raw).strip().zfill(3)   # 0 -> "000"
            notice_key = nid_str + "-" + ord_str    # 예: "20160305458-000"

            cid_str = str(c_raw).strip()

            if notice_key in notice_map and cid_str in company_map:
                n_vecs_list.append(notice_map[notice_key])
                c_vecs_list.append(company_map[cid_str])
                valid_rows.append((nid_str, o_raw, cid_str))  # 공고/업체 문자열로 저장

        if not valid_rows:
            print(f"[Batch {i+1}] 유효한 데이터 없음. 건너뜀. (원본 {chunk_len:,}건)")
            continue

        N_matrix = np.array(n_vecs_list)
        C_matrix = np.array(c_vecs_list)
        
        sim_scores = np.sum(N_matrix * C_matrix, axis=1)

        notice_ids, notice_ords, company_ids = zip(*valid_rows)

        df_results = pd.DataFrame({
            DB_NOTICE_ID_COL: notice_ids,
            DB_NOTICE_ORD_COL: notice_ords,
            DB_COMPANY_ID_COL: company_ids,
            SIMILARITY_SCORE_COL: sim_scores
        })

        fast_copy_to_db(df_results, engine, BID_SCORES_TABLE_NAME, DB_SCHEMA_TARGET)

        batch_time = time.time() - batch_start
        total_valid_rows += len(df_results)
        speed = chunk_len / batch_time if batch_time > 0 else 0.0

        print(
            f"[Batch {i+1}] 원본 {chunk_len:,}건 중 유효 {len(df_results):,}건 처리 "
            f"- {batch_time:.2f}초 ({speed:,.0f} rows/sec)"
        )

    total_time_min = (time.time() - start_total_time) / 60.0
    print(f"\n--- 모든 작업 완료 ---")
    print(f"총 원본 처리 건수: {total_rows:,}건")
    print(f"총 유효(임베딩 매칭 성공) 건수: {total_valid_rows:,}건")
    print(f"총 소요 시간: {total_time_min:.2f}분")


# ----------------------------------------------------------------------
# 4. Main Execution
# ----------------------------------------------------------------------
def main():
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')
    
    print("--- 1. 임베딩 데이터 로드 및 정규화 ---")
    notice_map = load_and_normalize_embeddings(NPZ_PATH_NOTICE)
    company_map = load_and_normalize_embeddings(NPZ_PATH_COMPANY)
    print("임베딩 로드 완료.\n")

    process_batches_optimized(engine, notice_map, company_map)


if __name__ == "__main__":
    main()
