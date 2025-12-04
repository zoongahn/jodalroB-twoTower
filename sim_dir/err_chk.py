# -*- coding: cp949 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
from typing import Dict

# ----------------------------------------------------------------------
# 0. 환경 설정
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
DB_NOTICE_ORD_COL = 'bidntceord'      # 공고차수
DB_COMPANY_ID_COL = 'bidprccorpbizrno'  # 업체

# D. 한 번에 검사할 행 수
BATCH_SIZE = 200000


# ----------------------------------------------------------------------
# 1. 임베딩 로드 + 정규화
# ----------------------------------------------------------------------
def load_and_normalize_embeddings(path: str) -> Dict[str, np.ndarray]:
    """
    NPZ 파일 로드 후 L2 정규화(Normalize) 수행.
    정규화된 벡터는 내적(Dot Product)만으로 코사인 유사도 계산이 가능함.

    반환:
        { id(str) : 정규화된 벡터(np.ndarray) }
    """
    try:
        data = np.load(path, allow_pickle=True)
        ids = data['ids']
        vectors = data['embeddings']

        if ids.ndim > 1:
            ids = ids.flatten()

        print(f"[{path}] 임베딩 개수: {len(ids):,}개")
        print(f"[{path}] 벡터 정규화(L2 Normalize) 수행 중...")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms

        keys = [str(x).strip() for x in ids]
        emb_map = {k: vec for k, vec in zip(keys, normalized_vectors)}

        print(f"[{path}] 로드 완료 - 키 개수: {len(emb_map):,}개")
        return emb_map

    except Exception as e:
        print(f"오류: NPZ 파일 로드 실패 ({path}): {e}")
        sys.exit(1)


# ----------------------------------------------------------------------
# 2. 한 개 배치만 검사
# ----------------------------------------------------------------------
def debug_one_batch(engine, notice_map: Dict[str, np.ndarray], company_map: Dict[str, np.ndarray]):
    print("\n--- 2. bid 테이블에서 한 개 배치 로드 ---")

    query = f"""
        SELECT {DB_NOTICE_ID_COL}, {DB_NOTICE_ORD_COL}, {DB_COMPANY_ID_COL}
        FROM {DB_SCHEMA_SOURCE}.{BID_TABLE_NAME}
        LIMIT {BATCH_SIZE};
    """

    df = pd.read_sql(query, engine)
    total = len(df)
    print(f"불러온 행 수: {total:,}건")

    if total == 0:
        print("데이터가 없습니다.")
        return

    # 샘플 출력
    print("\n[DEBUG] DB에서 읽은 샘플 5건:")
    print(df.head())

    valid_count = 0
    invalid_rows = []  # 매칭 실패한 행 샘플

    n_raw_list = df[DB_NOTICE_ID_COL].tolist()
    o_raw_list = df[DB_NOTICE_ORD_COL].tolist()
    c_raw_list = df[DB_COMPANY_ID_COL].tolist()

    for n_raw, o_raw, c_raw in zip(n_raw_list, o_raw_list, c_raw_list):
        nid_str = str(n_raw).strip()
        ord_str = str(o_raw).strip().zfill(3)   # 0 -> "000"
        notice_key = nid_str + "-" + ord_str    # 예: "20160305458-000"

        cid_str = str(c_raw).strip()

        has_notice = notice_key in notice_map
        has_company = cid_str in company_map

        if has_notice and has_company:
            valid_count += 1
        else:
            if len(invalid_rows) < 100:  # 샘플 100건까지만 저장
                invalid_rows.append({
                    "bidntceno": n_raw,
                    "bidntceord": o_raw,
                    "notice_key(조합)": notice_key,
                    "bidprccorpbizrno": c_raw,
                    "has_notice_emb": has_notice,
                    "has_company_emb": has_company,
                })

    invalid_count = total - valid_count

    print("\n--- 결과 요약 (한 개 배치 기준) ---")
    print(f"총 행 수: {total:,}건")
    print(f"유효(둘 다 임베딩 존재) 행 수: {valid_count:,}건")
    print(f"유효하지 않은 행 수: {invalid_count:,}건")

    if invalid_rows:
        print("\n[DEBUG] 유효하지 않은 행 샘플 (최대 100건):")
        df_invalid = pd.DataFrame(invalid_rows)
        print(df_invalid)
    else:
        print("\n유효하지 않은 행이 없습니다. (전부 임베딩 매칭 성공)")


# ----------------------------------------------------------------------
# 3. main
# ----------------------------------------------------------------------
def main():
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')

    print("--- 1. 임베딩 로드 및 정규화 ---")
    notice_map = load_and_normalize_embeddings(NPZ_PATH_NOTICE)
    company_map = load_and_normalize_embeddings(NPZ_PATH_COMPANY)

    print("\n임베딩 로드 완료. 한 개 배치 디버그 시작...")
    debug_one_batch(engine, notice_map, company_map)


if __name__ == "__main__":
    main()

