# -*- coding: cp949 -*-
import os
import gc

# CPU 자원 최대 활용
# os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import time
from typing import Dict, Set, Tuple

# ----------------------------------------------------------------------
# 0. Configuration (Server 1)
# ----------------------------------------------------------------------
NPZ_PATH_NOTICE = '/data/dev/jodalroB-twoTower/data/embeddings/notice.npz'
NPZ_PATH_COMPANY = '/data/dev/jodalroB-twoTower/data/embeddings/company.npz'

DB_HOST = '192.168.0.100'
DB_NAME = 'GFCON_PSQL'
DB_USER = 'postgres'
DB_PASS = '0000'

# [설정] 공고 20만 개 샘플링
EVAL_SAMPLE_SIZE = 200000     

# [설정] -0.5 ~ 0.55 까지 0.005 단위로 스캔
THRESHOLDS_TO_TEST = np.round(np.arange(-0.5, 0.5501, 0.005), 3).tolist()

BATCH_SIZE = 100              

DB_SCHEMA = 'data'
BID_TABLE = 'bid'
COL_NOTICE = 'bidntceno'      
COL_ORD = 'bidntceord'        
COL_COMPANY = 'bidprccorpbizrno'


# ----------------------------------------------------------------------
# 1. 데이터 로드 함수 (메모리 최적화)
# ----------------------------------------------------------------------
def load_sample_notices_only(path: str, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """공고 데이터 로드 및 샘플링 (메모리 즉시 해제)"""
    print(f"1. Loading Notice NPZ from {path}...")
    try:
        data = np.load(path, allow_pickle=True)
        ids = data['ids']
        vectors = data['embeddings']
        
        total_count = len(ids)
        print(f"   Total Notices: {total_count:,}")
        
        if total_count <= sample_size:
            indices = np.arange(total_count)
        else:
            indices = np.random.choice(total_count, sample_size, replace=False)
            
        sample_ids = ids[indices]
        sample_vecs = vectors[indices]
        
        del data, ids, vectors
        gc.collect()
        
        if sample_ids.ndim > 1:
            sample_ids = sample_ids.flatten()
        sample_ids = np.char.strip(sample_ids.astype(str))
        
        norms = np.linalg.norm(sample_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sample_vecs = sample_vecs / norms
        
        return sample_ids, sample_vecs

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def load_all_companies(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """업체 데이터 로드 및 인덱스 맵핑 생성"""
    print(f"2. Loading Company NPZ from {path}...")
    try:
        data = np.load(path, allow_pickle=True)
        ids = data['ids']
        vectors = data['embeddings']
        
        if ids.ndim > 1:
            ids = ids.flatten()
        ids = np.char.strip(ids.astype(str))
        
        # 업체 ID를 배열 인덱스로 빠르게 찾기 위한 맵 생성
        idx_map = {cid: i for i, cid in enumerate(ids)}
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms
        
        return ids, normalized_vectors, idx_map
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def get_ground_truth_optimized(engine, sample_composite_ids: np.ndarray) -> Dict[str, Set[str]]:
    """샘플링된 공고의 정답지만 DB 조회"""
    print(f"3. Fetching Ground Truth for {len(sample_composite_ids):,} samples...")
    
    plain_ids = set()
    for cid in sample_composite_ids:
        cid_str = str(cid).strip()
        # ID 분리 로직 (가장 오른쪽 하이픈 기준)
        if '-' in cid_str:
            plain_ids.add(cid_str.rsplit('-', 1)[0])
        else:
            plain_ids.add(cid_str)
            
    ids_list = ["'" + str(pid) + "'" for pid in list(plain_ids)]
    if not ids_list:
        return {}
    
    # 쿼리 최적화 (IN 절 분할 처리)
    chunk_size = 10000
    all_dfs = []
    
    try:
        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i:i + chunk_size]
            ids_tuple_str = ",".join(chunk)
            
            query = f"""
                SELECT {COL_NOTICE}, {COL_ORD}, {COL_COMPANY} 
                FROM {DB_SCHEMA}.{BID_TABLE}
                WHERE {COL_NOTICE} IN ({ids_tuple_str});
            """
            df_chunk = pd.read_sql(query, engine)
            all_dfs.append(df_chunk)
            
        if not all_dfs:
            return {}
        df = pd.concat(all_dfs, ignore_index=True)
        
        if df.empty:
            return {}

        # Composite Key 재조립
        df['composite_id'] = (
            df[COL_NOTICE].astype(str).str.strip() + '-' + 
            df[COL_ORD].astype(str).str.strip().str.zfill(3)
        )
        df[COL_COMPANY] = df[COL_COMPANY].astype(str).str.strip()
        
        target_set = set(sample_composite_ids)
        df_filtered = df[df['composite_id'].isin(target_set)]
        
        return df_filtered.groupby('composite_id')[COL_COMPANY].apply(set).to_dict()
        
    except Exception as e:
        print(f"DB Query Error: {e}")
        sys.exit(1)


# ----------------------------------------------------------------------
# 2. 평가 로직 (MAE 중심 분석)
# ----------------------------------------------------------------------
def evaluate_performance_mae_focus(
    n_ids_target: np.ndarray,
    n_vecs_target: np.ndarray,
    c_vecs: np.ndarray,
    company_idx_map: Dict[str, int],
    ground_truth_map: Dict[str, Set[str]]
):
    total_notices = len(n_ids_target)
    
    # 통계 저장소
    stats = {
        th: {'pred_sum': 0, 'actual_sum': 0, 'error_sum': 0}
        for th in THRESHOLDS_TO_TEST
    }
    
    print(f"\n--- Evaluation Started (Focus: MAE, Sample: {total_notices:,}) ---")
    start_time = time.time()

    for i in range(0, total_notices, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_notices)
        n_batch_vecs = n_vecs_target[i:batch_end]
        n_batch_ids = n_ids_target[i:batch_end]
        
        # 행렬 연산
        scores_matrix = np.dot(n_batch_vecs, c_vecs.T)
        
        for idx, notice_id in enumerate(n_batch_ids):
            if notice_id not in ground_truth_map:
                continue
            
            actual_companies = ground_truth_map[notice_id]
            num_actual = len(actual_companies)
            if num_actual == 0:
                continue
            
            scores_row = scores_matrix[idx]
            
            for th in THRESHOLDS_TO_TEST:
                num_predicted = np.sum(scores_row >= th)
                error = abs(num_predicted - num_actual)
                
                stats[th]['actual_sum'] += num_actual
                stats[th]['pred_sum'] += num_predicted
                stats[th]['error_sum'] += error

        if (i + BATCH_SIZE) % 5000 == 0:
            print(f" Progress: {batch_end}/{total_notices} ...")

    elapsed = time.time() - start_time
    
    # ------------------------------------------------------------------
    # 결과 리포트
    # ------------------------------------------------------------------
    results_list = []
    valid_notice_count = len(ground_truth_map) or 1
    
    for th in THRESHOLDS_TO_TEST:
        data = stats[th]
        if data['actual_sum'] == 0:
            continue
        
        avg_actual = data['actual_sum'] / valid_notice_count
        avg_pred = data['pred_sum'] / valid_notice_count
        
        mae = data['error_sum'] / valid_notice_count
        ratio = avg_pred / avg_actual if avg_actual > 0 else 9999.0
        diff = avg_pred - avg_actual
        
        results_list.append({
            'th': th,
            'mae': mae,
            'ratio': ratio,
            'diff': diff,
            'avg_pred': avg_pred,
            'avg_actual': avg_actual
        })

    sorted_results = sorted(results_list, key=lambda x: x['mae'])

    print("\n==========================================================================================")
    print(f"  [Best Thresholds Top 10] (Sorted by MAE: Mean Absolute Error)")
    print("==========================================================================================")
    print(f"{'Rank':<4} | {'Threshold':<10} | {'MAE (Error)':<12} | {'Pred Ratio':<12} | {'Avg Pred':<10} | {'Avg Actual':<10} | {'Diff':<10}")
    print("-" * 105)
    
    for rank, res in enumerate(sorted_results[:10], 1):
        th = res['th']
        mae = res['mae']
        ratio = res['ratio']
        avg_p = res['avg_pred']
        avg_a = res['avg_actual']
        diff = res['diff']
        
        comment = ""
        if rank == 1:
            comment = "★ Best Fit"
        elif ratio <= 2.0:
            comment = "(Good Ratio)"
        
        diff_str = f"{diff:+.1f}"
        print(
            f"{rank:<4} | {th:<10.3f} | +/- {mae:<9.1f} | {ratio:<11.2f}x | "
            f"{avg_p:<10.1f} | {avg_a:<10.1f} | {diff_str:<10} {comment}"
        )
        
    print("==========================================================================================")
    print(f" Total Time: {elapsed/60:.2f} min")


# ----------------------------------------------------------------------
# 3. Main Execution
# ----------------------------------------------------------------------
def main():
    # 1) 공고 20만 개 샘플링
    n_ids_target, n_vecs_target = load_sample_notices_only(
        NPZ_PATH_NOTICE,
        EVAL_SAMPLE_SIZE
    )
    
    # 2) 정답셋 조회
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')
    ground_truth_map = get_ground_truth_optimized(engine, n_ids_target)
    
    if not ground_truth_map:
        print("\n[Warning] No matching bid records found.")
        sys.exit(0)
        
    print(f" -> Found ground truth for {len(ground_truth_map):,} notices.")

    # 3) 업체 임베딩 전체 로드
    c_ids, c_vecs, company_idx_map = load_all_companies(NPZ_PATH_COMPANY)
    
    # 4) 임계값 탐색
    evaluate_performance_mae_focus(
        n_ids_target,
        n_vecs_target,
        c_vecs,
        company_idx_map,
        ground_truth_map
    )


if __name__ == "__main__":
    main()

