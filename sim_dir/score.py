# -*- coding: cp949 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import time
from typing import Dict, Set, Tuple

# ----------------------------------------------------------------------
# 0. 환경 설정 (Server 1)
# ----------------------------------------------------------------------
# A. 파일 경로
NPZ_PATH_NOTICE = '/data/dev/jodalroB-twoTower/data/embeddings/notice.npz'
NPZ_PATH_COMPANY = '/data/dev/jodalroB-twoTower/data/embeddings/company.npz'

# B. DB 접속 정보 (정답지 조회용)
DB_HOST = '192.168.0.100'
DB_NAME = 'GFCON_PSQL'
DB_USER = 'postgres'
DB_PASS = '0000'

# C. 평가 설정
EVAL_SAMPLE_SIZE = 10000      # 평가할 공고 샘플 수 (최소 1만 개 권장)
THRESHOLD = -0.5447           # [중요] 설정한 임계값
BATCH_SIZE = 100              # 한 번에 처리할 공고 수 (메모리 보호)

# D. 데이터 스키마
DB_SCHEMA = 'data'
BID_TABLE = 'bid'
COL_NOTICE = 'bidntceno'
COL_COMPANY = 'bidprccorpbizrno'


# ----------------------------------------------------------------------
# 1. 데이터 로드 및 정규화
# ----------------------------------------------------------------------
def load_and_normalize_embeddings(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """NPZ 로드 및 L2 정규화"""
    try:
        print(f"[{path}] 로드 중...")
        data = np.load(path, allow_pickle=True)
        ids = data['ids']
        vectors = data['embeddings']

        if ids.ndim > 1: ids = ids.flatten()
        
        # 정규화
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms
        
        # ID 문자열 변환 및 공백 제거
        ids = np.char.strip(ids.astype(str))
        return ids, normalized_vectors

    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)

def load_ground_truth(engine) -> Dict[str, Set[str]]:
    """
    DB에서 실제 입찰 정보를 읽어 {공고ID: {실제참여업체ID들}} 형태의 맵 생성
    """
    print("DB에서 정답지(Ground Truth) 로드 중...")
    try:
        query = f"SELECT {COL_NOTICE}, {COL_COMPANY} FROM {DB_SCHEMA}.{BID_TABLE};"
        df = pd.read_sql(query, engine)
        
        # 문자열 변환 및 공백 제거
        df[COL_NOTICE] = df[COL_NOTICE].astype(str).str.strip()
        df[COL_COMPANY] = df[COL_COMPANY].astype(str).str.strip()
        
        # GroupBy를 통해 공고별 참여 업체 집합(Set) 생성
        ground_truth = df.groupby(COL_NOTICE)[COL_COMPANY].apply(set).to_dict()
        print(f"정답지 로드 완료: 총 {len(ground_truth):,}개의 공고에 대한 입찰 기록 확보")
        return ground_truth
        
    except Exception as e:
        print(f"DB 정답지 로드 실패: {e}")
        sys.exit(1)


# ----------------------------------------------------------------------
# 2. 평가 실행 (배치 처리)
# ----------------------------------------------------------------------
def evaluate_performance(n_ids, n_vecs, c_ids, c_vecs, ground_truth_map):
    total_notices = len(n_ids)
    num_companies = len(c_ids)
    
    # 평가 지표 누적 변수
    total_tp = 0  # True Positive (맞춤)
    total_fp = 0  # False Positive (틀리게 예측함)
    total_fn = 0  # False Negative (놓침)
    
    print(f"\n--- 성능 평가 시작 (Sample: {total_notices:,}개, Threshold: {THRESHOLD}) ---")
    start_time = time.time()

    # 공고 배치 처리
    for i in range(0, total_notices, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_notices)
        
        # 1. 배치 데이터
        n_batch_vecs = n_vecs[i:batch_end]
        n_batch_ids = n_ids[i:batch_end]
        
        # 2. 행렬 연산 (유사도 계산)
        # (Batch, 128) x (128, All_Companies)
        scores_matrix = np.dot(n_batch_vecs, c_vecs.T)
        
        # 3. 개별 공고에 대해 평가 수행
        for idx, notice_id in enumerate(n_batch_ids):
            # 정답지가 없는 공고(DB에 입찰 기록이 없는 공고)는 평가에서 제외
            if notice_id not in ground_truth_map:
                continue
                
            # A. 모델의 예측 (임계값 이상인 업체 인덱스 추출)
            # np.where는 튜플(array,)을 반환하므로 [0]으로 인덱스 배열 가져옴
            pred_indices = np.where(scores_matrix[idx] >= THRESHOLD)[0]
            predicted_companies = set(c_ids[pred_indices])
            
            # B. 실제 정답 (Ground Truth)
            actual_companies = ground_truth_map[notice_id]
            
            # C. 혼동 행렬(Confusion Matrix) 요소 계산
            # TP: 예측했고, 실제로도 참여함 (교집합)
            tp = len(predicted_companies.intersection(actual_companies))
            
            # FP: 예측했으나, 실제로는 참여 안 함 (예측 집합 - 정답 집합)
            fp = len(predicted_companies - actual_companies)
            
            # FN: 예측 못 했으나, 실제로는 참여함 (정답 집합 - 예측 집합)
            fn = len(actual_companies - predicted_companies)
            
            # 누적
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # 진행 상황 출력
        if (i + BATCH_SIZE) % 1000 == 0:
            print(f" 진행률: {batch_end}/{total_notices} 완료... (현재 TP:{total_tp}, FP:{total_fp}, FN:{total_fn})")

    elapsed = time.time() - start_time
    
    # ------------------------------------------------------------------
    # 3. 최종 지표 계산 (Micro-Average 방식)
    # ------------------------------------------------------------------
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n========================================================")
    print(f"  [최종 성능 평가 결과] (임계값: {THRESHOLD})")
    print("========================================================")
    print(f" 1. 혼동 행렬 요약")
    print(f"    - True Positive (정답 맞춤) : {total_tp:,} 건")
    print(f"    - False Positive (과잉 예측): {total_fp:,} 건")
    print(f"    - False Negative (놓친 정답): {total_fn:,} 건")
    print("--------------------------------------------------------")
    print(f" 2. 성능 지표")
    print(f"    - Precision (정밀도) : {precision:.4f}")
    print(f"    - Recall    (재현율) : {recall:.4f}")
    print(f"    - F1-Score  (조화평균): {f1_score:.4f}")
    print("========================================================")
    print(f" 소요 시간: {elapsed/60:.2f}분")


# ----------------------------------------------------------------------
# 4. Main Execution
# ----------------------------------------------------------------------
def main():
    # 1. DB 연결
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')
    
    # 2. 정답지 로드 (전체 Bid 테이블)
    ground_truth_map = load_ground_truth(engine)
    
    # 3. 임베딩 로드 (전체)
    print("\n임베딩 데이터 로드 중...")
    n_ids_all, n_vecs_all = load_and_normalize_embeddings(NPZ_PATH_NOTICE)
    c_ids, c_vecs = load_and_normalize_embeddings(NPZ_PATH_COMPANY)
    
    # 4. 평가용 샘플링 (정답지에 있는 공고 중에서만 샘플링해야 함)
    # NPZ에도 있고, DB(정답지)에도 있는 공고 ID들의 교집합을 구함
    valid_notice_ids = list(set(n_ids_all).intersection(ground_truth_map.keys()))
    
    if len(valid_notice_ids) < EVAL_SAMPLE_SIZE:
        print(f"주의: 평가 가능한 공고 수({len(valid_notice_ids)})가 목표 샘플 수({EVAL_SAMPLE_SIZE})보다 적습니다. 전체를 평가합니다.")
        sample_ids = valid_notice_ids
    else:
        # 랜덤 샘플링
        sample_ids = np.random.choice(valid_notice_ids, EVAL_SAMPLE_SIZE, replace=False)
    
    print(f" -> 평가 대상 공고 선정 완료: {len(sample_ids):,} 개")
    
    # 샘플 ID에 해당하는 벡터 인덱스 찾기 (맵핑)
    # 속도를 위해 dictionary 생성
    notice_idx_map = {nid: i for i, nid in enumerate(n_ids_all)}
    target_indices = [notice_idx_map[nid] for nid in sample_ids]
    
    n_ids_target = n_ids_all[target_indices]
    n_vecs_target = n_vecs_all[target_indices]
    
    # 5. 평가 실행
    evaluate_performance(n_ids_target, n_vecs_target, c_ids, c_vecs, ground_truth_map)

if __name__ == "__main__":
    main()