# -*- coding: cp949 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys

# ----------------------------------------------------------------------
# 0. 환경 설정 (로컬 PC)
# ----------------------------------------------------------------------
# 서버 2 (DB 서버) 접속 정보
DB_HOST = '192.168.0.100'  
DB_NAME = 'GFCON_PSQL'
DB_USER = 'postgres'
DB_PASS = '0000'

# 저장된 테이블 정보
DB_SCHEMA = 'public'
TABLE_NAME = 'similarity_dh_full' # 3.9억 건이 저장된 테이블

# 컬럼명
COL_NOTICE_ID = 'bidntceno'
COL_SCORE = 'similarity_score'

# ----------------------------------------------------------------------
# 1. DB에서 "공고별 최소 유사도"만 가져오기 (핵심!)
# ----------------------------------------------------------------------
print("--- DB에 접속하여 공고별 최소 유사도를 집계 중입니다... ---")
print("(데이터가 많아 DB 서버에서 집계하는 데 시간이 조금 걸릴 수 있습니다)")

try:
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')
    
    # [최적화] 3.9억 건을 다 가져오지 않고, DB에서 MIN()을 수행해서 가져옴
    # 결과는 공고 개수만큼(약 200만 개)만 리턴되므로 로컬 메모리에 충분함
    sql_query = f"""
        SELECT {COL_NOTICE_ID}, MIN({COL_SCORE}) as min_score
        FROM {DB_SCHEMA}.{TABLE_NAME}
        GROUP BY {COL_NOTICE_ID}
    """
    
    # 데이터 로드
    df_min_scores = pd.read_sql(sql_query, engine)
    
    print(f"\n[로드 완료] 집계된 공고 수: {len(df_min_scores):,} 건")
    
except Exception as e:
    print(f"오류 발생: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# 2. 임계값(하위 10% 분위수) 계산
# ----------------------------------------------------------------------
if len(df_min_scores) > 0:
    min_scores = df_min_scores['min_score'].values
    
    # 분위수 계산 (10%)
    target_percentile = 10
    threshold = np.percentile(min_scores, target_percentile)
    
    print("\n=========================================================")
    print(f"  [최종 분석 결과] 유사도 임계값 결정")
    print("=========================================================")
    print(f" 대상 데이터: 실제 입찰이 있었던 공고 {len(min_scores):,}개")
    print(f" 전략: 각 공고별 최저 유사도 점수들의 하위 {target_percentile}% 지점")
    print("---------------------------------------------------------")
    print(f" >>> 최종 임계값 (Threshold): {threshold:.6f}")
    print("=========================================================")
    
    # 추가 통계 정보
    print(f"\n[참고] 최소 유사도 분포 통계")
    print(f" - 평균: {np.mean(min_scores):.4f}")
    print(f" - 중위수(50%): {np.median(min_scores):.4f}")
    print(f" - 최소값: {np.min(min_scores):.4f}")
    print(f" - 최대값: {np.max(min_scores):.4f}")
    
else:
    print("데이터가 없습니다.")