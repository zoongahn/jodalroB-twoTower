#!/usr/bin/env python3
"""
데이터베이스 테이블 용량 조사 스크립트

Usage:
    python scripts/check_db_sizes.py
    python scripts/check_db_sizes.py --detail
    python scripts/check_db_sizes.py --estimate-memory
"""

import argparse
import pandas as pd
from database.database_connector import DatabaseConnector


def format_bytes(bytes_value):
    """바이트를 사람이 읽기 쉬운 형식으로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def check_table_sizes(engine):
    """전체 테이블 크기 조사"""
    query = """
    SELECT
        schemaname AS schema_name,
        tablename AS table_name,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS indexes_size,
        pg_total_relation_size(schemaname||'.'||tablename) AS total_bytes
    FROM pg_tables
    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
    ORDER BY total_bytes DESC;
    """
    df = pd.read_sql(query, engine)
    return df


def check_table_details(engine):
    """테이블 상세 정보 (행 수 포함)"""
    query = """
    SELECT
        schemaname AS schema_name,
        tablename AS table_name,
        n_live_tup AS live_rows,
        n_dead_tup AS dead_rows,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS indexes_size,
        last_vacuum,
        last_autovacuum
    FROM pg_stat_user_tables
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
    """
    df = pd.read_sql(query, engine)
    return df


def check_project_tables(engine):
    """프로젝트 관련 테이블 (notice, company, pair)"""
    query = """
    SELECT
        schemaname AS schema_name,
        tablename AS table_name,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS indexes_size,
        pg_total_relation_size(schemaname||'.'||tablename) AS total_bytes
    FROM pg_tables
    WHERE tablename LIKE '%notice%'
       OR tablename LIKE '%company%'
       OR tablename LIKE '%pair%'
    ORDER BY total_bytes DESC;
    """
    df = pd.read_sql(query, engine)
    return df


def estimate_memory_usage(engine, schema='step1'):
    """GPU 메모리 사용량 추정"""
    print(f"\n{'='*80}")
    print("GPU 메모리 사용량 추정")
    print(f"{'='*80}\n")

    # Notice 테이블
    query_notice = f"""
    SELECT COUNT(*) as row_count FROM {schema}.notice;
    """
    notice_rows = pd.read_sql(query_notice, engine)['row_count'].iloc[0]

    # Company 테이블
    query_company = f"""
    SELECT COUNT(*) as row_count FROM {schema}.company;
    """
    company_rows = pd.read_sql(query_company, engine)['row_count'].iloc[0]

    # Pair 테이블
    query_pair = f"""
    SELECT COUNT(*) as row_count FROM {schema}.pair;
    """
    pair_rows = pd.read_sql(query_pair, engine)['row_count'].iloc[0]

    # 메모리 계산
    # Notice: 8 categorical (8 bytes each) + dense projection (256 * 2 bytes for FP16)
    notice_cat_memory = notice_rows * 8 * 8  # 8 columns * 8 bytes
    notice_dense_memory = notice_rows * 256 * 2  # 256 dims * 2 bytes (FP16)
    notice_total = notice_cat_memory + notice_dense_memory

    # Company: 8 categorical + dense projection (128 * 2 bytes for FP16)
    company_cat_memory = company_rows * 8 * 8
    company_dense_memory = company_rows * 128 * 2
    company_total = company_cat_memory + company_dense_memory

    # Pair: 3 IDs * 8 bytes
    pair_memory = pair_rows * 3 * 8

    total_memory = notice_total + company_total + pair_memory

    print(f"Notice 테이블:")
    print(f"  - 행 수: {notice_rows:,}")
    print(f"  - Categorical: {format_bytes(notice_cat_memory)}")
    print(f"  - Dense (FP16): {format_bytes(notice_dense_memory)}")
    print(f"  - 총: {format_bytes(notice_total)}")

    print(f"\nCompany 테이블:")
    print(f"  - 행 수: {company_rows:,}")
    print(f"  - Categorical: {format_bytes(company_cat_memory)}")
    print(f"  - Dense (FP16): {format_bytes(company_dense_memory)}")
    print(f"  - 총: {format_bytes(company_total)}")

    print(f"\nPair 테이블:")
    print(f"  - 행 수: {pair_rows:,}")
    print(f"  - 메모리: {format_bytes(pair_memory)}")

    print(f"\n{'='*80}")
    print(f"총 GPU 메모리 사용량 (추정): {format_bytes(total_memory)}")
    print(f"권장 GPU 메모리: {format_bytes(total_memory * 1.5)} (여유분 50%)")
    print(f"{'='*80}\n")

    # V2 적합성 판단
    if total_memory < 4 * 1024**3:  # 4GB
        print("✅ V2 파이프라인 적합: Feature가 GPU 메모리에 충분히 들어갑니다.")
        print("   → streaming 불필요, 전체 로드 권장")
    elif total_memory < 8 * 1024**3:  # 8GB
        print("⚠️  V2 파이프라인 사용 가능: 8GB+ GPU 필요")
        print("   → streaming 불필요, 전체 로드 가능")
    else:
        print("❌ V2 파이프라인 부적합: Feature가 GPU 메모리에 안 들어갑니다.")
        print("   → V1 파이프라인 (CPU collate) 사용 권장")
        print("   → 또는 batch size를 줄이고 일부 데이터만 사용")


def main():
    parser = argparse.ArgumentParser(description="데이터베이스 테이블 용량 조사")
    parser.add_argument("--detail", action="store_true", help="상세 정보 포함")
    parser.add_argument("--estimate-memory", action="store_true", help="GPU 메모리 사용량 추정")
    parser.add_argument("--schema", type=str, default="step1", help="스키마 이름")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("데이터베이스 테이블 용량 조사")
    print("="*80 + "\n")

    # DB 연결
    db = DatabaseConnector()
    engine = db.engine

    # 1. 전체 테이블 크기
    print("\n[1] 전체 테이블 크기 (크기 순)")
    print("-" * 80)
    df_sizes = check_table_sizes(engine)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_sizes.to_string(index=False))

    # 2. 프로젝트 관련 테이블
    print("\n\n[2] 프로젝트 관련 테이블 (notice, company, pair)")
    print("-" * 80)
    df_project = check_project_tables(engine)
    print(df_project.to_string(index=False))

    # 3. 상세 정보 (옵션)
    if args.detail:
        print("\n\n[3] 테이블 상세 정보 (행 수 포함)")
        print("-" * 80)
        df_details = check_table_details(engine)
        print(df_details.to_string(index=False))

    # 4. GPU 메모리 추정 (옵션)
    if args.estimate_memory:
        estimate_memory_usage(engine, schema=args.schema)

    print("\n" + "="*80)
    print("조사 완료")
    print("="*80 + "\n")

    # 요약 통계
    total_size = df_sizes['total_bytes'].sum()
    print(f"전체 테이블 총 용량: {format_bytes(total_size)}")
    print(f"테이블 수: {len(df_sizes)}")

    if not args.estimate_memory:
        print("\n💡 Tip: GPU 메모리 사용량을 확인하려면 --estimate-memory 옵션을 추가하세요.")
        print("   예: python scripts/check_db_sizes.py --estimate-memory")


if __name__ == "__main__":
    main()
