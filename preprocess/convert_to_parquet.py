#!/usr/bin/env python3
"""
DB 테이블을 Parquet 파일로 변환하는 스크립트

사용법:
    python preprocess/convert_to_parquet.py --output data/parquet

특징:
- step1.notice, bid_two_tower, company 테이블을 Parquet으로 변환
- 전처리된 테이블 (*_preprocessed) 사용
- 청크 단위로 읽어서 메모리 효율적으로 변환
- 텍스트 임베딩도 포함하여 저장
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import text
from tqdm import tqdm

from database.database_connector import DatabaseConnector
from preprocess.torchrec.schema import build_torchrec_schema_from_meta


def convert_table_to_parquet(
    engine,
    table_name: str,
    pk_cols: list,
    numeric_cols: list,
    categorical_cols: list,
    text_cols: list,
    output_path: Path,
    chunksize: int = 500_000,
    limit: Optional[int] = None,
    use_preprocessed: bool = True,
):
    """
    DB 테이블을 Parquet 파일로 변환

    Args:
        engine: SQLAlchemy engine
        table_name: 테이블명 (schema.table 형식)
        pk_cols: Primary key 컬럼 리스트
        numeric_cols: 수치형 컬럼 리스트
        categorical_cols: 범주형 컬럼 리스트
        text_cols: 텍스트 임베딩 컬럼 리스트
        output_path: 출력 Parquet 파일 경로
        chunksize: 청크 크기
        limit: 최대 행 수 (None이면 전체)
        use_preprocessed: preprocessed 테이블 사용 여부
    """
    # 테이블명 처리: _preprocessed 추가 (옵션)
    if not use_preprocessed:
        full_table = table_name
    elif table_name.endswith('_preprocessed'):
        full_table = table_name
    else:
        table_parts = table_name.split('.')
        if len(table_parts) == 2:
            schema_name, table_name_only = table_parts
            full_table = f"{schema_name}.{table_name_only}_preprocessed"
        else:
            full_table = f"{table_name}_preprocessed"

    # 총 행 수 계산
    count_sql = f"SELECT COUNT(*) FROM {full_table}"
    with engine.connect() as cx:
        total_rows = cx.execute(text(count_sql)).scalar()

    if limit:
        total_rows = min(total_rows, limit)

    print(f"\n📊 {full_table}")
    print(f"   총 {total_rows:,}개 행 변환 중...")

    # SELECT 쿼리 구성
    cols = pk_cols + numeric_cols + categorical_cols

    # 텍스트 임베딩 컬럼은 float4[]로 캐스팅
    if text_cols:
        for col in text_cols:
            cols.append(f"{col}::float4[] AS {col}")

    select_cols = ", ".join(cols)
    sql = f"SELECT {select_cols} FROM {full_table}"
    if limit:
        sql += f" LIMIT {limit}"

    # Parquet writer 준비
    writer = None
    schema = None

    with engine.connect() as cx:
        # PostgreSQL 세션 최적화
        cx.execute(text("SET work_mem = '256MB'"))
        cx.execute(text("SET max_parallel_workers_per_gather = 4"))

        rs = cx.execution_options(stream_results=True).exec_driver_sql(sql)

        with tqdm(total=total_rows, unit="rows", desc=f"  {table_name.split('.')[-1]}") as pbar:
            chunk_num = 0
            while True:
                # 청크 읽기
                rows = rs.fetchmany(chunksize)
                if not rows:
                    break

                df = pd.DataFrame(rows, columns=rs.keys())

                # 데이터 타입 변환
                if numeric_cols:
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = df[col].astype('float32')

                if categorical_cols:
                    for col in categorical_cols:
                        if col in df.columns:
                            df[col] = df[col].astype('int64')

                # 텍스트 임베딩은 numpy array로 유지 (메모리 효율적)
                if text_cols:
                    for col in text_cols:
                        if col in df.columns:
                            # None 처리: 메모리 효율적으로
                            df[col] = df[col].apply(
                                lambda x: np.array(x, dtype=np.float32) if x is not None else None
                            )

                # PyArrow Table로 변환
                table = pa.Table.from_pandas(df, preserve_index=False)

                # 첫 번째 청크에서 writer 초기화
                if writer is None:
                    schema = table.schema
                    # 메모리 버퍼 제한 설정
                    writer = pq.ParquetWriter(
                        output_path,
                        schema,
                        compression='snappy',
                        write_batch_size=10000,  # 배치 크기 제한
                        use_dictionary=True,
                    )

                # 쓰기
                writer.write_table(table)

                pbar.update(len(df))
                chunk_num += 1

                # 메모리 해제 (중요!)
                del df, table
                if chunk_num % 10 == 0:
                    import gc
                    gc.collect()

    # Writer 종료
    if writer:
        writer.close()
        print(f"   ✅ 저장 완료: {output_path} ({chunk_num} chunks)")
    else:
        print(f"   ⚠️ 데이터가 없습니다: {full_table}")


def main():
    parser = argparse.ArgumentParser(description="DB 테이블을 Parquet 파일로 변환")
    parser.add_argument(
        "--output",
        type=str,
        default="data/parquet",
        help="Parquet 파일 저장 디렉토리 (기본값: data/parquet)"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="청크 크기 (기본값: 500,000)"
    )
    parser.add_argument(
        "--limit",
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=None,
        help="최대 행 수 (기본값: None = 전체)"
    )
    parser.add_argument(
        "--tables",
        type=str,
        nargs='+',
        choices=['notice', 'company', 'pairs', 'all'],
        default=['all'],
        help="변환할 테이블 선택 (기본값: all, 예: --tables notice pairs)"
    )
    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 테이블 선택 처리
    if 'all' in args.tables:
        tables_to_convert = ['notice', 'company', 'pairs']
    else:
        tables_to_convert = args.tables

    print("=" * 80)
    print("DB 테이블 → Parquet 변환")
    print("=" * 80)
    print(f"출력 디렉토리: {output_dir}")
    print(f"청크 크기: {args.chunksize:,}")
    if args.limit:
        print(f"최대 행 수: {args.limit:,}")
    else:
        print(f"최대 행 수: 전체")
    print(f"변환할 테이블: {', '.join(tables_to_convert)}")

    # DB 연결
    print("\n데이터베이스 연결 중...")
    db = DatabaseConnector()

    # 스키마 로드
    schema = build_torchrec_schema_from_meta(
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv"
    )

    start_time = time.time()

    task_num = 0
    total_tasks = len(tables_to_convert)

    # 1. Notice 테이블 변환
    if 'notice' in tables_to_convert:
        task_num += 1
        print(f"\n[{task_num}/{total_tasks}] Notice 테이블 변환 중...")
        convert_table_to_parquet(
            engine=db.engine,
            table_name=schema.notice.table,
            pk_cols=schema.notice.pk_cols,
            numeric_cols=schema.notice.numeric,
            categorical_cols=schema.notice.categorical,
            text_cols=schema.notice.text,
            output_path=output_dir / "notice.parquet",
            chunksize=args.chunksize,
            limit=args.limit,
        )

    # 2. Company 테이블 변환
    if 'company' in tables_to_convert:
        task_num += 1
        print(f"\n[{task_num}/{total_tasks}] Company 테이블 변환 중...")
        convert_table_to_parquet(
            engine=db.engine,
            table_name=schema.company.table,
            pk_cols=schema.company.pk_cols,
            numeric_cols=schema.company.numeric,
            categorical_cols=schema.company.categorical,
            text_cols=schema.company.text,
            output_path=output_dir / "company.parquet",
            chunksize=args.chunksize,
            limit=args.limit,
        )

    # 3. Pair 테이블 변환
    if 'pairs' in tables_to_convert:
        task_num += 1
        print(f"\n[{task_num}/{total_tasks}] Pair 테이블 변환 중...")
        # Pair 테이블은 PK + FK만 있음 (preprocessed 버전 없음)
        pair_cols = schema.pair.notice_id_cols + schema.pair.company_id_cols
        convert_table_to_parquet(
            engine=db.engine,
            table_name=schema.pair.table,
            pk_cols=pair_cols,  # pair 테이블은 모든 컬럼이 FK
            numeric_cols=[],
            categorical_cols=[],
            text_cols=[],
            output_path=output_dir / "pairs.parquet",
            chunksize=args.chunksize,
            limit=args.limit,
            use_preprocessed=False,  # Pair 테이블은 원본 사용
        )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ 변환 완료! (소요 시간: {elapsed:.2f}초)")
    print("=" * 80)
    print(f"\n{output_dir} 디렉토리의 모든 Parquet 파일:")
    for file in sorted(output_dir.glob("*.parquet")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
