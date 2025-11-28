#!/usr/bin/env python3
"""
Parquet 파일 유효성 검증 스크립트

사용법:
    python scripts/validate_parquet.py --parquet_dir data/parquet
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def validate_parquet_files(parquet_dir: str):
    """Parquet 파일들이 올바르게 생성되었는지 검증"""
    parquet_path = Path(parquet_dir)

    print("=" * 80)
    print("Parquet 파일 유효성 검증")
    print("=" * 80)
    print(f"디렉토리: {parquet_path}\n")

    # 필수 파일 확인
    required_files = ["notice.parquet", "company.parquet", "pairs.parquet"]
    missing_files = []

    for filename in required_files:
        filepath = parquet_path / filename
        if not filepath.exists():
            missing_files.append(filename)
            print(f"❌ {filename}: 파일이 없습니다")
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename}: {size_mb:.2f} MB")

    if missing_files:
        print(f"\n⚠️ 누락된 파일: {', '.join(missing_files)}")
        print("먼저 convert_to_parquet.py를 실행하세요.")
        return False

    print("\n" + "=" * 80)
    print("데이터 구조 검증")
    print("=" * 80)

    all_valid = True

    # Notice 검증
    print("\n1. Notice 데이터")
    try:
        notice_df = pd.read_parquet(parquet_path / "notice.parquet")
        print(f"   행 수: {len(notice_df):,}")
        print(f"   컬럼 수: {len(notice_df.columns)}")

        # PK 확인
        required_pk = ["bidntceno", "bidntceord"]
        for pk in required_pk:
            if pk in notice_df.columns:
                print(f"   ✅ PK '{pk}' 존재")
            else:
                print(f"   ❌ PK '{pk}' 누락")
                all_valid = False

        # 중복 확인
        duplicates = notice_df.duplicated(subset=required_pk).sum()
        if duplicates == 0:
            print(f"   ✅ PK 중복 없음")
        else:
            print(f"   ⚠️ PK 중복: {duplicates:,}개")
            all_valid = False

        # NULL 확인
        null_counts = notice_df[required_pk].isnull().sum()
        if null_counts.sum() == 0:
            print(f"   ✅ PK NULL 없음")
        else:
            print(f"   ⚠️ PK NULL: {null_counts.to_dict()}")
            all_valid = False

        print(f"\n   샘플 데이터:")
        print(notice_df.head(3).to_string(index=False))

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        all_valid = False

    # Company 검증
    print("\n2. Company 데이터")
    try:
        company_df = pd.read_parquet(parquet_path / "company.parquet")
        print(f"   행 수: {len(company_df):,}")
        print(f"   컬럼 수: {len(company_df.columns)}")

        # PK 확인
        required_pk = ["bizno"]
        for pk in required_pk:
            if pk in company_df.columns:
                print(f"   ✅ PK '{pk}' 존재")
            else:
                print(f"   ❌ PK '{pk}' 누락")
                all_valid = False

        # 중복 확인
        duplicates = company_df.duplicated(subset=required_pk).sum()
        if duplicates == 0:
            print(f"   ✅ PK 중복 없음")
        else:
            print(f"   ⚠️ PK 중복: {duplicates:,}개")
            all_valid = False

        # NULL 확인
        null_counts = company_df[required_pk].isnull().sum()
        if null_counts.sum() == 0:
            print(f"   ✅ PK NULL 없음")
        else:
            print(f"   ⚠️ PK NULL: {null_counts.to_dict()}")
            all_valid = False

        print(f"\n   샘플 데이터:")
        print(company_df.head(3).to_string(index=False))

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        all_valid = False

    # Pairs 검증
    print("\n3. Pairs 데이터")
    try:
        pairs_df = pd.read_parquet(parquet_path / "pairs.parquet")
        print(f"   행 수: {len(pairs_df):,}")
        print(f"   컬럼 수: {len(pairs_df.columns)}")

        # FK 확인
        required_fk = ["bidntceno", "bidntceord", "bizno"]
        for fk in required_fk:
            if fk in pairs_df.columns:
                print(f"   ✅ FK '{fk}' 존재")
            else:
                print(f"   ❌ FK '{fk}' 누락")
                all_valid = False

        # NULL 확인
        null_counts = pairs_df[required_fk].isnull().sum()
        if null_counts.sum() == 0:
            print(f"   ✅ FK NULL 없음")
        else:
            print(f"   ⚠️ FK NULL: {null_counts.to_dict()}")
            all_valid = False

        print(f"\n   샘플 데이터:")
        print(pairs_df.head(3).to_string(index=False))

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        all_valid = False

    # 참조 무결성 검증
    print("\n" + "=" * 80)
    print("참조 무결성 검증")
    print("=" * 80)

    try:
        # Notice FK 검증
        notice_keys = set(zip(notice_df['bidntceno'], notice_df['bidntceord']))
        pair_notice_keys = set(zip(pairs_df['bidntceno'], pairs_df['bidntceord']))

        missing_notices = pair_notice_keys - notice_keys
        if len(missing_notices) == 0:
            print(f"✅ Pairs → Notice FK: 모두 유효 ({len(pair_notice_keys):,}개)")
        else:
            print(f"⚠️ Pairs → Notice FK: {len(missing_notices):,}개 누락")
            print(f"   샘플: {list(missing_notices)[:5]}")
            all_valid = False

        # Company FK 검증
        company_keys = set(company_df['bizno'].astype(str))
        pair_company_keys = set(pairs_df['bizno'].astype(str))

        missing_companies = pair_company_keys - company_keys
        if len(missing_companies) == 0:
            print(f"✅ Pairs → Company FK: 모두 유효 ({len(pair_company_keys):,}개)")
        else:
            print(f"⚠️ Pairs → Company FK: {len(missing_companies):,}개 누락")
            print(f"   샘플: {list(missing_companies)[:5]}")
            all_valid = False

    except Exception as e:
        print(f"❌ 참조 무결성 검증 오류: {e}")
        all_valid = False

    # 최종 결과
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ 모든 검증 통과!")
        print("=" * 80)
        print("\n학습 시 사용법:")
        print(f"PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \\")
        print(f"  --use_parquet \\")
        print(f"  --parquet_dir {parquet_dir} \\")
        print(f"  --batch_size 256 \\")
        print(f"  --num_epochs 3")
        return True
    else:
        print("⚠️ 일부 검증 실패")
        print("=" * 80)
        print("\nParquet 파일을 다시 생성하려면:")
        print(f"python preprocess/convert_to_parquet.py --output {parquet_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Parquet 파일 유효성 검증")
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default="data/parquet",
        help="Parquet 파일 디렉토리 (기본값: data/parquet)"
    )
    args = parser.parse_args()

    success = validate_parquet_files(args.parquet_dir)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
