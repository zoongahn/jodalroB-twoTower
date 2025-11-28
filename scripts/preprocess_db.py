#!/usr/bin/env python3
"""
DB 전처리 스크립트

소스 DB에서 데이터를 읽어 전처리 후 타겟 DB에 저장

Usage:
    # 테스트 (10개 샘플)
    python scripts/preprocess_db.py --test

    # 전체 데이터 전처리 + DB 저장
    python scripts/preprocess_db.py --batch --save-to-db

    # 최근 1000개만 전처리
    python scripts/preprocess_db.py --batch --limit 1000

    # notice만 전처리
    python scripts/preprocess_db.py --batch --tables notice --save-to-db

    # 스키마 지정 (tmp.notice → tmp.notice_preprocessed)
    python scripts/preprocess_db.py --batch --source-schema tmp --target-schema tmp --save-to-db

    # 특정 공고들
    python scripts/preprocess_db.py --notice-ids "20240406546,000" "20240406548,000" --tables notice
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv

from preprocess.db.pipeline import get_multiple_notices_data
from preprocess.db.preprocess_notice import NoticePreprocessor
from preprocess.db.preprocess_company import CompanyPreprocessor
from preprocess.db.preprocess_bid import BidPreprocessor
from database.database_connector import DatabaseConnector
from database.query_helper import QueryHelper

load_dotenv()


def parse_notice_ids(notice_id_strs: List[str]) -> List[Tuple[str, str]]:
    """
    문자열 리스트를 (bidntceno, bidntceord) 튜플 리스트로 변환

    Examples:
        ["20240406546,000", "20240406548,000"] → [("20240406546", "000"), ...]
    """
    result = []
    for s in notice_id_strs:
        parts = s.split(',')
        if len(parts) != 2:
            raise ValueError(f"잘못된 형식: {s}. '공고번호,차수' 형식이어야 합니다.")
        result.append((parts[0].strip(), parts[1].strip()))
    return result


def load_column_types() -> dict:
    """
    메타데이터에서 컬럼 타입 정보 로드

    Returns:
        {
            'notice': {'pk': [...], 'numeric': [...], 'categorical': [...], 'text': [...]},
            'company': {'pk': [...], 'numeric': [...], 'categorical': [...], 'text': [...]}
        }
    """
    metadata_file = os.getenv("METADATA_FILE_PATH")
    use_keyword = os.getenv("METADATA_USE_KEYWORD")

    try:
        meta_df = pd.read_csv(metadata_file, dtype=str)
        use_df = meta_df[meta_df[use_keyword].str.upper() == 'Y']

        # 테이블별 PK 정의
        table_pk_map = {
            'notice': ['bidntceno', 'bidntceord'],
            'company': ['bizno']
        }

        column_types = {}

        for table_name in ['notice', 'company']:
            table_meta = use_df[use_df['테이블명'] == table_name]

            column_types[table_name] = {
                'pk': table_pk_map.get(table_name, []),
                'numeric': [],
                'categorical': [],
                'text': []
            }

            pk_columns = set(table_pk_map.get(table_name, []))

            for _, row in table_meta.iterrows():
                col_name = row['컬럼명']

                # PK 컬럼은 건너뛰기
                if col_name in pk_columns:
                    continue

                data_type = row['타입'].lower()
                categorical_flag = row.get('범주형 여부')

                # 수치형 분류
                if any(numeric_type in data_type for numeric_type in
                       ['integer', 'bigint', 'numeric', 'double precision', 'int', 'float']):
                    column_types[table_name]['numeric'].append(col_name)

                # 텍스트/범주형 분류
                elif any(text_type in data_type for text_type in
                         ['text', 'character', 'varchar', 'char']):
                    if categorical_flag == 'Y':
                        column_types[table_name]['categorical'].append(col_name)
                    else:
                        column_types[table_name]['text'].append(col_name)

        print("✅ 메타데이터 기반 컬럼 분류 완료:")
        for table in column_types:
            print(f"  📋 {table}: PK {len(column_types[table]['pk'])}개, "
                  f"수치형 {len(column_types[table]['numeric'])}개, "
                  f"범주형 {len(column_types[table]['categorical'])}개, "
                  f"텍스트 {len(column_types[table]['text'])}개")

        return column_types

    except Exception as e:
        print(f"⚠️ 메타데이터 로드 실패: {e}")
        # 기본값 설정
        return {
            'notice': {'pk': ['bidntceno', 'bidntceord'], 'numeric': [], 'categorical': [], 'text': []},
            'company': {'pk': ['bizno'], 'numeric': [], 'categorical': [], 'text': []}
        }


def test_preprocessing(tables=['notice', 'company', 'bid'], source_schema='public', target_schema='step1', save_to_db=False):
    """10개 공고로 전처리 테스트"""
    notice_ids = [
        ("20240406546", "000"),
        ("20240406548", "000"),
        ("20240406553", "000"),
        ("20240406556", "000"),
        ("20240406557", "000"),
        ("20140228597", "000"),
        ("20240406558", "000"),
        ("20140232077", "000"),
        ("20240406559", "000"),
        ("20240406563", "000")
    ]

    print(f"🧪 테스트 모드: 10개 공고로 전처리 실행")
    print(f"   소스: {source_schema}.{{table}}")
    print(f"   타겟: {target_schema}.{{table}}_preprocessed")
    print(f"   테이블: {', '.join(tables)}")
    print(f"   DB 저장: {'예' if save_to_db else '아니오 (CSV만)'}")
    print("=" * 80)

    # 1. 컬럼 타입 로드
    column_types = load_column_types()

    # 2. 데이터 조회 (bid는 별도 처리하므로 notice, company만)
    query_tables = [t for t in tables if t in ['notice', 'company']]
    data = get_multiple_notices_data(notice_ids, source_schema=source_schema, tables=query_tables)

    # 3. 전처리 실행
    os.makedirs("output/preprocessed", exist_ok=True)
    db_connector = DatabaseConnector() if save_to_db else None

    # Notice 전처리
    if 'notice' in tables:
        notice_preprocessor = NoticePreprocessor(column_types['notice'])
        preprocessed_notice = notice_preprocessor.preprocess(data['notice'].copy())

        notice_path = "output/preprocessed/test_notice_preprocessed.csv"
        preprocessed_notice.to_csv(notice_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 공고 전처리 완료: {notice_path} ({len(preprocessed_notice)}행, {len(preprocessed_notice.columns)}개 컬럼)")

        if save_to_db:
            notice_preprocessor.save_to_db(preprocessed_notice, target_schema, db_connector)

    # Company 전처리
    if 'company' in tables:
        company_preprocessor = CompanyPreprocessor(column_types['company'])
        preprocessed_company = company_preprocessor.preprocess(data['company'].copy())

        company_path = "output/preprocessed/test_company_preprocessed.csv"
        preprocessed_company.to_csv(company_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 업체 전처리 완료: {company_path} ({len(preprocessed_company)}행, {len(preprocessed_company.columns)}개 컬럼)")

        if save_to_db:
            company_preprocessor.save_to_db(preprocessed_company, target_schema, db_connector)

    # Bid 전처리 (공고-업체 페어 생성)
    if 'bid' in tables:
        bid_preprocessor = BidPreprocessor()
        preprocessed_bid = bid_preprocessor.preprocess(data['bid'].copy())

        bid_path = "output/preprocessed/test_bid_two_tower.csv"
        preprocessed_bid.to_csv(bid_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 페어 전처리 완료: {bid_path} ({len(preprocessed_bid)}행)")

        if save_to_db:
            bid_preprocessor.save_to_db(preprocessed_bid, target_schema, db_connector)

    if db_connector:
        db_connector.close()


def batch_preprocess(limit: int = None, tables=['notice', 'company', 'bid'], source_schema='public', target_schema='step1', save_to_db=False):
    """배치 전처리: 최근 N개 공고 전처리 (limit이 None이면 전체)"""
    db_connector = DatabaseConnector()
    helper = QueryHelper(db_connector, source_schema=source_schema)

    print(f"🔄 배치 전처리 모드: {'전체' if limit is None else f'최근 {limit}개'} 공고")
    print(f"   소스: {source_schema}.{{table}}")
    print(f"   타겟: {target_schema}.{{table}}_preprocessed")
    print(f"   테이블: {', '.join(tables)}")
    print(f"   DB 저장: {'예' if save_to_db else '아니오 (CSV만)'}")
    print("=" * 80)

    # 최근 공고 조회
    recent_notices = helper.get_recent_notices(limit=limit)

    if recent_notices.empty:
        print("❌ 조회된 공고가 없습니다.")
        db_connector.close()
        return

    # 공고 ID 추출
    pk_map = helper.get_table_pk_columns('notice')
    notice_ids = [(row[pk_map[0]], row[pk_map[1]]) for _, row in recent_notices.iterrows()]

    print(f"📋 {len(notice_ids)}개 공고 전처리 시작...")

    # 1. 컬럼 타입 로드
    column_types = load_column_types()

    # 2. 데이터 조회
    query_tables = [t for t in tables if t in ['notice', 'company']]
    data = get_multiple_notices_data(notice_ids, source_schema=source_schema, tables=query_tables)

    # 3. 전처리 실행
    os.makedirs("output/preprocessed", exist_ok=True)

    # Notice 전처리
    if 'notice' in tables:
        notice_preprocessor = NoticePreprocessor(column_types['notice'])
        preprocessed_notice = notice_preprocessor.preprocess(data['notice'].copy())

        notice_path = "output/preprocessed/batch_notice_preprocessed.csv"
        preprocessed_notice.to_csv(notice_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 공고 전처리 완료: {notice_path} ({len(preprocessed_notice)}행)")

        if save_to_db:
            notice_preprocessor.save_to_db(preprocessed_notice, target_schema, db_connector)

    # Company 전처리
    if 'company' in tables:
        company_preprocessor = CompanyPreprocessor(column_types['company'])
        preprocessed_company = company_preprocessor.preprocess(data['company'].copy())

        company_path = "output/preprocessed/batch_company_preprocessed.csv"
        preprocessed_company.to_csv(company_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 업체 전처리 완료: {company_path} ({len(preprocessed_company)}행)")

        if save_to_db:
            company_preprocessor.save_to_db(preprocessed_company, target_schema, db_connector)

    # Bid 전처리
    if 'bid' in tables:
        bid_preprocessor = BidPreprocessor()
        preprocessed_bid = bid_preprocessor.preprocess(data['bid'].copy())

        bid_path = "output/preprocessed/batch_bid_two_tower.csv"
        preprocessed_bid.to_csv(bid_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 페어 전처리 완료: {bid_path} ({len(preprocessed_bid)}행)")

        if save_to_db:
            bid_preprocessor.save_to_db(preprocessed_bid, target_schema, db_connector)

    db_connector.close()


def batch_preprocess_all_companies(source_schema='public', target_schema='step1', save_to_db=False, limit=None):
    """업체 테이블 전체 전처리 (공고와 무관하게 독립적으로)"""
    db_connector = DatabaseConnector()

    print(f"🔄 업체 전체 전처리 모드")
    print(f"   소스: {source_schema}.company")
    print(f"   타겟: {target_schema}.company_preprocessed")
    print(f"   제한: {'전체' if limit is None else f'{limit}개'}")
    print(f"   DB 저장: {'예' if save_to_db else '아니오 (CSV만)'}")
    print("=" * 80)

    # 1. company 테이블에서 직접 데이터 조회
    query = f"SELECT * FROM {source_schema}.company"
    if limit:
        query += f" LIMIT {limit}"

    print(f"📋 업체 데이터 조회 중...")
    company_data = pd.read_sql(query, db_connector.engine)
    print(f"✅ {len(company_data)}개 업체 데이터 조회 완료")

    if company_data.empty:
        print("❌ 조회된 업체가 없습니다.")
        db_connector.close()
        return

    # 2. 컬럼 타입 로드
    column_types = load_column_types()

    # 3. 전처리 실행
    preprocessor = CompanyPreprocessor(column_types['company'])
    preprocessed_company = preprocessor.preprocess(company_data.copy())

    # 4. 결과 저장
    os.makedirs("output/preprocessed", exist_ok=True)
    company_path = "output/preprocessed/all_company_preprocessed.csv"
    preprocessed_company.to_csv(company_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 업체 전처리 완료: {company_path} ({len(preprocessed_company)}행)")

    if save_to_db:
        preprocessor.save_to_db(preprocessed_company, target_schema, db_connector)

    db_connector.close()


def batch_preprocess_all_bids(source_schema='public', target_schema='step1', save_to_db=False, limit=None):
    """bid 테이블 전체를 bid_two_tower 형태로 변환"""
    from preprocess.db.preprocess_bid import BidPreprocessor

    db_connector = DatabaseConnector()

    print(f"🔄 Bid 전체 전처리 모드 (bid → bid_two_tower)")
    print(f"   소스: {source_schema}.bid")
    print(f"   타겟: {target_schema}.bid_two_tower")
    print(f"   제한: {'전체' if limit is None else f'{limit}개'}")
    print(f"   DB 저장: {'예' if save_to_db else '아니오 (CSV만)'}")
    print("=" * 80)

    # 1. bid 테이블에서 직접 데이터 조회
    query = f"SELECT * FROM {source_schema}.bid"
    if limit:
        query += f" LIMIT {limit}"

    print(f"📋 Bid 데이터 조회 중...")
    bid_data = pd.read_sql(query, db_connector.engine)
    print(f"✅ {len(bid_data)}개 Bid 데이터 조회 완료")

    if bid_data.empty:
        print("❌ 조회된 Bid가 없습니다.")
        db_connector.close()
        return

    # 2. 전처리 실행 (bid → bid_two_tower 변환)
    bid_preprocessor = BidPreprocessor()
    preprocessed_bid = bid_preprocessor.preprocess(bid_data.copy())

    # 3. 결과 저장
    os.makedirs("output/preprocessed", exist_ok=True)
    bid_path = "output/preprocessed/all_bid_two_tower.csv"
    preprocessed_bid.to_csv(bid_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Bid 전처리 완료: {bid_path} ({len(preprocessed_bid)}행)")

    if save_to_db:
        bid_preprocessor.save_to_db(preprocessed_bid, target_schema, db_connector)

    db_connector.close()


def preprocess_specific_notices(notice_ids: List[Tuple[str, str]], tables=['notice', 'company', 'bid'], source_schema='public', target_schema='step1', save_to_db=False):
    """특정 공고들 전처리"""
    print(f"🎯 특정 공고 전처리: {len(notice_ids)}개")
    print(f"   소스: {source_schema}.{{table}}")
    print(f"   타겟: {target_schema}.{{table}}_preprocessed")
    print(f"   테이블: {', '.join(tables)}")
    print(f"   DB 저장: {'예' if save_to_db else '아니오 (CSV만)'}")
    print("=" * 80)

    for notice_id in notice_ids:
        print(f"  - {notice_id}")

    # 1. 컬럼 타입 로드
    column_types = load_column_types()

    # 2. 데이터 조회
    query_tables = [t for t in tables if t in ['notice', 'company']]
    data = get_multiple_notices_data(notice_ids, source_schema=source_schema, tables=query_tables)

    # 3. 전처리 실행
    os.makedirs("output/preprocessed", exist_ok=True)
    db_connector = DatabaseConnector() if save_to_db else None

    # Notice 전처리
    if 'notice' in tables:
        notice_preprocessor = NoticePreprocessor(column_types['notice'])
        preprocessed_notice = notice_preprocessor.preprocess(data['notice'].copy())

        notice_path = "output/preprocessed/specific_notice_preprocessed.csv"
        preprocessed_notice.to_csv(notice_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 공고 전처리 완료: {notice_path} ({len(preprocessed_notice)}행)")

        if save_to_db:
            notice_preprocessor.save_to_db(preprocessed_notice, target_schema, db_connector)

    # Company 전처리
    if 'company' in tables:
        company_preprocessor = CompanyPreprocessor(column_types['company'])
        preprocessed_company = company_preprocessor.preprocess(data['company'].copy())

        company_path = "output/preprocessed/specific_company_preprocessed.csv"
        preprocessed_company.to_csv(company_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 업체 전처리 완료: {company_path} ({len(preprocessed_company)}행)")

        if save_to_db:
            company_preprocessor.save_to_db(preprocessed_company, target_schema, db_connector)

    # Bid 전처리
    if 'bid' in tables:
        bid_preprocessor = BidPreprocessor()
        preprocessed_bid = bid_preprocessor.preprocess(data['bid'].copy())

        bid_path = "output/preprocessed/specific_bid_two_tower.csv"
        preprocessed_bid.to_csv(bid_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 페어 전처리 완료: {bid_path} ({len(preprocessed_bid)}행)")

        if save_to_db:
            bid_preprocessor.save_to_db(preprocessed_bid, target_schema, db_connector)

    if db_connector:
        db_connector.close()


def main():
    parser = argparse.ArgumentParser(
        description='DB 전처리: 소스 DB → 타겟 DB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 테스트 모드 (10개 샘플)
  python scripts/preprocess_db.py --test

  # 전체 데이터 전처리 + DB 저장
  python scripts/preprocess_db.py --batch --save-to-db

  # 최근 1000개만 전처리
  python scripts/preprocess_db.py --batch --limit 1000 --save-to-db

  # notice만 전처리 (tmp 스키마)
  python scripts/preprocess_db.py --batch --tables notice --source-schema tmp --target-schema tmp --save-to-db
        """
    )

    # 실행 모드
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', action='store_true', help='테스트 모드 (10개 샘플)')
    group.add_argument('--batch', action='store_true', help='배치 모드 (최근 N개 공고)')
    group.add_argument('--notice-ids', nargs='+', help='특정 공고 ID들 (예: "20240406546,000" "20240406548,000")')
    group.add_argument('--all-companies', action='store_true', help='업체 테이블 전체 전처리 (공고와 무관)')
    group.add_argument('--all-bids', action='store_true', help='bid 테이블 전체를 bid_two_tower로 변환')

    # 데이터 옵션
    parser.add_argument('--tables', nargs='+', choices=['notice', 'company', 'bid'], default=['notice', 'company', 'bid'],
                        help='전처리할 테이블 선택 (기본: notice company bid 전부)')
    parser.add_argument('--limit', type=int, default=None,
                        help='배치 모드에서 처리할 최대 공고 수 (기본: None = 전체)')

    # DB 옵션
    parser.add_argument('--source-schema', type=str, default='public',
                        help='소스 스키마 (데이터를 읽어올 스키마, 기본: public)')
    parser.add_argument('--target-schema', type=str, default='step1',
                        help='타겟 스키마 (전처리 결과를 저장할 스키마, 기본: step1)')
    parser.add_argument('--save-to-db', action='store_true',
                        help='DB에 저장 (기본: CSV만 저장)')

    args = parser.parse_args()

    try:
        if args.test:
            test_preprocessing(
                tables=args.tables,
                source_schema=args.source_schema,
                target_schema=args.target_schema,
                save_to_db=args.save_to_db
            )
        elif args.batch:
            batch_preprocess(
                limit=args.limit,
                tables=args.tables,
                source_schema=args.source_schema,
                target_schema=args.target_schema,
                save_to_db=args.save_to_db
            )
        elif args.notice_ids:
            notice_ids = parse_notice_ids(args.notice_ids)
            preprocess_specific_notices(
                notice_ids,
                tables=args.tables,
                source_schema=args.source_schema,
                target_schema=args.target_schema,
                save_to_db=args.save_to_db
            )
        elif args.all_companies:
            batch_preprocess_all_companies(
                source_schema=args.source_schema,
                target_schema=args.target_schema,
                save_to_db=args.save_to_db,
                limit=args.limit
            )
        elif args.all_bids:
            batch_preprocess_all_bids(
                source_schema=args.source_schema,
                target_schema=args.target_schema,
                save_to_db=args.save_to_db,
                limit=args.limit
            )

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
