#!/usr/bin/env python3
"""
업종 필터 가정 검증 스크립트

가정: "공고의 종목코드에 해당하는 종목을 가진 업체만 참여할 수 있다"

검증 로직:
1. public.bid에서 실제 공고-업체 참여 정보 조회
2. 각 공고의 종목코드 조회 (public.notice_industry_type)
3. 각 업체의 종목코드 조회 (public.company_industry_type)
4. 공고 종목코드와 업체 종목코드가 교집합이 있는지 확인

출력:
- 전체 검증 건수
- 일치/불일치 건수 및 비율
- 불일치 샘플 출력

Usage:
    python scripts/validate_industry_filter.py [--limit N] [--sample N]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm
from database.database_connector import DatabaseConnector


def parse_args():
    parser = argparse.ArgumentParser(
        description="업종 필터 가정 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="검증할 최대 공고-업체 쌍 수 (기본: 전체)"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="불일치 샘플 출력 개수 (기본: 20)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="DB 쿼리 청크 크기 (기본: 1000)"
    )

    return parser.parse_args()


def load_bid_data(engine, limit=None, chunk_size=1000):
    """
    실제 참여 데이터 로드 (공고-업체 쌍)
    """
    print("\n1. 실제 참여 데이터 로드 중...")

    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT DISTINCT bidntceno, bidntceord, bidprccorpbizrno as bizno, dqlfctnrsn, bidprcdate
    FROM public.bid
    WHERE bidprccorpbizrno IS NOT NULL
      AND bidprcdate IS NOT NULL
    ORDER BY bidprcdate DESC
    {limit_clause}
    """

    df = pd.read_sql(query, engine)
    print(f"   ✓ 로드 완료: {len(df):,}개 공고-업체 쌍 (입찰일 최근순)")

    return df


def load_notice_industry_codes(engine, notice_ids, chunk_size=500):
    """
    공고별 종목코드 로드

    Returns:
        {(bidntceno, bidntceord): set(종목코드)} 딕셔너리
    """
    print("\n2. 공고별 종목코드 로드 중...")

    notice_to_codes = {}
    unique_notices = list(set(notice_ids))
    num_chunks = (len(unique_notices) + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(num_chunks), desc="   공고 종목코드"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(unique_notices))
        chunk = unique_notices[start_idx:end_idx]

        values = ", ".join([
            f"('{ntce_no}', '{ntce_ord}')"
            for ntce_no, ntce_ord in chunk
        ])

        query = f"""
        SELECT bidntceno, bidntceord, lcnslmtnm_code
        FROM public.notice_industry_type
        WHERE (bidntceno, bidntceord) IN ({values})
        """

        chunk_df = pd.read_sql(query, engine)

        for _, row in chunk_df.iterrows():
            key = (row['bidntceno'], row['bidntceord'])
            if key not in notice_to_codes:
                notice_to_codes[key] = set()
            notice_to_codes[key].add(row['lcnslmtnm_code'])

    notices_with_codes = len(notice_to_codes)
    notices_without_codes = len(unique_notices) - notices_with_codes
    print(f"   ✓ 종목코드 있음: {notices_with_codes:,}개 공고")
    print(f"   ✓ 종목코드 없음: {notices_without_codes:,}개 공고")

    return notice_to_codes


def load_company_industry_codes(engine, company_ids, chunk_size=500):
    """
    업체별 종목코드 로드

    Returns:
        {bizno: set(종목코드)} 딕셔너리
    """
    print("\n3. 업체별 종목코드 로드 중...")

    company_to_codes = {}
    unique_companies = list(set(company_ids))
    num_chunks = (len(unique_companies) + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(num_chunks), desc="   업체 종목코드"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(unique_companies))
        chunk = unique_companies[start_idx:end_idx]

        values = "','".join(str(c) for c in chunk)

        query = f"""
        SELECT bizno, indstrytycd
        FROM public.company_industry_type
        WHERE bizno IN ('{values}')
        """

        chunk_df = pd.read_sql(query, engine)

        for _, row in chunk_df.iterrows():
            bizno = row['bizno']
            if bizno not in company_to_codes:
                company_to_codes[bizno] = set()
            company_to_codes[bizno].add(row['indstrytycd'])

    companies_with_codes = len(company_to_codes)
    companies_without_codes = len(unique_companies) - companies_with_codes
    print(f"   ✓ 종목코드 있음: {companies_with_codes:,}개 업체")
    print(f"   ✓ 종목코드 없음: {companies_without_codes:,}개 업체")

    return company_to_codes


def validate_industry_match(bid_df, notice_to_codes, company_to_codes):
    """
    공고-업체 쌍에 대해 종목코드 일치 여부 검증

    Returns:
        results: List[Dict] - 검증 결과
    """
    print("\n4. 종목코드 일치 여부 검증 중...")

    results = []

    for _, row in tqdm(bid_df.iterrows(), total=len(bid_df), desc="   검증"):
        notice_id = (row['bidntceno'], row['bidntceord'])
        bizno = row['bizno']
        dqlfctnrsn = row.get('dqlfctnrsn')

        notice_codes = notice_to_codes.get(notice_id)
        company_codes = company_to_codes.get(bizno)

        # 종목코드 없는 경우 처리
        if notice_codes is None:
            status = "notice_no_code"
            match = None
            intersection = None
        elif company_codes is None:
            status = "company_no_code"
            match = None
            intersection = None
        else:
            intersection = notice_codes & company_codes
            if len(intersection) > 0:
                status = "match"
                match = True
            else:
                status = "mismatch"
                match = False

        results.append({
            'bidntceno': row['bidntceno'],
            'bidntceord': row['bidntceord'],
            'bizno': bizno,
            'dqlfctnrsn': dqlfctnrsn,
            'notice_codes': notice_codes,
            'company_codes': company_codes,
            'intersection': intersection,
            'status': status,
            'match': match
        })

    return results


def print_summary(results, sample_count=20):
    """
    검증 결과 요약 출력
    """
    print("\n" + "=" * 80)
    print("검증 결과 요약")
    print("=" * 80)

    total = len(results)

    # 상태별 집계
    status_counts = {}
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\n전체 검증 건수: {total:,}개")
    print("\n상태별 분포:")

    for status, count in sorted(status_counts.items()):
        pct = count / total * 100
        status_desc = {
            'match': '✓ 일치 (공고-업체 종목 교집합 있음)',
            'mismatch': '✗ 불일치 (공고-업체 종목 교집합 없음)',
            'notice_no_code': '- 공고에 종목코드 없음',
            'company_no_code': '- 업체에 종목코드 없음'
        }.get(status, status)
        print(f"  {status_desc}: {count:,}개 ({pct:.2f}%)")

    # 검증 가능한 케이스만 따로 계산
    verifiable = [r for r in results if r['match'] is not None]
    if verifiable:
        match_count = sum(1 for r in verifiable if r['match'])
        mismatch_count = len(verifiable) - match_count
        match_rate = match_count / len(verifiable) * 100

        print(f"\n검증 가능한 케이스 (공고/업체 모두 종목코드 있음): {len(verifiable):,}개")
        print(f"  - 일치: {match_count:,}개 ({match_rate:.2f}%)")
        print(f"  - 불일치: {mismatch_count:,}개 ({100-match_rate:.2f}%)")

    # 불일치 케이스의 dqlfctnrsn 분포 분석
    mismatches = [r for r in results if r['status'] == 'mismatch']
    if mismatches:
        print(f"\n" + "-" * 80)
        print(f"불일치 케이스의 dqlfctnrsn(실격사유) 분포 ({len(mismatches):,}건)")
        print("-" * 80)

        # dqlfctnrsn 값별 집계
        dqlfctnrsn_counts = {}
        for r in mismatches:
            reason = r['dqlfctnrsn']
            if reason is None or (isinstance(reason, float) and pd.isna(reason)):
                reason = "(NULL/없음)"
            elif isinstance(reason, str) and reason.strip() == "":
                reason = "(빈 문자열)"
            dqlfctnrsn_counts[reason] = dqlfctnrsn_counts.get(reason, 0) + 1

        # 빈도순 정렬
        sorted_reasons = sorted(dqlfctnrsn_counts.items(), key=lambda x: -x[1])

        print(f"\n{'순위':<6}{'건수':>10}{'비율':>10}  {'실격사유'}")
        print("-" * 80)

        for rank, (reason, count) in enumerate(sorted_reasons[:30], 1):
            pct = count / len(mismatches) * 100
            # 긴 사유는 잘라서 표시
            reason_display = str(reason)[:50] + "..." if len(str(reason)) > 50 else str(reason)
            print(f"{rank:<6}{count:>10,}{pct:>9.2f}%  {reason_display}")

        if len(sorted_reasons) > 30:
            print(f"  ... 외 {len(sorted_reasons) - 30}개 사유")

    # 불일치 샘플 출력
    if mismatches and sample_count > 0:
        print(f"\n" + "-" * 80)
        print(f"불일치 샘플 (상위 {min(sample_count, len(mismatches))}개)")
        print("-" * 80)

        for i, r in enumerate(mismatches[:sample_count]):
            print(f"\n[{i+1}] 공고: {r['bidntceno']}-{r['bidntceord']}, 업체: {r['bizno']}")
            print(f"    공고 종목코드: {sorted(r['notice_codes']) if r['notice_codes'] else 'None'}")
            print(f"    업체 종목코드: {sorted(r['company_codes']) if r['company_codes'] else 'None'}")
            dqlfctnrsn_display = r['dqlfctnrsn'] if r['dqlfctnrsn'] else "(없음)"
            print(f"    실격사유(dqlfctnrsn): {dqlfctnrsn_display}")

    return status_counts


def main():
    args = parse_args()

    print("=" * 80)
    print("업종 필터 가정 검증")
    print("=" * 80)
    print(f"가정: 공고의 종목코드에 해당하는 종목을 가진 업체만 참여할 수 있다")

    # DB 연결
    db = DatabaseConnector()
    engine = db.engine
    print("✓ DB 연결 완료")

    try:
        # 1. 실제 참여 데이터 로드
        bid_df = load_bid_data(engine, limit=args.limit, chunk_size=args.chunk_size)

        # 2. 공고별 종목코드 로드
        notice_ids = [(row['bidntceno'], row['bidntceord']) for _, row in bid_df.iterrows()]
        notice_to_codes = load_notice_industry_codes(engine, notice_ids, chunk_size=args.chunk_size)

        # 3. 업체별 종목코드 로드
        company_ids = bid_df['bizno'].tolist()
        company_to_codes = load_company_industry_codes(engine, company_ids, chunk_size=args.chunk_size)

        # 4. 검증 수행
        results = validate_industry_match(bid_df, notice_to_codes, company_to_codes)

        # 5. 결과 출력
        print_summary(results, sample_count=args.sample)

    finally:
        db.close()

    print("\n✓ 검증 완료")


if __name__ == "__main__":
    main()
