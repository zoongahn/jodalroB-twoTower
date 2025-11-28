#!/usr/bin/env python3
"""
Predict: Two-Tower 모델을 사용한 Notice-Company 추천 예측

주요 기능:
1. 새로운 Notice에 대해 적합한 Company 추천
2. 배치 예측 지원
3. 결과를 CSV로 저장
4. Top-K 및 유사도 threshold 기반 필터링

사용법:
    # 단일 Notice 예측 - Top 10
    python scripts/predict.py \
        --checkpoint output/models/20251023_055430_final/checkpoint_epoch_1.pt \
        --vector_db output/vector/embeddings_v1 \
        --notice "20230106038" "000" \
        --top_k 10

    # 유사도 threshold로 필터링 (0.5 이상)
    python scripts/predict.py \
        --checkpoint output/models/20251023_055430_final/checkpoint_epoch_1.pt \
        --vector_db output/vector/embeddings_v1 \
        --notice "20230106038" "000" \
        --min_similarity 0.5

    # Top-K와 threshold 함께 사용 (둘 다 만족)
    python scripts/predict.py \
        --checkpoint output/models/20251023_055430_final/checkpoint_epoch_1.pt \
        --vector_db output/vector/embeddings_v1 \
        --notice "20230106038" "000" \
        --top_k 100 \
        --min_similarity 0.3

    # 배치 예측 (CSV 파일 사용)
    python scripts/predict.py \
        --checkpoint output/models/20251023_055430_final/checkpoint_epoch_1.pt \
        --vector_db output/vector/embeddings_v1 \
        --batch notices.csv \
        --output predictions.csv \
        --min_similarity 0.4

    # CSV 형식:
    # bidntceno,bidntceord
    # 20230106038,000
    # 20230106039,000
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.database_connector import DatabaseConnector
from src.prediction import TwoTowerPredictor


def get_display_width(text: str) -> int:
    """
    텍스트의 실제 화면 표시 너비 계산 (한글은 2칸, 영문/숫자는 1칸)
    """
    width = 0
    for char in text:
        # 한글 및 전각 문자는 2칸
        if ord(char) >= 0x1100:  # 한글 및 대부분의 동아시아 문자
            width += 2
        else:
            width += 1
    return width


def pad_text(text: str, width: int) -> str:
    """
    텍스트를 지정된 너비로 패딩 (한글 너비 고려)
    """
    current_width = get_display_width(text)
    padding = width - current_width
    if padding > 0:
        return text + ' ' * padding
    return text


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Two-Tower 모델을 사용한 Notice-Company 추천 예측",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 필수 인자
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="모델 체크포인트 경로 (예: output/models/20251023_055430/best_model.pt)"
    )

    parser.add_argument(
        "--vector_db",
        type=str,
        default=None,
        help="벡터 DB 경로 (접두사, 예: data/vectorize/embeddings). --no-vector-db 사용 시 불필요"
    )

    parser.add_argument(
        "--no-vector-db",
        action="store_true",
        help="Vector DB 없이 DB에서 직접 Company 임베딩 생성 (느리지만 최신 데이터 사용)"
    )

    # 예측 모드
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--notice",
        nargs=2,
        metavar=("BIDNTCENO", "BIDNTCEORD"),
        help="단일 Notice 예측 (예: --notice '20230106038' '000')"
    )

    group.add_argument(
        "--batch",
        type=str,
        metavar="CSV_FILE",
        help="배치 예측용 CSV 파일 (컬럼: bidntceno, bidntceord)"
    )

    # 선택 인자
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="모델 설정 파일 경로 (기본값: 체크포인트와 같은 디렉토리의 config.json)"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="추천할 상위 K개 Company (기본값: None). --min_similarity와 함께 사용 가능"
    )

    parser.add_argument(
        "--min_similarity",
        type=float,
        default=None,
        help="최소 유사도 threshold (예: 0.5). 이 값보다 높은 유사도의 Company만 반환. "
             "top_k와 함께 사용하면 둘 다 만족하는 결과 반환"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 CSV 파일 경로 (배치 모드에서만 사용)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="사용할 디바이스 (기본값: cuda)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="진행 상황 출력 최소화"
    )

    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="결과를 JSON 파일로 저장할 경로 (단일 예측 모드에서만 사용). None이면 저장하지 않음"
    )

    parser.add_argument(
        "--return-embeddings",
        action="store_true",
        help="임베딩 벡터를 결과에 포함 (JSON 출력 시 함께 저장됨)"
    )

    return parser.parse_args()


def predict_single(
    predictor: TwoTowerPredictor,
    bidntceno: str,
    bidntceord: str,
    top_k: int,
    min_similarity: float = None,
    json_output_path: str = None,
    return_embeddings: bool = False
):
    """단일 Notice 예측"""
    result = predictor.predict_for_notice(
        bidntceno=bidntceno,
        bidntceord=bidntceord,
        top_k=top_k,
        min_similarity=min_similarity,
        return_embeddings=return_embeddings
    )

    print("\n" + "=" * 80)
    print("예측 결과")
    print("=" * 80)
    print(f"\n공고 번호: {result['notice_id']}")

    # 공고명 출력
    if result.get('notice_title'):
        print(f"공고명: {result['notice_title']}")

    # 필터링 조건 출력
    filter_desc = []
    if top_k:
        filter_desc.append(f"Top-{top_k}")
    if min_similarity is not None:
        filter_desc.append(f"유사도 >= {min_similarity}")

    filter_str = " & ".join(filter_desc) if filter_desc else "모든"
    print(f"\n예측 참여 업체 ({filter_str}):")
    print(f"총 {len(result['top_k_companies'])}개")

    # 헤더 출력
    header_rank = pad_text("순번", 6)
    header_bizno = pad_text("사업자등록번호", 18)
    header_name = pad_text("업체명", 50)
    header_score = pad_text("유사도", 12)
    print(f"{header_rank} {header_bizno} {header_name} {header_score}")
    print("-" * 90)

    company_names = result.get('company_names', {})
    for rank, (company_id, score) in enumerate(result['top_k_companies'], 1):
        company_name = company_names.get(company_id, '(조회 실패)')

        # 각 컬럼을 한글 너비 고려하여 패딩
        col_rank = pad_text(str(rank), 6)
        col_bizno = pad_text(str(company_id), 18)
        col_name = pad_text(company_name[:25], 50)  # 업체명이 너무 길면 25자로 자름
        col_score = f"{score:.6f}"

        print(f"{col_rank} {col_bizno} {col_name} {col_score}")

    # JSON 파일로 저장 (옵션)
    if json_output_path:
        import json
        import numpy as np

        # notice_id가 tuple이나 list인 경우 분리
        notice_id_val = result['notice_id']
        if isinstance(notice_id_val, (tuple, list)):
            bidntceno_val, bidntceord_val = notice_id_val[0], notice_id_val[1]
        else:
            # notice_id가 "bidntceno_bidntceord" 형식이면 분리
            parts = str(notice_id_val).split('_')
            if len(parts) >= 2:
                bidntceno_val, bidntceord_val = parts[0], parts[1]
            else:
                bidntceno_val, bidntceord_val = str(notice_id_val), "000"

        # Notice 정보 구성
        notice_info = {
            "bidntceno": bidntceno_val,
            "bidntceord": bidntceord_val,
            "title": result.get('notice_title')
        }

        # 임베딩이 포함되어 있으면 추가
        if return_embeddings and 'notice_embedding' in result:
            notice_info['embedding'] = result['notice_embedding'].tolist()

        # Company 정보 구성 (dict 형식으로)
        top_k_companies_dict = {}
        company_names = result.get('company_names', {})

        for company_id, similarity_score in result['top_k_companies']:
            company_info = {
                "similarity": similarity_score,
                "name": company_names.get(company_id, None)
            }

            # Company 임베딩이 포함되어 있으면 추가
            if return_embeddings and 'company_embeddings' in result and company_id in result['company_embeddings']:
                company_info['embedding'] = result['company_embeddings'][company_id].tolist()

            top_k_companies_dict[company_id] = company_info

        # 최종 JSON 구조
        json_result = {
            'notice': notice_info,
            'top_k_companies': top_k_companies_dict
        }

        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)

        print(f"\n✓ JSON 결과 저장: {json_output_path}")


def predict_batch(
    predictor: TwoTowerPredictor,
    batch_file: str,
    top_k: int,
    min_similarity: float = None,
    output_file: str = None
):
    """배치 예측"""
    # CSV 파일 로드
    print(f"\n배치 파일 로드 중: {batch_file}")
    df = pd.read_csv(batch_file)

    if 'bidntceno' not in df.columns or 'bidntceord' not in df.columns:
        raise ValueError("CSV 파일에 'bidntceno'와 'bidntceord' 컬럼이 필요합니다")

    # Notice 리스트 생성
    notice_ids = [(str(row['bidntceno']), str(row['bidntceord'])) for _, row in df.iterrows()]
    print(f"✓ {len(notice_ids)}개 Notice 로드 완료")

    # 배치 예측
    results = predictor.predict_for_notices_batch(
        notice_ids=notice_ids,
        top_k=top_k,
        min_similarity=min_similarity,
        show_progress=True
    )

    # 결과를 DataFrame으로 변환
    rows = []
    for result in results:
        if 'error' in result:
            # 에러 발생한 경우
            rows.append({
                'bidntceno': result['notice_id'][0],
                'bidntceord': result['notice_id'][1],
                'notice_title': None,
                'rank': None,
                'company_id': None,
                'company_name': None,
                'similarity_score': None,
                'error': result['error']
            })
        else:
            # 정상 예측
            notice_title = result.get('notice_title')
            company_names = result.get('company_names', {})

            for rank, (company_id, score) in enumerate(result['top_k_companies'], 1):
                rows.append({
                    'bidntceno': result['notice_id'][0],
                    'bidntceord': result['notice_id'][1],
                    'notice_title': notice_title,
                    'rank': rank,
                    'company_id': str(company_id),
                    'company_name': company_names.get(company_id, None),
                    'similarity_score': score,
                    'error': None
                })

    result_df = pd.DataFrame(rows)

    # 결과 출력
    print("\n" + "=" * 80)
    print("배치 예측 결과")
    print("=" * 80)
    print(f"\n총 예측 수: {len(results)}개 공고")
    print(f"성공: {len([r for r in results if 'error' not in r])}개")
    print(f"실패: {len([r for r in results if 'error' in r])}개")

    # 결과 저장
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 결과 저장: {output_path}")
        print(f"  행 수: {len(result_df)}")
    else:
        # 화면에 출력 (처음 20개만)
        print("\n[처음 20개 결과]")
        print(result_df.head(20).to_string(index=False))

    return result_df


def main():
    args = parse_args()

    # 인자 검증
    use_vector_db = not args.no_vector_db
    if use_vector_db and args.vector_db is None:
        print("❌ 오류: Vector DB 사용 시 --vector_db 경로가 필요합니다")
        print("   또는 --no-vector-db 플래그를 사용하세요")
        sys.exit(1)

    if not args.quiet:
        print("=" * 80)
        print("Two-Tower 모델 예측 시작")
        print("=" * 80)
        print(f"체크포인트: {args.checkpoint}")
        if use_vector_db:
            print(f"벡터 DB: {args.vector_db}")
        else:
            print(f"벡터 DB: 미사용 (DB에서 직접 생성)")
        print(f"Top-K: {args.top_k}")
        print(f"디바이스: {args.device}")
        print()

    start_time = datetime.now()

    # 데이터베이스 연결
    if not args.quiet:
        print("=" * 80)
        print("데이터베이스 연결")
        print("=" * 80)

    db = DatabaseConnector()
    engine = db.engine

    if not args.quiet:
        print("✓ 데이터베이스 연결 완료\n")

    # 예측기 초기화
    predictor = TwoTowerPredictor(
        checkpoint_path=args.checkpoint,
        db_engine=engine,
        vector_db_path=args.vector_db if use_vector_db else None,
        use_vector_db=use_vector_db,
        config_path=args.config,
        device=args.device
    )

    # 예측 실행
    if args.notice:
        # 단일 예측
        bidntceno, bidntceord = args.notice
        predict_single(
            predictor,
            bidntceno,
            bidntceord,
            args.top_k,
            args.min_similarity,
            json_output_path=args.json_output,
            return_embeddings=args.return_embeddings
        )

    elif args.batch:
        # 배치 예측
        predict_batch(predictor, args.batch, args.top_k, args.min_similarity, args.output)

    # 소요 시간
    elapsed = datetime.now() - start_time

    if not args.quiet:
        print("\n" + "=" * 80)
        print("✓ 예측 완료!")
        print("=" * 80)
        print(f"소요 시간: {elapsed}")

    db.close()


if __name__ == "__main__":
    main()
