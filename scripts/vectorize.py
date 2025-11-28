#!/usr/bin/env python3
"""
Vectorize: 학습된 Two-Tower 모델로부터 임베딩을 추출하여 벡터 DB에 저장

주요 기능:
1. 모델 체크포인트 로드
2. 전체 Notice/Company 임베딩 추출
3. Faiss 벡터 DB에 저장
4. 디스크에 영구 저장

사용법:
    # 기본 사용
    python scripts/vectorize.py \\
        --checkpoint output/models/20251026_040021/best_model.pt \\
        --output data/vectorize/embeddings

    # 배치 크기 조정
    python scripts/vectorize.py \\
        --checkpoint output/models/20251026_040021/best_model.pt \\
        --output data/vectorize/embeddings \\
        --batch_size 2048

    # CPU 사용
    python scripts/vectorize.py \\
        --checkpoint output/models/20251026_040021/best_model.pt \\
        --output data/vectorize/embeddings \\
        --device cpu
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.database_connector import DatabaseConnector
from preprocess.torchrec.schema import build_torchrec_schema_from_meta
from src.vectorize import EmbeddingExtractor, EmbeddingStore


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="학습된 Two-Tower 모델로부터 임베딩 추출 및 벡터 DB 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 사용
  python scripts/vectorize.py \\
      --checkpoint output/models/20251026_040021/best_model.pt \\
      --output data/vectorize/embeddings

  # 배치 크기 조정 (GPU 메모리에 따라)
  python scripts/vectorize.py \\
      --checkpoint output/models/20251026_040021/best_model.pt \\
      --output data/vectorize/embeddings \\
      --batch_size 2048

  # CPU 사용
  python scripts/vectorize.py \\
      --checkpoint output/models/20251026_040021/best_model.pt \\
      --output data/vectorize/embeddings \\
      --device cpu
        """
    )

    # 필수 인자
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="모델 체크포인트 경로 (예: output/models/20251026_040021/best_model.pt)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="벡터 DB 저장 경로 (접두사, 예: data/vectorize/embeddings)"
    )

    # 선택 인자
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="모델 설정 파일 경로 (기본값: 체크포인트와 같은 디렉토리의 config.json)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="임베딩 추출 배치 크기 (기본값: 1024)"
    )

    parser.add_argument(
        "--feature_chunksize",
        type=int,
        default=5000,
        help="피처 로딩 청크 크기 (기본값: 5000)"
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
        "--skip_analysis",
        action="store_true",
        help="임베딩 품질 분석 생략"
    )

    return parser.parse_args()


def analyze_embeddings(notice_embeddings, company_embeddings):
    """임베딩 품질 간단 분석"""
    import numpy as np

    print("\n" + "=" * 80)
    print("임베딩 품질 분석")
    print("=" * 80)

    # Notice 임베딩
    print(f"\n[Notice 임베딩]")
    print(f"  Shape: {notice_embeddings.shape}")
    print(f"  Mean: {notice_embeddings.mean():.6f}")
    print(f"  Std: {notice_embeddings.std():.6f}")
    print(f"  Min: {notice_embeddings.min():.6f}")
    print(f"  Max: {notice_embeddings.max():.6f}")

    # L2 norm 통계
    notice_norms = np.linalg.norm(notice_embeddings, axis=1)
    print(f"  L2 Norm - Mean: {notice_norms.mean():.6f}, Std: {notice_norms.std():.6f}")

    # Company 임베딩
    print(f"\n[Company 임베딩]")
    print(f"  Shape: {company_embeddings.shape}")
    print(f"  Mean: {company_embeddings.mean():.6f}")
    print(f"  Std: {company_embeddings.std():.6f}")
    print(f"  Min: {company_embeddings.min():.6f}")
    print(f"  Max: {company_embeddings.max():.6f}")

    # L2 norm 통계
    company_norms = np.linalg.norm(company_embeddings, axis=1)
    print(f"  L2 Norm - Mean: {company_norms.mean():.6f}, Std: {company_norms.std():.6f}")

    # 정규화 여부 확인
    norm_check_notice = np.abs(notice_norms - 1.0).mean()
    norm_check_company = np.abs(company_norms - 1.0).mean()

    print(f"\n[정규화 체크]")
    if norm_check_notice < 0.01 and norm_check_company < 0.01:
        print(f"  ✓ 임베딩이 L2 정규화되어 있습니다 (코사인 유사도 사용 가능)")
    else:
        print(f"  ⚠ 임베딩이 정규화되지 않았습니다 (유클리드 거리 사용)")
        print(f"    Notice deviation: {norm_check_notice:.6f}")
        print(f"    Company deviation: {norm_check_company:.6f}")


def main():
    args = parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 80)
        print("Vectorize: 임베딩 추출 및 벡터 DB 생성")
        print("=" * 80)
        print(f"체크포인트: {args.checkpoint}")
        print(f"출력 경로: {args.output}")
        print(f"배치 크기: {args.batch_size}")
        print(f"디바이스: {args.device}")
        print()

    start_time = datetime.now()

    # 1. 데이터베이스 연결
    if verbose:
        print("=" * 80)
        print("데이터베이스 연결")
        print("=" * 80)

    db = DatabaseConnector()
    engine = db.engine

    if verbose:
        print("✓ 데이터베이스 연결 완료")

    # 2. 스키마 구축
    if verbose:
        print("\n" + "=" * 80)
        print("스키마 구축")
        print("=" * 80)

    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    if verbose:
        print(f"✓ Notice 피처: {len(schema.notice.categorical)}개 범주형, {len(schema.notice.numeric)}개 수치형")
        print(f"✓ Company 피처: {len(schema.company.categorical)}개 범주형, {len(schema.company.numeric)}개 수치형")

    # 3. 임베딩 추출기 초기화
    if verbose:
        print("\n" + "=" * 80)
        print("임베딩 추출기 초기화")
        print("=" * 80)

    extractor = EmbeddingExtractor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )

    # 4. 전체 임베딩 추출
    (notice_ids, notice_embeddings), (company_ids, company_embeddings) = extractor.extract_all_embeddings(
        db_engine=engine,
        schema=schema,
        batch_size=args.batch_size,
        feature_chunksize=args.feature_chunksize,
        verbose=verbose
    )

    # 5. 임베딩 분석 (선택)
    if not args.skip_analysis and verbose:
        analyze_embeddings(notice_embeddings, company_embeddings)

    # 6. 벡터 DB에 저장
    if verbose:
        print("\n" + "=" * 80)
        print("벡터 DB 생성 및 저장")
        print("=" * 80)

    embedding_dim = notice_embeddings.shape[1]
    store = EmbeddingStore(dimension=embedding_dim)

    # 임베딩 추가 (ID를 튜플 그대로 저장)
    # Notice IDs: (bidntceno, bidntceord) 튜플
    # Company IDs: (bizno,) 튜플 또는 단일값
    store.add_notices(notice_ids, notice_embeddings)
    store.add_companies(company_ids, company_embeddings)

    # 7. 디스크에 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store.save(str(output_path))

    if verbose:
        print(f"\n✓ 벡터 DB 저장 완료: {output_path}_*")

    # 8. 통계 출력
    if verbose:
        print("\n" + "=" * 80)
        print("최종 통계")
        print("=" * 80)

        stats = store.get_stats()
        print(f"  임베딩 차원: {stats['dimension']}")
        print(f"  Notice 개수: {stats['num_notices']:,}")
        print(f"  Company 개수: {stats['num_companies']:,}")
        print(f"  Notice 인덱스 크기: {stats['notice_index_size']:,}")
        print(f"  Company 인덱스 크기: {stats['company_index_size']:,}")

        elapsed = datetime.now() - start_time
        print(f"\n  소요 시간: {elapsed}")

    # 9. 사용 예시 출력
    if verbose:
        print("\n" + "=" * 80)
        print("사용 방법")
        print("=" * 80)
        print(f"\n벡터 DB를 불러오려면:")
        print(f"""
from src.vectorize import EmbeddingStore

store = EmbeddingStore()
store.load("{args.output}")

# Notice 임베딩 조회 (복합키 사용)
notice_emb = store.get_notice_embedding(("12345", "1"))  # (bidntceno, bidntceord)

# Company 임베딩 조회
company_emb = store.get_company_embedding(("1234567890",))  # (bizno,)

# 배치 조회
notice_ids = [("12345", "1"), ("12345", "2"), ("12345", "3")]
valid_ids, embeddings = store.get_notice_embeddings_batch(notice_ids)
        """)

        print(f"\n임베딩 분석:")
        print(f"  python test/analyze_embeddings.py")

    db.close()

    if verbose:
        print("\n" + "=" * 80)
        print("✓ 완료!")
        print("=" * 80)


if __name__ == "__main__":
    main()
