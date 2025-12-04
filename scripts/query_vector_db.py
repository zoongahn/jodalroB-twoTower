#!/usr/bin/env python3
"""
Vector DB Query Script

벡터 DB에서 임베딩을 조회하고 유사도를 검색하는 스크립트입니다.

주요 기능:
1. 특정 Notice/Company ID로 임베딩 조회
2. 유사한 Notice/Company 검색 (top-k similarity search)
3. Notice-Company 간 유사도 계산
4. 벡터 DB 통계 확인

사용법:
    # 벡터 DB 통계 확인
    python scripts/query_vector_db.py --db output/vector/embeddings_v1 --stats

    # Notice 임베딩 조회
    python scripts/query_vector_db.py --db output/vector/embeddings_v1 --notice "20230106038_000"

    # Company 임베딩 조회
    python scripts/query_vector_db.py --db output/vector/embeddings_v1 --company "8928101595"

    # 유사한 Notice 검색 (top-10)
    python scripts/query_vector_db.py --db output/vector/embeddings_v1 --similar_notices "20230106038_000" --top_k 10

    # Notice-Company 유사도 계산
    python scripts/query_vector_db.py --db output/vector/embeddings_v1 --notice "20230106038_000" --company "8928101595" --similarity
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorize import EmbeddingStore


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="벡터 DB 조회 및 유사도 검색",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 통계 확인
  python scripts/query_vector_db.py --db output/vector/embeddings_v1 --stats

  # Notice 임베딩 조회
  python scripts/query_vector_db.py --db output/vector/embeddings_v1 --notice "20230106038_000"

  # 유사한 Notice 검색
  python scripts/query_vector_db.py --db output/vector/embeddings_v1 --similar_notices "20230106038_000" --top_k 10

  # Notice-Company 유사도
  python scripts/query_vector_db.py --db output/vector/embeddings_v1 --notice "20230106038_000" --company "8928101595" --similarity
        """
    )

    # 필수: 벡터 DB 경로
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="벡터 DB 경로 (접두사, 예: output/vector/embeddings_v1)"
    )

    # 조회 옵션
    parser.add_argument(
        "--stats",
        action="store_true",
        help="벡터 DB 통계 출력"
    )

    parser.add_argument(
        "--notice",
        type=str,
        help="조회할 Notice ID (예: '20230106038_000' 또는 튜플 형식)"
    )

    parser.add_argument(
        "--company",
        type=str,
        help="조회할 Company ID (예: '8928101595')"
    )

    # 유사도 검색
    parser.add_argument(
        "--similar_notices",
        type=str,
        help="유사한 Notice를 검색할 기준 Notice ID"
    )

    parser.add_argument(
        "--similar_companies",
        type=str,
        help="유사한 Company를 검색할 기준 Company ID"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="유사도 검색 시 반환할 상위 k개 (기본값: 10)"
    )

    parser.add_argument(
        "--similarity",
        action="store_true",
        help="--notice와 --company 간 유사도 계산"
    )

    return parser.parse_args()


def print_stats(store: EmbeddingStore):
    """벡터 DB 통계 출력"""
    print("=" * 80)
    print("벡터 DB 통계")
    print("=" * 80)

    stats = store.get_stats()
    print(f"\n임베딩 차원: {stats['dimension']}")
    print(f"Notice 개수: {stats['num_notices']:,}")
    print(f"Company 개수: {stats['num_companies']:,}")
    print(f"Notice 인덱스 크기: {stats['notice_index_size']:,}")
    print(f"Company 인덱스 크기: {stats['company_index_size']:,}")


def query_notice(store: EmbeddingStore, notice_id: str):
    """Notice 임베딩 조회"""
    print("=" * 80)
    print(f"Notice 임베딩 조회: {notice_id}")
    print("=" * 80)

    # Try as string first, then as tuple
    emb = store.get_notice_embedding(notice_id)

    if emb is None:
        # Try tuple format
        if "_" in notice_id:
            parts = notice_id.split("_")
            if len(parts) == 2:
                emb = store.get_notice_embedding((parts[0], parts[1]))

    if emb is None:
        print(f"\n✗ Notice ID '{notice_id}'를 찾을 수 없습니다.")
        return None

    print(f"\n✓ 임베딩 조회 성공")
    print(f"  Shape: {emb.shape}")
    print(f"  L2 Norm: {np.linalg.norm(emb):.6f}")
    print(f"  Mean: {emb.mean():.6f}")
    print(f"  Std: {emb.std():.6f}")
    print(f"  Min: {emb.min():.6f}")
    print(f"  Max: {emb.max():.6f}")
    print(f"\n  First 10 values: {emb[:10]}")

    return emb


def query_company(store: EmbeddingStore, company_id: str):
    """Company 임베딩 조회"""
    print("=" * 80)
    print(f"Company 임베딩 조회: {company_id}")
    print("=" * 80)

    # Try as string first, then as tuple
    emb = store.get_company_embedding(company_id)

    if emb is None:
        # Try tuple format
        emb = store.get_company_embedding((company_id,))

    if emb is None:
        print(f"\n✗ Company ID '{company_id}'를 찾을 수 없습니다.")
        return None

    print(f"\n✓ 임베딩 조회 성공")
    print(f"  Shape: {emb.shape}")
    print(f"  L2 Norm: {np.linalg.norm(emb):.6f}")
    print(f"  Mean: {emb.mean():.6f}")
    print(f"  Std: {emb.std():.6f}")
    print(f"  Min: {emb.min():.6f}")
    print(f"  Max: {emb.max():.6f}")
    print(f"\n  First 10 values: {emb[:10]}")

    return emb


def search_similar_notices(store: EmbeddingStore, notice_id: str, top_k: int = 10):
    """유사한 Notice 검색"""
    print("=" * 80)
    print(f"유사한 Notice 검색: {notice_id} (Top-{top_k})")
    print("=" * 80)

    # Get query embedding
    query_emb = store.get_notice_embedding(notice_id)
    if query_emb is None and "_" in notice_id:
        parts = notice_id.split("_")
        if len(parts) == 2:
            query_emb = store.get_notice_embedding((parts[0], parts[1]))

    if query_emb is None:
        print(f"\n✗ Notice ID '{notice_id}'를 찾을 수 없습니다.")
        return

    # Search in Faiss index
    query_emb = query_emb.reshape(1, -1).astype('float32')
    distances, indices = store.notice_index.search(query_emb, top_k + 1)

    print(f"\n✓ 유사도 검색 완료 (L2 거리 기준)")
    print(f"\n{'Rank':<6} {'Notice ID':<20} {'Distance':<12} {'Similarity':<12}")
    print("-" * 50)

    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue

        # Get notice ID
        if idx < len(store.notice_ids):
            nid = store.notice_ids[idx]
        else:
            nid = "Unknown"

        # Convert distance to similarity (assuming L2 normalized embeddings)
        # For L2 normalized vectors: cosine_similarity = 1 - (L2_distance^2 / 2)
        similarity = 1 - (dist ** 2 / 2)

        # Skip self (rank 0 is usually the query itself)
        if rank == 0:
            print(f"{rank+1:<6} {str(nid):<20} {dist:<12.6f} {similarity:<12.6f} (Query)")
        else:
            print(f"{rank+1:<6} {str(nid):<20} {dist:<12.6f} {similarity:<12.6f}")


def search_similar_companies(store: EmbeddingStore, company_id: str, top_k: int = 10):
    """유사한 Company 검색"""
    print("=" * 80)
    print(f"유사한 Company 검색: {company_id} (Top-{top_k})")
    print("=" * 80)

    # Get query embedding
    query_emb = store.get_company_embedding(company_id)
    if query_emb is None:
        query_emb = store.get_company_embedding((company_id,))

    if query_emb is None:
        print(f"\n✗ Company ID '{company_id}'를 찾을 수 없습니다.")
        return

    # Search in Faiss index
    query_emb = query_emb.reshape(1, -1).astype('float32')
    distances, indices = store.company_index.search(query_emb, top_k + 1)

    print(f"\n✓ 유사도 검색 완료 (L2 거리 기준)")
    print(f"\n{'Rank':<6} {'Company ID':<20} {'Distance':<12} {'Similarity':<12}")
    print("-" * 50)

    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue

        # Get company ID
        if idx < len(store.company_ids):
            cid = store.company_ids[idx]
        else:
            cid = "Unknown"

        # Convert distance to similarity
        similarity = 1 - (dist ** 2 / 2)

        # Skip self (rank 0 is usually the query itself)
        if rank == 0:
            print(f"{rank+1:<6} {str(cid):<20} {dist:<12.6f} {similarity:<12.6f} (Query)")
        else:
            print(f"{rank+1:<6} {str(cid):<20} {dist:<12.6f} {similarity:<12.6f}")


def compute_similarity(
    store: EmbeddingStore,
    notice_id: str,
    company_id: str
):
    """Notice-Company 간 유사도 계산"""
    print("=" * 80)
    print(f"Notice-Company 유사도 계산")
    print("=" * 80)
    print(f"  Notice ID: {notice_id}")
    print(f"  Company ID: {company_id}")

    # Get embeddings
    notice_emb = store.get_notice_embedding(notice_id)
    if notice_emb is None and "_" in notice_id:
        parts = notice_id.split("_")
        if len(parts) == 2:
            notice_emb = store.get_notice_embedding((parts[0], parts[1]))

    company_emb = store.get_company_embedding(company_id)
    if company_emb is None:
        company_emb = store.get_company_embedding((company_id,))

    if notice_emb is None:
        print(f"\n✗ Notice ID '{notice_id}'를 찾을 수 없습니다.")
        return

    if company_emb is None:
        print(f"\n✗ Company ID '{company_id}'를 찾을 수 없습니다.")
        return

    # Compute similarities
    # 1. Cosine similarity
    cosine_sim = np.dot(notice_emb, company_emb) / (
        np.linalg.norm(notice_emb) * np.linalg.norm(company_emb) + 1e-8
    )

    # 2. L2 distance
    l2_dist = np.linalg.norm(notice_emb - company_emb)

    # 3. Dot product
    dot_product = np.dot(notice_emb, company_emb)

    print(f"\n✓ 유사도 계산 완료")
    print(f"\n  Cosine Similarity: {cosine_sim:.6f}")
    print(f"  L2 Distance: {l2_dist:.6f}")
    print(f"  Dot Product: {dot_product:.6f}")

    # Additional features for downstream model (STEP2-1)
    distance_vec = notice_emb - company_emb
    print(f"\n  Feature Vector Shape for ML Model:")
    print(f"    - Notice embedding: {notice_emb.shape[0]} dims")
    print(f"    - Company embedding: {company_emb.shape[0]} dims")
    print(f"    - Distance vector: {distance_vec.shape[0]} dims")
    print(f"    - Cosine similarity: 1 dim")
    print(f"    - Total: {notice_emb.shape[0] + company_emb.shape[0] + distance_vec.shape[0] + 1} dims")


def main():
    args = parse_args()

    print("=" * 80)
    print("벡터 DB 조회 스크립트")
    print("=" * 80)
    print(f"벡터 DB: {args.db}")
    print()

    # Load vector DB
    print("벡터 DB 로딩 중...")
    store = EmbeddingStore()
    store.load(args.db)
    print("✓ 로딩 완료\n")

    # Execute commands
    if args.stats:
        print_stats(store)

    if args.notice and not args.similarity:
        query_notice(store, args.notice)

    if args.company and not args.similarity:
        query_company(store, args.company)

    if args.similar_notices:
        search_similar_notices(store, args.similar_notices, args.top_k)

    if args.similar_companies:
        search_similar_companies(store, args.similar_companies, args.top_k)

    if args.similarity:
        if not args.notice or not args.company:
            print("✗ --similarity 옵션은 --notice와 --company가 모두 필요합니다.")
        else:
            compute_similarity(store, args.notice, args.company)

    # If no command, show stats by default
    if not any([
        args.stats,
        args.notice,
        args.company,
        args.similar_notices,
        args.similar_companies,
        args.similarity
    ]):
        print_stats(store)

    print("\n" + "=" * 80)
    print("✓ 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
