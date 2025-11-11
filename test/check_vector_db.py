#!/usr/bin/env python3
"""
Vector DB에 저장된 Notice ID들을 확인하는 스크립트
"""
import pickle
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_vector_db(metadata_path: str):
    """Vector DB metadata 로드 및 확인"""
    print(f"Loading metadata from: {metadata_path}")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    notice_ids = metadata['notice_ids']
    company_ids = metadata['company_ids']
    notice_id_to_idx = metadata['notice_id_to_idx']

    print("\n" + "=" * 80)
    print("Vector DB 통계")
    print("=" * 80)
    print(f"총 Notice 수: {len(notice_ids):,}")
    print(f"총 Company 수: {len(company_ids):,}")
    print(f"Embedding 차원: {metadata.get('dimension', 'N/A')}")

    # Notice ID 형식 확인
    print("\n" + "=" * 80)
    print("Notice ID 샘플 (처음 10개)")
    print("=" * 80)
    for i, notice_id in enumerate(notice_ids[:10]):
        idx = notice_id_to_idx.get(notice_id)
        print(f"{i+1}. {notice_id} (타입: {type(notice_id).__name__}, Faiss idx: {idx})")

    # 특정 Notice 검색
    target_notice = ("20210408284", "000")
    print("\n" + "=" * 80)
    print(f"특정 Notice 검색: {target_notice}")
    print("=" * 80)

    if target_notice in notice_id_to_idx:
        idx = notice_id_to_idx[target_notice]
        print(f"✓ 발견! Faiss 인덱스: {idx}")
    else:
        print("✗ Vector DB에 없음")

        # 유사한 ID 찾기
        print("\n유사한 Notice ID 검색 중...")
        similar = [nid for nid in notice_ids if "20210408284" in str(nid)]
        if similar:
            print(f"발견된 유사 ID: {similar[:5]}")
        else:
            print("유사한 ID 없음")

    # Company ID 샘플
    print("\n" + "=" * 80)
    print("Company ID 샘플 (처음 10개)")
    print("=" * 80)
    for i, company_id in enumerate(company_ids[:10]):
        print(f"{i+1}. {company_id} (타입: {type(company_id).__name__})")

    return notice_ids, company_ids

if __name__ == "__main__":
    metadata_path = "output/vector/embeddings_v1_metadata.pkl"
    notice_ids, company_ids = check_vector_db(metadata_path)

    print("\n" + "=" * 80)
    print("테스트용 Notice ID (실제 존재하는 것)")
    print("=" * 80)
    print("다음 명령어로 예측 테스트 가능:")
    print()

    # 처음 5개 Notice로 테스트 명령 출력
    for i, notice_id in enumerate(notice_ids[:5]):
        if isinstance(notice_id, tuple) and len(notice_id) == 2:
            print(f"python scripts/predict.py \\")
            print(f"    --checkpoint output/models/20251023_055430_final/checkpoint_epoch_1.pt \\")
            print(f"    --vector_db output/vector/embeddings_v1 \\")
            print(f"    --notice \"{notice_id[0]}\" \"{notice_id[1]}\" \\")
            print(f"    --top_k 10")
            print()
            if i >= 2:  # 3개만 출력
                break
