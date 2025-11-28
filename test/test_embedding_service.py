"""
EmbeddingService 테스트 스크립트
"""

import numpy as np
from src.prediction.embedding_service import EmbeddingService
from database.database_connector import DatabaseConnector

# DB 연결
db = DatabaseConnector()
engine = db.engine

# 서비스 초기화
service = EmbeddingService(
    checkpoint_path="output/models/20251119_024125_final/checkpoint_epoch_10.pt",  # 실제 경로로 변경
    db_engine=engine,
    device="cuda"
)

print("\n" + "=" * 80)
print("테스트 시작")
print("=" * 80)

# 1. 공고 임베딩 테스트
print("\n[1] 공고 임베딩 테스트")
notice_emb = service.get_notice_embedding("20240414633", "000")
if notice_emb is not None:
    print(f"  ✓ shape: {notice_emb.shape}")
    print(f"  ✓ norm: {np.linalg.norm(notice_emb):.4f}")
    print(f"  ✓ min/max: {notice_emb.min():.4f} / {notice_emb.max():.4f}")
else:
    print("  ✗ 공고를 찾을 수 없음")

# 2. 업체 임베딩 테스트
print("\n[2] 업체 임베딩 테스트")
company_emb = service.get_company_embedding("4108617283")
if company_emb is not None:
    print(f"  ✓ shape: {company_emb.shape}")
    print(f"  ✓ norm: {np.linalg.norm(company_emb):.4f}")
    print(f"  ✓ min/max: {company_emb.min():.4f} / {company_emb.max():.4f}")
else:
    print("  ✗ 업체를 찾을 수 없음")

# 3. 배치 테스트
print("\n[3] 배치 테스트")
notice_ids = [("20240414633", "000"), ("20140302595", "000")]
valid_ids, embeddings = service.get_notice_embeddings_batch(notice_ids)
print(f"  ✓ 요청: {len(notice_ids)}개, 결과: {len(valid_ids)}개")
if len(valid_ids) > 0:
    print(f"  ✓ embeddings shape: {embeddings.shape}")

print("\n" + "=" * 80)
print("테스트 완료")
print("=" * 80)
