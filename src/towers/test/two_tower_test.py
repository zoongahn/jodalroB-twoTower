"""
TwoTowerModelV2 테스트 - 실제 DB 데이터 사용

기존 two_tower_test.py와 동일한 구조, EmbeddingBagCollection 버전
"""

import torch
from src.towers.two_tower_model_v2 import TwoTowerModelV2
from src.towers.kjt_utils import create_ebc_collate_fn_v2
from preprocess.schema import build_torchrec_schema_from_meta
from preprocess.feature_store import build_feature_store
from preprocess.feature_preprocessor import FeaturePreprocessor
from data.database_connector import DatabaseConnector
from torchrec.modules.embedding_configs import PoolingType


def test_two_tower_model_v2():
    """TwoTowerModelV2 간단 테스트 - 실제 DB 데이터"""
    print("="*80)
    print("TwoTowerModelV2 테스트 시작 (실제 DB 데이터)")
    print("="*80)

    # DB 연결 및 스키마 구축
    print("\n1. DB 연결 및 스키마 구축...")
    db = DatabaseConnector()
    engine = db.engine

    config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**config)

    # 실제 피처 데이터 로딩 (limit=1000으로 빠른 테스트)
    print("\n2. 실제 피처 데이터 로딩...")
    limit = 1000

    notice_store = build_feature_store(
        engine, schema.notice,
        chunksize=1000,
        limit=limit,
        show_progress=True
    )

    company_store = build_feature_store(
        engine, schema.company,
        chunksize=1000,
        limit=limit,
        show_progress=True
    )

    print(f"\n   Notice 데이터: {len(notice_store['ids'])}개")
    print(f"   Company 데이터: {len(company_store['ids'])}개")

    # 전처리 (projection)
    print("\n3. 피처 전처리 (projection)...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preprocessor = FeaturePreprocessor(
        schema=schema,
        device=device,
        num_proj_dim=128,
        text_proj_dim=128,
        batch_size=1024
    )

    preprocessed_stores = preprocessor.preprocess_stores(notice_store, company_store)

    print(f"   Notice dense_projected: {preprocessed_stores['notice']['dense_projected'].shape}")
    print(f"   Company dense_projected: {preprocessed_stores['company']['dense_projected'].shape}")

    # TwoTowerModelV2 생성
    print("\n4. TwoTowerModelV2 생성...")

    # CRITICAL: Notice와 Company는 dense 차원이 다름!
    # Notice: 256, Company: 128
    from src.towers.tower_v2 import NoticeTowerV2, CompanyTowerV2

    notice_tower = NoticeTowerV2(
        metadata_path=config["metadata_path"],
        categorical_embedding_dim=32,
        dense_input_dim=256,  # Notice dense
        tower_hidden_dims=[256, 128],
        final_embedding_dim=128,
        dropout_rate=0.2,
        pooling_mode=PoolingType.MEAN,
        device=device,
        use_fp16=False
    )

    company_tower = CompanyTowerV2(
        metadata_path=config["metadata_path"],
        categorical_embedding_dim=32,
        dense_input_dim=128,  # Company dense (다름!)
        tower_hidden_dims=[256, 128],
        final_embedding_dim=128,
        dropout_rate=0.2,
        pooling_mode=PoolingType.MEAN,
        device=device,
        use_fp16=False
    )

    # 간단한 래퍼
    class TwoTowerWrapper(torch.nn.Module):
        def __init__(self, notice_tower, company_tower):
            super().__init__()
            self.notice_tower = notice_tower
            self.company_tower = company_tower

        def forward(self, batch):
            notice_emb = self.notice_tower(batch["notice"])
            company_emb = self.company_tower(batch["company"])
            return {"notice_embeddings": notice_emb, "company_embeddings": company_emb}

    model = TwoTowerWrapper(notice_tower, company_tower)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   총 파라미터: {total_params:,}")

    # Collate function 생성
    print("\n5. Collate function 생성...")
    collate_fn = create_ebc_collate_fn_v2(
        notice_store=preprocessed_stores['notice'],
        company_store=preprocessed_stores['company'],
        notice_keys=schema.notice.categorical,
        company_keys=schema.company.categorical,
        device=device,
        use_fp16=False,
        cache_key="test_v2"
    )

    # 더미 배치 생성 (실제 인덱스 사용)
    print("\n6. 더미 배치 생성 (실제 데이터)...")
    batch_size = min(64, len(notice_store['ids']))

    # 실제 인덱스로 배치 생성
    dummy_batch = [
        {"notice_idx": i, "company_idx": i % len(company_store['ids'])}
        for i in range(batch_size)
    ]

    batch = collate_fn(dummy_batch)

    print(f"   Notice dense: {batch['notice']['dense'].shape}")
    print(f"   Notice KJT stride: {batch['notice']['kjt'].stride()}")
    print(f"   Company dense: {batch['company']['dense'].shape}")
    print(f"   Company KJT stride: {batch['company']['kjt'].stride()}")

    # Forward Pass 테스트
    print("\n7. Forward Pass 테스트...")
    model.eval()

    with torch.no_grad():
        outputs = model(batch)

        notice_emb = outputs["notice_embeddings"]
        company_emb = outputs["company_embeddings"]

        print(f"   Notice 임베딩: {notice_emb.shape}")
        print(f"   Company 임베딩: {company_emb.shape}")

        # L2 norm 확인 (정규화되어 있으므로 모두 1이어야 함)
        notice_norms = torch.norm(notice_emb, p=2, dim=1)
        company_norms = torch.norm(company_emb, p=2, dim=1)

        print(f"   Notice L2 norms: min={notice_norms.min():.4f}, max={notice_norms.max():.4f}, mean={notice_norms.mean():.4f}")
        print(f"   Company L2 norms: min={company_norms.min():.4f}, max={company_norms.max():.4f}, mean={company_norms.mean():.4f}")

        # 유사도 계산
        similarity_matrix = torch.mm(notice_emb, company_emb.t())
        print(f"   유사도 행렬: {similarity_matrix.shape}")
        print(f"   대각선 평균 (positive pairs): {torch.diag(similarity_matrix).mean():.4f}")
        print(f"   전체 평균: {similarity_matrix.mean():.4f}")

    print("\n   ✅ Forward pass 성공!")

    # Gradient 테스트
    print("\n8. Gradient 테스트...")
    model.train()

    outputs = model(batch)
    notice_emb = outputs["notice_embeddings"]
    company_emb = outputs["company_embeddings"]

    # 유사도 행렬
    similarity_matrix = torch.mm(notice_emb, company_emb.t())

    # Cross entropy loss (대각선을 positive로)
    labels = torch.arange(len(notice_emb), device=device)
    loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)

    print(f"   손실: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Gradient 확인
    grad_count = 0
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            grad_count += 1

    print(f"   총 {grad_count}개 파라미터에 gradient 계산됨")
    if grad_norms:
        print(f"   평균 gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")

    print("\n   ✅ Gradient 테스트 성공!")

    print("\n" + "="*80)
    print("TwoTowerModelV2 테스트 완료!")
    print("="*80)


if __name__ == "__main__":
    test_two_tower_model_v2()
