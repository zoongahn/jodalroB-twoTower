#!/usr/bin/env python3
"""
Positive Pair Alignment 검증 스크립트
notice-company 페어가 올바르게 정렬되었는지 확인
"""

import torch
from data.database_connector import DatabaseConnector
from preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders
from src.towers.two_tower_train_task import create_two_tower_train_task


def verify_pair_alignment():
    """배치 내 notice-company 정렬 검증"""
    print("=== Positive Pair Alignment 검증 시작 ===\n")

    # 1. 데이터베이스 연결
    db = DatabaseConnector()
    engine = db.engine

    # 2. 스키마 구축
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # 3. 작은 데이터로 DataLoader 생성
    print("DataLoader 생성 중... (test_mode=True, pair_limit=10000)")
    train_loader, test_loader = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=256,
        test_split=0.2,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        chunk_size=1000000,
        feature_chunksize=1000,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=10000,
    )

    print(f"Train 배치 수: {len(train_loader)}\n")

    # 4. 모델 생성
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    notice_categorical_keys = schema.notice.categorical
    company_categorical_keys = schema.company.categorical

    train_task = create_two_tower_train_task(
        notice_categorical_keys=notice_categorical_keys,
        company_categorical_keys=company_categorical_keys,
        metadata_path="meta/metadata.csv",
        categorical_embedding_dim=32,
        notice_dense_input_dim=256,
        company_dense_input_dim=128,
        tower_hidden_dims=[512, 256],
        final_embedding_dim=128,
        dropout_rate=0.1,
        temperature=1.0,
        loss_type="cross_entropy",
        device=device
    )

    # 5. 첫 번째 배치로 정렬 검증
    print("첫 번째 배치 추출 중...")
    batch = next(iter(train_loader))

    # Forward pass (alignment check 포함)
    train_task.eval()
    with torch.no_grad():
        result = train_task(batch, return_metrics=True)

        similarity_matrix = result["similarity_matrix"]
        accuracy = result["accuracy"]

        # 대각선 vs 최대값 분석
        B = similarity_matrix.size(0)
        diag_values = similarity_matrix.diag()
        max_values = similarity_matrix.max(dim=1)[0]
        diag_is_max = (diag_values == max_values).float().mean().item()

        # 정확도 분석
        predicted_indices = torch.argmax(similarity_matrix, dim=1)
        correct_indices = torch.arange(B, device=similarity_matrix.device)
        row_accuracy = (predicted_indices == correct_indices).float().mean().item()

        print("\n=== 검증 결과 ===")
        print(f"배치 크기: {B}")
        print(f"Top-1 Accuracy: {row_accuracy:.3f}")
        print(f"대각선이 row-max인 비율: {diag_is_max:.3f}")
        print(f"Positive Similarity (평균): {result['positive_similarity_mean']:.4f}")
        print(f"Negative Similarity (평균): {result['negative_similarity_mean']:.4f}")
        print(f"Similarity Gap: {result['similarity_gap']:.4f}")

        # 판정
        print("\n=== 판정 ===")
        if row_accuracy > 0.7:
            print("✅ PASS: Positive pair 정렬이 올바릅니다!")
        elif row_accuracy > 0.3:
            print("⚠️  WARNING: Positive pair 정렬이 불완전합니다.")
        else:
            print("❌ FAIL: Positive pair 정렬이 실패했습니다!")
            print("   → DataLoader shuffle 설정을 확인하세요")


if __name__ == "__main__":
    verify_pair_alignment()
