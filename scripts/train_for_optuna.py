#!/usr/bin/env python3
"""
Optuna 하이퍼파라미터 튜닝 전용 학습 스크립트
train_common.py의 공통 함수를 재사용하여 간결하게 구현
"""

import torch
import warnings

# Project imports
from data.database_connector import DatabaseConnector
from scripts.train_common import (
    setup_cuda_optimizations,
    build_schema,
    build_dataloaders,
    build_model,
    create_optimizer_and_scheduler,
    train_one_epoch,
    validate,
    evaluate_comprehensive
)

# 경고 메시지 억제
warnings.filterwarnings('ignore')


# Optuna용 기본 설정 (빠른 평가)
OPTUNA_DEFAULTS = {
    # 데이터 설정 (작은 데이터셋)
    "batch_size": 256,
    "test_split": 0.2,
    "shuffle_seed": 42,
    "pair_limit": 1000000,  # 원본 5M → 1M (빠른 평가)

    # DataLoader 설정
    "num_workers": 0,
    "pin_memory": False,
    "streaming": False,
    "load_all_features": True,
    "chunk_size": 1000000,
    "feature_chunksize": 1000,
    "use_preprocessor": True,
    "test_mode": True,
    "prefetch_factor": 2,

    # 모델 아키텍처
    "categorical_embedding_dim": 32,
    "notice_dense_input_dim": 256,
    "company_dense_input_dim": 128,
    "tower_hidden_dims": [512, 256],
    "final_embedding_dim": 128,
    "dropout_rate": 0.1,
    "temperature": 1.0,
    "loss_type": "cross_entropy",
    "label_smoothing": 0.0,

    # 학습 설정 (짧은 학습)
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "num_epochs": 3,  # 원본 10+ → 3 (빠른 평가)
    "warmup_ratio": 0.05,
    "log_interval": 100,  # 원본 20 → 100 (로그 감소)

    # 모델 저장 비활성화
    "save_best": False,
    "save_final": False,

    # CUDA 최적화
    "enable_tf32": True,
    "enable_cudnn_benchmark": True,
    "enable_torch_compile": False,
    "compile_mode": "reduce-overhead",

    # 메타데이터
    "metadata_path": "meta/metadata.csv"
}


def train_for_optuna(
    # 튜닝 대상 하이퍼파라미터
    learning_rate=None,
    batch_size=None,
    categorical_embedding_dim=None,
    tower_hidden_dims=None,
    final_embedding_dim=None,
    dropout_rate=None,
    temperature=None,
    weight_decay=None,

    # Optuna용 고정 설정
    num_epochs=None,
    pair_limit=None,

    # 출력 제어
    verbose=False,

    # 추가 설정 (kwargs로 받음)
    **kwargs
):
    """
    Optuna 하이퍼파라미터 튜닝 전용 학습 함수

    Args:
        learning_rate: 학습률
        batch_size: 배치 크기
        categorical_embedding_dim: 범주형 임베딩 차원
        tower_hidden_dims: 타워 히든 레이어 차원 리스트
        final_embedding_dim: 최종 임베딩 차원
        dropout_rate: 드롭아웃 비율
        temperature: Temperature 파라미터
        weight_decay: 가중치 감쇠
        num_epochs: 에포크 수
        pair_limit: 데이터 페어 제한
        verbose: 로그 출력 여부
        **kwargs: 추가 설정

    Returns:
        dict: {'val_loss': float, 'val_acc': float, 'train_loss': float, ...}
    """
    # 기본 설정 복사
    config = OPTUNA_DEFAULTS.copy()

    # 사용자 지정 하이퍼파라미터로 덮어쓰기
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    if batch_size is not None:
        config["batch_size"] = batch_size
    if categorical_embedding_dim is not None:
        config["categorical_embedding_dim"] = categorical_embedding_dim
    if tower_hidden_dims is not None:
        config["tower_hidden_dims"] = tower_hidden_dims
    if final_embedding_dim is not None:
        config["final_embedding_dim"] = final_embedding_dim
    if dropout_rate is not None:
        config["dropout_rate"] = dropout_rate
    if temperature is not None:
        config["temperature"] = temperature
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    if num_epochs is not None:
        config["num_epochs"] = num_epochs
    if pair_limit is not None:
        config["pair_limit"] = pair_limit

    # 추가 kwargs 병합
    config.update(kwargs)

    if verbose:
        print("=== Optuna Two-Tower 모델 학습 시작 ===")
        print(f"🔧 설정된 하이퍼파라미터:")
        print(f"   - Pair Limit: {config['pair_limit']:,}")
        print(f"   - Batch Size: {config['batch_size']}")
        print(f"   - Embedding Dim: {config['categorical_embedding_dim']} → {config['final_embedding_dim']}")
        print(f"   - Hidden Dims: {config['tower_hidden_dims']}")
        print(f"   - Learning Rate: {config['learning_rate']}")
        print(f"   - Temperature: {config['temperature']}")
        print(f"   - Num Epochs: {config['num_epochs']}")

    # 1. CUDA 최적화 설정
    setup_cuda_optimizations(
        enable_tf32=config["enable_tf32"],
        enable_cudnn_benchmark=config["enable_cudnn_benchmark"]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"디바이스: {device}")

    # 2. 데이터베이스 연결
    if verbose:
        print("\n데이터베이스 연결 중...")
    db = DatabaseConnector()
    engine = db.engine

    # 3. 스키마 구축
    if verbose:
        print("스키마 구축 중...")
    schema = build_schema(metadata_path=config["metadata_path"])

    if verbose:
        print(f"Notice 피처: {len(schema.notice.categorical)}개 범주형, {len(schema.notice.numeric)}개 수치형")
        print(f"Company 피처: {len(schema.company.categorical)}개 범주형, {len(schema.company.numeric)}개 수치형")

    # 4. 데이터로더 생성
    if verbose:
        print("\n데이터로더 생성 중...")
    train_loader, test_loader = build_dataloaders(
        db_engine=engine,
        schema=schema,
        config=config
    )

    if verbose:
        print(f"Train 배치 수: {len(train_loader)}")
        print(f"Test 배치 수: {len(test_loader) if test_loader else 'None'}")

    # 5. 모델 생성
    if verbose:
        print("\n모델 생성 중...")
    train_task = build_model(schema, config, device)

    if verbose:
        total_params = sum(p.numel() for p in train_task.parameters())
        trainable_params = sum(p.numel() for p in train_task.parameters() if p.requires_grad)
        print(f"모델 파라미터: {total_params:,}개 (학습 가능: {trainable_params:,}개)")

    # 6. 옵티마이저 및 스케줄러 설정
    optimizer, scheduler = create_optimizer_and_scheduler(train_task, train_loader, config)

    if verbose:
        warmup_steps = max(1, int(len(train_loader) * config["warmup_ratio"]))
        print(f"Learning Rate Warmup: {warmup_steps} steps")

    # 7. 학습 루프
    if verbose:
        print("\n=== 학습 시작 ===")

    num_epochs = config["num_epochs"]
    avg_train_loss = 0.0
    avg_train_acc = 0.0
    avg_val_loss = 0.0
    avg_val_acc = 0.0

    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")

        # 학습
        avg_train_loss, avg_train_acc = train_one_epoch(
            train_task=train_task,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            verbose=verbose
        )

        if verbose:
            print(f"Train - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.3f}")

        # 검증
        avg_val_loss, avg_val_acc = validate(
            train_task=train_task,
            test_loader=test_loader,
            device=device,
            config=config,
            verbose=verbose
        )

        if avg_val_loss is not None and verbose:
            print(f"Val   - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.3f}")

    # 8. 최종 평가
    if verbose:
        print("\n=== 최종 평가 ===")

    test_metrics = {}
    if test_loader is not None:
        test_metrics = evaluate_comprehensive(
            train_task=train_task,
            test_loader=test_loader,
            device=device,
            verbose=verbose
        )

    # 9. 결과 정리 및 반환
    results = {
        "val_loss": avg_val_loss if avg_val_loss is not None else avg_train_loss,
        "val_acc": avg_val_acc if avg_val_acc is not None else avg_train_acc,
        "train_loss": avg_train_loss,
        "train_acc": avg_train_acc,
        "recall@5": test_metrics.get("recall@5", 0.0),
        "recall@10": test_metrics.get("recall@10", 0.0),
        "mrr": test_metrics.get("mrr", 0.0),
        "similarity_gap": test_metrics.get("similarity_gap", 0.0),
    }

    # 10. GPU 메모리 정리
    del train_task, optimizer, train_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print("\n=== 학습 완료 ===")
        print(f"최종 결과: {results}")

    return results


if __name__ == "__main__":
    """
    독립적으로 실행 가능한 테스트 코드
    """
    import sys

    # 간단한 테스트 실행
    print("=== train_for_optuna.py 테스트 실행 ===")

    # verbose 모드로 실행
    results = train_for_optuna(
        learning_rate=1e-3,
        num_epochs=2,  # 테스트용 2 에포크
        pair_limit=500000,  # 작은 데이터셋
        verbose=True
    )

    print(f"\n테스트 완료! 결과: {results}")
