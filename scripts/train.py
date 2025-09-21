#!/usr/bin/env python3
"""
Simple Two-Tower Model Training Script
간단한 학습 실행 및 결과 확인
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import time
from pathlib import Path
import pandas as pd
from datetime import datetime
import os

# Project imports
from data.database_connector import DatabaseConnector
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders
from src.towers.two_tower_train_task import create_two_tower_train_task
from src.evaluation.evaluator import TwoTowerEvaluator


def save_training_results(hyperparams, metrics, output_file="train_results.csv"):
    """
    학습 결과를 CSV 파일에 기록

    Args:
        hyperparams: 하이퍼파라미터 딕셔너리
        metrics: 성능 지표 딕셔너리
        output_file: 출력 CSV 파일명
    """
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 결과 딕셔너리 구성
    result_row = {
        "timestamp": timestamp,
        "batch_size": hyperparams.get("batch_size", "N/A"),
        "model_params": hyperparams.get("model_params", "N/A"),
        "embedding_dim": hyperparams.get("embedding_dim", "N/A"),
        "final_embedding_dim": hyperparams.get("final_embedding_dim", "N/A"),
        "hidden_dims": str(hyperparams.get("hidden_dims", "N/A")),
        "learning_rate": hyperparams.get("learning_rate", "N/A"),
        "epochs": hyperparams.get("epochs", "N/A"),
        "train_loss": metrics.get("train_loss", "N/A"),
        "train_acc": metrics.get("train_acc", "N/A"),
        "val_loss": metrics.get("val_loss", "N/A"),
        "val_acc": metrics.get("val_acc", "N/A"),
        "recall_at_5": metrics.get("recall_at_5", "N/A"),
        "recall_at_10": metrics.get("recall_at_10", "N/A"),
        "mrr": metrics.get("mrr", "N/A"),
        "similarity_gap": metrics.get("similarity_gap", "N/A"),
        "train_batches": hyperparams.get("train_batches", "N/A"),
        "test_batches": hyperparams.get("test_batches", "N/A"),
        "gpu_optimization": hyperparams.get("gpu_optimization", "N/A"),
    }

    # 결과를 DataFrame으로 변환
    new_row_df = pd.DataFrame([result_row])

    # CSV 파일 존재 확인
    if os.path.exists(output_file):
        # 기존 파일에 추가
        existing_df = pd.read_csv(output_file)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        print(f"기존 결과에 새 행 추가: {output_file}")
    else:
        # 새 파일 생성
        updated_df = new_row_df
        print(f"새 결과 파일 생성: {output_file}")

    # CSV 파일 저장
    updated_df.to_csv(output_file, index=False)
    print(f"학습 결과 저장 완료: {len(updated_df)}행")


def main():
    print("=== Two-Tower 모델 학습 시작 ===")

    # ===========================================
    # 하이퍼파라미터 설정 (중앙 관리)
    # ===========================================
    config = {
        # 데이터 설정
        "batch_size": 256,
        "test_split": 0.2,
        "shuffle_seed": 42,
        "pair_limit": 1000000,

        # DataLoader 설정
        "num_workers": 0,
        "pin_memory": False,
        "streaming": False,
        "load_all_features": True,
        "chunk_size": 1000000,
        "feature_chunksize": 1000,
        "use_preprocessor": True,
        "test_mode": True,          # 빠진 중요 하이퍼파라미터!
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

        # 학습 설정
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "num_epochs": 1,
        "warmup_ratio": 0.05,
        "log_interval": 20,

        # 모델 저장/로딩
        "output_dir": "output/models",
        "save_best": True,
        "save_final": True,

        # CUDA 최적화
        "enable_tf32": True,
        "enable_cudnn_benchmark": True,
        "enable_torch_compile": False,  # 현재 주석처리됨
        "compile_mode": "reduce-overhead",

        # 시스템 설정
        "gpu_optimization": "GPU Collate + CUDA Streams + TF32/cuDNN",
        "metadata_path": "meta/metadata.csv"
    }

    print(f"🔧 설정된 하이퍼파라미터:")
    print(f"   - Test Mode: {config['test_mode']} (Pair Limit: {config['pair_limit']:,})")
    print(f"   - Batch Size: {config['batch_size']}")
    print(f"   - Embedding Dim: {config['categorical_embedding_dim']} → {config['final_embedding_dim']}")
    print(f"   - Hidden Dims: {config['tower_hidden_dims']}")
    print(f"   - Learning Rate: {config['learning_rate']}")
    print(f"   - GPU Optimization: {config['enable_torch_compile']}")
    print(f"   - Temperature: {config['temperature']}")

    # CUDA 가속 최적화 (config 기반)
    if config["enable_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True         # TF32 허용 (Ampere+)
        torch.set_float32_matmul_precision("high")           # cublas 고정밀 속도 모드
    if config["enable_cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True                # Conv/Norm autotune (MLP에도 도움)

    # 1. 기본 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    print("CUDA 가속 최적화 활성화: TF32, cuDNN benchmark, high precision matmul")
    
    # 2. 데이터베이스 연결
    print("\n데이터베이스 연결 중...")
    db = DatabaseConnector()
    engine = db.engine
    
    # 3. 스키마 구축
    print("스키마 구축 중...")
    schema_config = {
        "notice_table": "notice",
        "company_table": "company",
        "pair_table": "bid_two_tower",
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)
    
    print(f"Notice 피처: {len(schema.notice.categorical)}개 범주형, {len(schema.notice.numeric)}개 수치형")
    print(f"Company 피처: {len(schema.company.categorical)}개 범주형, {len(schema.company.numeric)}개 수치형")
    
    # 4. 데이터로더 생성 (Test Mode + GPU 최적화)
    print("\n데이터로더 생성 중... (Test Mode + GPU Collate 최적화)")
    train_loader, test_loader = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=config["batch_size"],
        test_split=config["test_split"],
        shuffle_seed=config["shuffle_seed"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        streaming=config["streaming"],
        load_all_features=config["load_all_features"],
        chunk_size=config["chunk_size"],
        feature_chunksize=config["feature_chunksize"],
        use_preprocessor=config["use_preprocessor"],
        test_mode=config["test_mode"],
        pair_limit=config["pair_limit"],
    )
    
    print(f"Train 배치 수: {len(train_loader)}")
    print(f"Test 배치 수: {len(test_loader) if test_loader else 'None'}")
    
    # 5. 모델 생성
    print("\n모델 생성 중...")
    
    # 범주형 키 추출
    notice_categorical_keys = schema.notice.categorical
    company_categorical_keys = schema.company.categorical
    
    # TrainTask 생성 (config 사용)
    train_task = create_two_tower_train_task(
        notice_categorical_keys=notice_categorical_keys,
        company_categorical_keys=company_categorical_keys,
        metadata_path=config["metadata_path"],
        categorical_embedding_dim=config["categorical_embedding_dim"],
        notice_dense_input_dim=config["notice_dense_input_dim"],
        company_dense_input_dim=config["company_dense_input_dim"],
        tower_hidden_dims=config["tower_hidden_dims"],
        final_embedding_dim=config["final_embedding_dim"],
        dropout_rate=config["dropout_rate"],
        temperature=config["temperature"],
        loss_type=config["loss_type"],
        device=device
    )

    # torch.compile 최적화 (config 기반)
    if config["enable_torch_compile"]:
        print("torch.compile 최적화 적용 중...")
        train_task = torch.compile(train_task, mode=config["compile_mode"], fullgraph=False)
        print(f"torch.compile 적용 완료 ({config['compile_mode']} 모드)")
    else:
        print("torch.compile 비활성화됨")
    
    # 6. 옵티마이저 및 스케줄러 설정 (config 사용)
    optimizer = optim.Adam(train_task.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Learning Rate Warmup 스케줄러 (config warmup_ratio 사용)
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = max(1, int(len(train_loader) * config["warmup_ratio"]))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
    # 7. 모델 정보 출력
    total_params = sum(p.numel() for p in train_task.parameters())
    trainable_params = sum(p.numel() for p in train_task.parameters() if p.requires_grad)
    print(f"\n모델 파라미터: {total_params:,}개 (학습 가능: {trainable_params:,}개)")
    print(f"Learning Rate Warmup: {warmup_steps} steps ({warmup_steps/len(train_loader)*100:.1f}% of epoch)")
    
    evaluator = TwoTowerEvaluator(device=device)
    
    # 8. GPU 최적화 설정
    print("\n=== GPU 최적화 모드 (GPU Collate + 비동기 전송) ===")
    print(f"  - GPU Collate: KJT를 GPU에서 직접 생성")
    print(f"  - Pin memory: False (GPU 직접 생성)")
    print(f"  - 비동기 H2D 전송 준비")

    # CUDA 스트림 설정 (비동기 전송용)
    prefetch_stream = torch.cuda.Stream()

    def _to_device_async(batch, device):
        """비동기 GPU 전송 (필요한 경우만)"""
        # 이미 GPU라면 바로 리턴
        if batch["notice"]["dense"].is_cuda and batch["company"]["dense"].is_cuda:
            return batch
        with torch.cuda.stream(prefetch_stream):
            batch["notice"]["dense"] = batch["notice"]["dense"].to(device, non_blocking=True)
            batch["company"]["dense"] = batch["company"]["dense"].to(device, non_blocking=True)
            if hasattr(batch["notice"]["kjt"], "to"):
                batch["notice"]["kjt"] = batch["notice"]["kjt"].to(device)
            if hasattr(batch["company"]["kjt"], "to"):
                batch["company"]["kjt"] = batch["company"]["kjt"].to(device)
        return batch

    # 9. 학습 루프 (config 사용)
    print("\n=== 학습 시작 (GPU 최적화) ===")
    num_epochs = config["num_epochs"]
    best_val_loss = float('inf')
    output_dir = Path(config["output_dir"])

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 학습
        train_task.train()
        train_losses = []
        train_accuracies = []
        
        # 이중 라인 tqdm 설정 (윗줄: 진행바, 아랫줄: 상세 정보)
        train_pbar = tqdm(
            train_loader,
            desc="Training",
            unit="batch",
            position=0,  # 0번째 줄 (윗줄)
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}',
            ncols=100,
            mininterval=0.5,
            leave=False  # 완료 후 진행바 제거
        )

        # 상세 정보용 tqdm (아랫줄)
        info_bar = tqdm(
            total=0,
            position=1,  # 1번째 줄 (아랫줄)
            bar_format='{desc}',
            ncols=100,
            leave=False
        )
        
        step_count = 0
        log_interval = config["log_interval"]  # config에서 로그 간격 설정

        for batch in train_pbar:
            # 비동기 GPU 전송 (필요한 경우만)
            batch = _to_device_async(batch, device)
            torch.cuda.current_stream().wait_stream(prefetch_stream)  # 안전 동기화

            # Forward pass
            optimizer.zero_grad()
            result = train_task(batch, return_metrics=True)

            loss = result["loss"]
            accuracy = result["accuracy"]

            # Backward pass
            loss.backward()
            optimizer.step()

            # Scheduler step (optimizer.step() 이후에 호출)
            scheduler.step()

            step_count += 1

            # 메트릭 저장 (매 스텝)
            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())

            # Progress bar 업데이트 (주기적으로만 - GPU 동기화 감소)
            if step_count % log_interval == 0:
                # Two-Tower 핵심 지표 5개 추출
                loss_val = loss.item()
                accuracy_val = accuracy.item()
                pos_sim = result.get("positive_similarity_mean", torch.tensor(0.0)).item()
                neg_sim = result.get("negative_similarity_mean", torch.tensor(0.0)).item()
                sim_gap = result.get("similarity_gap", torch.tensor(0.0)).item()

                # Z-gap 계산 (similarity_gap을 표준편차로 정규화한 지표)
                z_gap = sim_gap / max(abs(neg_sim) + 1e-8, 1e-8)  # 0 division 방지

                # 아랫줄에 Two-Tower 핵심 지표 표시
                info_str = f"📊 Loss: {loss_val:.3f} | Acc: {accuracy_val:.3f} | Pos: {pos_sim:.3f} | Neg: {neg_sim:.3f} | Z-gap: {z_gap:.2f} | Batch: {config['batch_size']}"
                info_bar.set_description_str(info_str)
            
        
        # 진행바 종료
        train_pbar.close()
        info_bar.close()

        # 에포크 결과
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)

        print(f"Train - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.3f}")
        
        # 검증 (있다면)
        avg_val_loss = avg_train_loss  # 기본값
        if test_loader is not None:
            train_task.eval()
            val_losses = []
            val_accuracies = []
            
            with torch.no_grad():
                # Validation용 이중 라인 설정
                val_pbar = tqdm(
                    test_loader,
                    desc="Validation",
                    unit="batch",
                    position=0,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}',
                    ncols=100,
                    mininterval=0.5,
                    leave=False
                )

                val_info_bar = tqdm(
                    total=0,
                    position=1,
                    bar_format='{desc}',
                    ncols=100,
                    leave=False
                )

                for batch in val_pbar:
                    result = train_task(batch, return_metrics=True)
                    
                    loss = result["loss"]
                    accuracy = result["accuracy"]
                    
                    val_losses.append(loss.item())
                    val_accuracies.append(accuracy.item())

                    # Validation Two-Tower 핵심 지표 5개
                    val_loss = loss.item()
                    val_acc = accuracy.item()
                    val_pos_sim = result.get("positive_similarity_mean", torch.tensor(0.0)).item()
                    val_neg_sim = result.get("negative_similarity_mean", torch.tensor(0.0)).item()
                    val_sim_gap = result.get("similarity_gap", torch.tensor(0.0)).item()

                    # Validation Z-gap 계산
                    val_z_gap = val_sim_gap / max(abs(val_neg_sim) + 1e-8, 1e-8)

                    # Validation 정보 아랫줄에 표시
                    info_str = f"🔍 Loss: {val_loss:.3f} | Acc: {val_acc:.3f} | Pos: {val_pos_sim:.3f} | Neg: {val_neg_sim:.3f} | Z-gap: {val_z_gap:.2f}"
                    val_info_bar.set_description_str(info_str)

                # Validation 진행바 종료
                val_pbar.close()
                val_info_bar.close()
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_acc = sum(val_accuracies) / len(val_accuracies)
            
            print(f"Val   - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.3f}")
        
        # 체크포인트 저장 (config 기반)
        save_checkpoint(train_task, optimizer, epoch, avg_val_loss, output_dir)

        # 최고 성능 모델 저장
        if config["save_best"] and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(train_task, optimizer, epoch, avg_val_loss, output_dir, is_best=True)
            print(f"새로운 최고 성능! Loss: {avg_val_loss:.4f}")
            
            
    # 학습 완료
    print("\n=== 학습 완료 ===")
    print("순차 처리 방식으로 학습 완료!")
            
    print("\n=== 최종 평가 및 추론 테스트 ===")
    
    # 테스트 데이터 종합 평가
    if test_loader is not None:
        print("테스트 데이터 종합 평가:")
        test_metrics = evaluator.evaluate_comprehensive(train_task, test_loader, verbose=True)
    else:
        print("훈련 데이터 샘플로 평가:")
        sample_batch = next(iter(train_loader))
        sample_metrics = evaluator.evaluate_single_batch(train_task, sample_batch, verbose=True)
    
    # 추론 시연
    test_batch = next(iter(train_loader))
    evaluator.demonstrate_predictions(train_task, test_batch, top_k=10)

    # 학습 결과 CSV 저장
    print("\n=== 학습 결과 기록 ===")

    # 하이퍼파라미터 딕셔너리 구성 (config 기반)
    hyperparams = {
        "batch_size": config["batch_size"],
        "model_params": total_params,
        "embedding_dim": config["categorical_embedding_dim"],
        "final_embedding_dim": config["final_embedding_dim"],
        "hidden_dims": config["tower_hidden_dims"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "dropout_rate": config["dropout_rate"],
        "temperature": config["temperature"],
        "epochs": config["num_epochs"],
        "train_batches": len(train_loader),
        "test_batches": len(test_loader) if test_loader else 0,
        "gpu_optimization": config["gpu_optimization"]
    }

    # 성능 지표 딕셔너리 구성 (최종 평가 결과 사용)
    final_metrics = {
        "train_loss": avg_train_loss,
        "train_acc": avg_train_acc,
        "val_loss": avg_val_loss if test_loader else "N/A",
        "val_acc": avg_val_acc if test_loader else "N/A",
        "recall@5": test_metrics.get("recall@5", "N/A") if test_loader else "N/A",
        "recall@10": test_metrics.get("recall@10", "N/A") if test_loader else "N/A",
        "mrr": test_metrics.get("mrr", "N/A") if test_loader else "N/A",
        "similarity_gap": test_metrics.get("similarity_gap", "N/A") if test_loader else "N/A"
    }

    # CSV 저장
    save_training_results(hyperparams, final_metrics)

    # 9. 최종 모델 저장 (config 기반)
    if config["save_final"]:
        save_checkpoint(train_task, optimizer, num_epochs-1, 0.0, output_dir, is_final=True)
        print(f"\n최종 모델 저장: {output_dir}")

    print("\n=== 학습 완료 ===")


def save_checkpoint(model, optimizer, epoch, loss, save_dir, is_best=False, is_final=False):
    """체크포인트 저장"""
    import os
    from pathlib import Path
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 체크포인트 데이터
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # 일반 체크포인트 저장
    if not is_final:
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"체크포인트 저장: {checkpoint_path}")
    
    # 최고 성능 모델 저장
    if is_best:
        best_path = save_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"최고 성능 모델 저장: {best_path}")
    
    # 최종 모델 저장 (inference용)
    if is_final:
        final_path = save_dir / 'final_model.pt'
        torch.save(checkpoint, final_path)
        
        # 모델만 따로 저장 (추론용)
        model_only_path = save_dir / 'model_weights.pt'
        torch.save(model.state_dict(), model_only_path)
        print(f"최종 모델 저장: {final_path}")
        print(f"모델 가중치 저장: {model_only_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """체크포인트 불러오기"""
    print(f"체크포인트 불러오는 중: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    last_loss = checkpoint['loss']
    
    print(f"Epoch {checkpoint['epoch']+1}부터 이어서 학습 시작")
    print(f"이전 손실: {last_loss:.4f}")
    
    return start_epoch, last_loss


def resume_training_example():
    """이어학습 예제"""
    print("=== 이어학습 모드 ===")
    
    # 기본 설정은 main()과 동일...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 데이터로더 생성 (main()과 동일)
    # ... 생략 ...
    
    # 모델 생성 (main()과 동일) 
    # ... 생략 ...
    
    # 체크포인트 불러오기
    checkpoint_path = "output/checkpoint_epoch_3.pt"  # 예시
    if Path(checkpoint_path).exists():
        start_epoch, last_loss = load_checkpoint(train_task, optimizer, checkpoint_path)
    else:
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return
    
    # 남은 에포크 학습
    num_epochs = 10  # 전체 목표 에포크
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} (이어학습)")
        # ... 학습 코드 ...


if __name__ == "__main__":
    import sys
    
    # 이어학습 모드인지 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        resume_training_example()
    else:
        main()