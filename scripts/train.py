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

# Project imports
from data.database_connector import DatabaseConnector
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders
from src.towers.two_tower_train_task import create_two_tower_train_task
from src.evaluation.evaluator import TwoTowerEvaluator


def main():
    print("=== Two-Tower 모델 학습 시작 ===")
    
    # 1. 기본 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
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
    
    # 4. 데이터로더 생성
    print("\n데이터로더 생성 중...")
    train_loader, test_loader = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=256,           # 작은 배치 크기로 시작
        limit=None,              # 소량 데이터로 테스트
        test_split=0.2,
        shuffle_seed=42,
        num_workers=0,
        load_all_features=True,
        streaming=True,
        chunk_size=256*1000,
        feature_chunksize=1000,
        feature_limit=100000
    )
    
    print(f"Train 배치 수: {len(train_loader)}")
    print(f"Test 배치 수: {len(test_loader) if test_loader else 'None'}")
    
    # 5. 모델 생성
    print("\n모델 생성 중...")
    
    # 범주형 키 추출
    notice_categorical_keys = schema.notice.categorical
    company_categorical_keys = schema.company.categorical
    
    # TrainTask 생성
    train_task = create_two_tower_train_task(
        notice_categorical_keys=notice_categorical_keys,
        company_categorical_keys=company_categorical_keys,
        metadata_path="meta/metadata.csv",
        categorical_embedding_dim=32,      # 작은 임베딩 차원
        notice_dense_input_dim=256,
        company_dense_input_dim=128,
        tower_hidden_dims=[128, 64],       # 작은 히든 레이어
        final_embedding_dim=64,            # 작은 최종 차원
        dropout_rate=0.1,
        temperature=1.0,
        loss_type="cross_entropy",
        device=device
    )
    
    # 6. 옵티마이저 설정
    optimizer = optim.Adam(train_task.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 7. 모델 정보 출력
    total_params = sum(p.numel() for p in train_task.parameters())
    trainable_params = sum(p.numel() for p in train_task.parameters() if p.requires_grad)
    print(f"\n모델 파라미터: {total_params:,}개 (학습 가능: {trainable_params:,}개)")
    
    evaluator = TwoTowerEvaluator(device=device)
    
    # 8. 학습 루프
    print("\n=== 학습 시작 ===")
    num_epochs = 5
    best_val_loss = float('inf')
    output_dir = Path("outputs/models")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 학습
        train_task.train()
        train_losses = []
        train_accuracies = []
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(train_pbar):
            # Forward pass
            optimizer.zero_grad()
            result = train_task(batch, return_metrics=True)
            
            loss = result["loss"]
            accuracy = result["accuracy"]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # 메트릭 저장
            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())
            
            # Progress bar 업데이트
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy.item():.3f}'
            })
        
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
                val_pbar = tqdm(test_loader, desc="Validation")
                for batch in val_pbar:
                    result = train_task(batch, return_metrics=True)
                    
                    loss = result["loss"]
                    accuracy = result["accuracy"]
                    
                    val_losses.append(loss.item())
                    val_accuracies.append(accuracy.item())
                    
                    val_pbar.set_postfix({
                        'val_loss': f'{loss.item():.4f}',
                        'val_acc': f'{accuracy.item():.3f}'
                    })
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_acc = sum(val_accuracies) / len(val_accuracies)
            
            print(f"Val   - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.3f}")
        
        # 체크포인트 저장
        save_checkpoint(train_task, optimizer, epoch, avg_val_loss, output_dir)
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(train_task, optimizer, epoch, avg_val_loss, output_dir, is_best=True)
            print(f"새로운 최고 성능! Loss: {avg_val_loss:.4f}")
            
            
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

    
    # 9. 최종 모델 저장
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
    checkpoint_path = "outputs/checkpoint_epoch_3.pt"  # 예시
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