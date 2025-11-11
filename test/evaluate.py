#!/usr/bin/env python3
"""
Two-Tower Model Evaluation Script
저장된 모델 체크포인트를 불러와서 성능 평가만 수행하는 스크립트
"""

import torch
import argparse
import json
from pathlib import Path
from datetime import datetime

# Project imports
from data.database_connector import DatabaseConnector
from preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders
from src.towers.two_tower_train_task import create_two_tower_train_task
from src.evaluation.evaluator import TwoTowerEvaluator


def load_config(config_path):
    """저장된 설정 파일 로드"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="Two-Tower 모델 평가 스크립트")

    # 필수 인자
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="평가할 모델 체크포인트 경로 (예: checkpoints/best_model.pt)")

    # 선택 인자
    parser.add_argument("--config", type=str, default=None,
                        help="모델 설정 파일 경로 (기본값: 체크포인트와 같은 디렉토리의 config.json)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="평가 배치 크기 (설정파일 값을 오버라이드)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="사용할 디바이스 (cuda 또는 cpu)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="예측 시연에 사용할 top-k 값")
    parser.add_argument("--no_demo", action="store_true",
                        help="예측 시연 생략")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="테스트 데이터 비율")
    parser.add_argument("--pair_limit", type=lambda x: None if x.lower() == 'none' else int(x),
                        default=None,
                        help="평가에 사용할 최대 페어 수 (none이면 전체 데이터)")
    parser.add_argument("--metrics_only", action="store_true",
                        help="저장된 메트릭만 표시하고 재평가 생략 (데이터 로딩 없음)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Two-Tower Model Evaluation")
    print("=" * 80)
    print(f"체크포인트: {args.checkpoint}")
    print(f"디바이스: {args.device}")
    if args.metrics_only:
        print(f"모드: 메트릭만 표시 (재평가 생략)")
    print()

    # metrics_only 모드: 체크포인트에서 메트릭만 로드하고 종료
    if args.metrics_only:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        print("=" * 80)
        print("저장된 메트릭 정보")
        print("=" * 80)
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Loss: {checkpoint.get('loss', 'N/A'):.4f}" if isinstance(checkpoint.get('loss'), float) else f"Loss: {checkpoint.get('loss', 'N/A')}")
        print()

        metrics = checkpoint.get('metrics', None)
        if metrics is not None:
            print("📊 평가 메트릭:")
            print()

            # 기본 메트릭
            if 'train_loss' in metrics:
                print(f"  [훈련]")
                print(f"    Loss:     {metrics['train_loss']:.4f}")
                print(f"    Accuracy: {metrics.get('train_acc', 0):.4f}")
                print()

            if 'val_loss' in metrics:
                print(f"  [검증]")
                print(f"    Loss:     {metrics['val_loss']:.4f}")
                print(f"    Accuracy: {metrics.get('val_acc', 0):.4f}")
                print()

            # 검색 메트릭
            if 'recall_at_5' in metrics or 'mrr' in metrics:
                print(f"  [테스트 - 검색 성능]")
                if 'recall_at_5' in metrics:
                    print(f"    Recall@5:  {metrics['recall_at_5']:.4f}")
                if 'recall_at_10' in metrics:
                    print(f"    Recall@10: {metrics['recall_at_10']:.4f}")
                if 'mrr' in metrics:
                    print(f"    MRR:       {metrics['mrr']:.4f}")
                if 'accuracy' in metrics:
                    print(f"    Top-1 Acc: {metrics['accuracy']:.4f}")
                if 'similarity_gap' in metrics:
                    print(f"    Sim Gap:   {metrics['similarity_gap']:.4f}")
                print()
        else:
            print("⚠️  이 체크포인트에는 평가 메트릭이 저장되어 있지 않습니다.")
            print("   새로 학습하면 메트릭이 자동으로 저장됩니다.")
            print()

        print("=" * 80)
        return  # 메트릭만 표시하고 종료

    # 1. 설정 파일 로드
    if args.config is None:
        # 체크포인트와 같은 디렉토리의 config.json 사용
        checkpoint_dir = Path(args.checkpoint).parent
        config_path = checkpoint_dir / "config.json"
    else:
        config_path = Path(args.config)

    print(f"설정 파일 로드: {config_path}")
    config = load_config(config_path)
    print(f"✓ 설정 파일 로드 완료")
    print()

    # 배치 크기 오버라이드
    if args.batch_size is not None:
        print(f"배치 크기 오버라이드: {config.get('batch_size')} -> {args.batch_size}")
        config['batch_size'] = args.batch_size

    # 2. 데이터베이스 연결 및 스키마 로드
    print("데이터베이스 연결 중...")
    db_connector = DatabaseConnector()
    engine = db_connector.engine

    # 3. 스키마 구축
    print("스키마 구축 중...")
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": config.get("metadata_path", "meta/metadata.csv")
    }
    schema = build_torchrec_schema_from_meta(**schema_config)
    print(f"✓ Notice 피처: {len(schema.notice.categorical)}개 범주형, {len(schema.notice.numeric)}개 수치형")
    print(f"✓ Company 피처: {len(schema.company.categorical)}개 범주형, {len(schema.company.numeric)}개 수치형")
    print()

    # 4. 데이터 로더 생성
    print("데이터 로더 생성 중...")
    train_loader, test_loader = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=config['batch_size'],
        test_split=args.test_split,
        shuffle_seed=config.get('shuffle_seed', 42),
        pair_limit=args.pair_limit,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        shuffle=config.get('shuffle', False),
        streaming=config.get('streaming', False),
        test_mode=config.get('test_mode', True)
    )
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Test loader: {len(test_loader)} batches")
    print()

    # 5. 모델 생성
    print("모델 초기화 중...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_task = create_two_tower_train_task(
        schema=schema,
        categorical_embedding_dim=config['categorical_embedding_dim'],
        final_embedding_dim=config['final_embedding_dim'],
        hidden_dims=config.get('tower_hidden_dims', [256, 128]),
        temperature=config.get('temperature', 0.07),
        device=device
    )
    print(f"✓ 모델 생성 완료 (device: {device})")
    print()

    # 6. 체크포인트 로드
    print(f"체크포인트 로드: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_task.load_state_dict(checkpoint['model_state_dict'])
    train_task.eval()  # 평가 모드로 전환

    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'loss': checkpoint.get('loss', 'N/A'),
        'metrics': checkpoint.get('metrics', None)
    }
    print(f"✓ 체크포인트 로드 완료")
    print(f"  - Epoch: {checkpoint_info['epoch']}")
    print(f"  - Loss: {checkpoint_info['loss']:.4f}" if isinstance(checkpoint_info['loss'], float) else f"  - Loss: {checkpoint_info['loss']}")

    # 저장된 메트릭 표시
    if checkpoint_info['metrics'] is not None:
        print(f"\n📊 저장된 평가 메트릭:")
        metrics = checkpoint_info['metrics']

        # 기본 메트릭
        if 'train_loss' in metrics:
            print(f"  [Train] Loss: {metrics['train_loss']:.4f}, Acc: {metrics.get('train_acc', 0):.4f}")
        if 'val_loss' in metrics:
            print(f"  [Val]   Loss: {metrics['val_loss']:.4f}, Acc: {metrics.get('val_acc', 0):.4f}")

        # 검색 메트릭
        if 'recall_at_5' in metrics:
            print(f"  [Test]  Recall@5: {metrics['recall_at_5']:.4f}, Recall@10: {metrics.get('recall_at_10', 0):.4f}")
        if 'mrr' in metrics:
            print(f"          MRR: {metrics['mrr']:.4f}, Sim Gap: {metrics.get('similarity_gap', 0):.4f}")
    else:
        print(f"\n⚠️  이 체크포인트에는 평가 메트릭이 저장되어 있지 않습니다.")
        print(f"   새로 학습하면 메트릭이 자동으로 저장됩니다.")

    print()

    # 7. 평가 수행
    print("=" * 80)
    print("평가 시작")
    print("=" * 80)
    print()

    evaluator = TwoTowerEvaluator(device=device)

    # 전체 데이터셋 평가
    print("📊 Test Set 평가 중...")
    print("-" * 80)
    test_metrics = evaluator.evaluate_comprehensive(
        train_task,
        test_loader,
        verbose=True
    )
    print()

    # 8. 예측 시연 (옵션)
    if not args.no_demo and test_loader:
        print("=" * 80)
        print("예측 시연 (첫 번째 배치)")
        print("=" * 80)
        print()

        test_batch = next(iter(test_loader))
        evaluator.demonstrate_predictions(
            train_task,
            test_batch,
            top_k=args.top_k
        )
        print()

    # 9. 결과 요약
    print("=" * 80)
    print("평가 완료")
    print("=" * 80)
    print()
    print("📈 최종 성능 지표:")
    print(f"  - Recall@5:  {test_metrics.get('recall_at_5', 0):.4f}")
    print(f"  - Recall@10: {test_metrics.get('recall_at_10', 0):.4f}")
    print(f"  - MRR:       {test_metrics.get('mrr', 0):.4f}")
    print(f"  - Top-1 Acc: {test_metrics.get('accuracy', 0):.4f}")
    print(f"  - Sim Gap:   {test_metrics.get('similarity_gap', 0):.4f}")
    print()

    # 10. 결과 저장 (선택)
    output_file = Path(args.checkpoint).parent / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {
        'checkpoint': str(args.checkpoint),
        'config': config,
        'checkpoint_info': checkpoint_info,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 평가 결과 저장: {output_file}")
    print()


if __name__ == "__main__":
    main()
