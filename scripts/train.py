#!/usr/bin/env python3
"""
Two-Tower Model Training Script - V2 (EmbeddingBagCollection)

GPU 최적화 파이프라인:
- PairLoaderV2: ID-only DataLoader (H2D 최소화)
- EmbeddingBagCollection: GPU 최적화 임베딩
- GPU gather + KJT 생성
- torch.compile + AMP + TF32
"""

import warnings
import os

# LR scheduler warning 무시 (매 batch마다 step을 호출하므로 의도된 동작)
warnings.filterwarnings('ignore', message='Detected call of.*lr_scheduler.step.*before.*optimizer.step')

# torch.fx tracing warning 무시 (torch.compile 시 발생)
warnings.filterwarnings('ignore', category=UserWarning, module='torch.fx._symbolic_trace')

# torch 내부 logger warning 억제
os.environ['TORCH_LOGS'] = '-torch.fx._symbolic_trace'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
import time
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse
import json
from collections import deque

# Project imports
from database.database_connector import DatabaseConnector
from preprocess.torchrec.schema import build_torchrec_schema_from_meta
from src.towers.pairs.pair_loader import create_pair_dataloaders
from src.towers.tower import NoticeTower, CompanyTower
from src.towers.kjt_utils import create_kjt_from_batch_gpu
from src.towers.embedding_config import get_notice_feature_metadata, get_company_feature_metadata
from src.evaluation.evaluator import TwoTowerEvaluator


def save_training_results(hyperparams, metrics, output_file="train_results.csv"):
    """학습 결과를 CSV 파일에 기록"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        "train_pos_sim": metrics.get("train_pos_sim", "N/A"),
        "train_sim_gap": metrics.get("train_sim_gap", "N/A"),
        "val_loss": metrics.get("val_loss", "N/A"),
        "val_acc": metrics.get("val_acc", "N/A"),
        "val_pos_sim": metrics.get("val_pos_sim", "N/A"),
        "val_sim_gap": metrics.get("val_sim_gap", "N/A"),
        "recall_at_5": metrics.get("recall_at_5", "N/A"),
        "recall_at_10": metrics.get("recall_at_10", "N/A"),
        "mrr": metrics.get("mrr", "N/A"),
        "train_batches": hyperparams.get("train_batches", "N/A"),
        "test_batches": hyperparams.get("test_batches", "N/A"),
        "gpu_optimization": hyperparams.get("gpu_optimization", "N/A"),
        "throughput_batch_s": metrics.get("throughput_batch_s", "N/A"),
        "throughput_sample_s": metrics.get("throughput_sample_s", "N/A"),
    }

    new_row_df = pd.DataFrame([result_row])

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        print(f"   기존 결과에 새 행 추가: {output_file}")
    else:
        updated_df = new_row_df
        print(f"   새 결과 파일 생성: {output_file}")

    updated_df.to_csv(output_file, index=False)
    print(f"   학습 결과 저장 완료: {len(updated_df)}번째 행")


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="Two-Tower V2 모델 학습 스크립트 (EmbeddingBagCollection)")

    # 데이터 설정
    parser.add_argument("--batch_size", type=int, default=256, help="배치 크기")
    parser.add_argument("--test_split", type=float, default=0.2, help="테스트 데이터 비율")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--pair_limit", type=lambda x: None if x.lower() == 'none' else int(x),
                        default=None, help="학습에 사용할 최대 페어 수 (none이면 전체 데이터 사용)")

    # DataLoader 설정
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory 사용")
    parser.add_argument("--shuffle", action="store_true", default=False, help="데이터 셔플 활성화")
    parser.add_argument("--streaming", action="store_true", default=False, help="스트리밍 모드 사용")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="스트리밍 시 pair chunk 크기")
    parser.add_argument("--feature_chunksize", type=int, default=100000, help="피처 로딩 시 chunk 크기")
    parser.add_argument("--test_mode", action="store_true", default=False, help="Test mode: pair_limit에 해당하는 feature만 로딩")
    parser.add_argument("--use_parquet", action="store_true", default=False, help="DB 대신 Parquet 파일 사용")
    parser.add_argument("--parquet_dir", type=str, default="data/parquet", help="Parquet 파일 디렉토리")

    # 모델 아키텍처
    parser.add_argument("--categorical_embedding_dim", type=int, default=32, help="범주형 임베딩 차원")
    parser.add_argument("--final_embedding_dim", type=int, default=128, help="최종 임베딩 차원")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="드롭아웃 비율")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")

    # 학습 설정
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="가중치 감쇠")
    parser.add_argument("--num_epochs", type=int, default=3, help="학습 에포크 수")
    parser.add_argument("--log_interval", type=int, default=10, help="로그 출력 간격")
    parser.add_argument("--resume", type=str, default=None, help="이어학습할 체크포인트 경로")

    # 모델 저장/로딩
    parser.add_argument("--output_dir", type=str, default="output/models", help="모델 저장 디렉토리")
    parser.add_argument("--save_best", action="store_true", default=True, help="최고 성능 모델 저장")
    parser.add_argument("--save_final", action="store_true", default=True, help="최종 모델 저장")

    # CUDA 최적화
    parser.add_argument("--enable_tf32", action="store_true", default=True, help="TF32 활성화")
    parser.add_argument("--enable_cudnn_benchmark", action="store_true", default=True, help="cuDNN benchmark 활성화")
    parser.add_argument("--enable_torch_compile", action="store_true", default=True, help="torch.compile 활성화")
    parser.add_argument("--enable_amp", action="store_true", default=True, help="AMP (Mixed Precision) 활성화")

    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, loss, save_dir, is_best=False, is_final=False, metrics=None, preprocessor=None, scheduler=None):
    """체크포인트 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Scheduler 상태 저장
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    if not is_final:
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"   체크포인트 저장: {checkpoint_path}")

    if is_best:
        best_path = save_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"   최고 성능 모델 저장: {best_path}")

    if is_final:
        final_path = save_dir / 'final_model.pt'
        torch.save(checkpoint, final_path)
        model_only_path = save_dir / 'model_weights.pt'
        torch.save(model.state_dict(), model_only_path)
        print(f"최종 모델 저장: {final_path}")
        print(f"모델 가중치 저장: {model_only_path}")

        # Preprocessor 저장
        if preprocessor is not None:
            preprocessor_path = save_dir / 'preprocessor.pt'
            torch.save(preprocessor, preprocessor_path)
            print(f"전처리기 저장: {preprocessor_path}")


def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None, config_lr=None):
    """체크포인트 불러오기"""
    print(f"체크포인트 불러오는 중: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Scheduler 상태 복원
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Scheduler 상태 복원 완료 (last_epoch: {checkpoint['scheduler_state_dict'].get('last_epoch', 'N/A')})")
    elif scheduler is not None:
        print(f"⚠️ 체크포인트에 scheduler 상태가 없습니다.")

        # Optimizer의 lr이 0이면 config의 lr로 복원
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr == 0.0 and config_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config_lr
            print(f"   ⚠️ Optimizer lr이 0이었습니다. config lr ({config_lr})로 복원")

        # Scheduler의 base_lrs와 last_epoch 설정
        # 새로운 scheduler가 이어서 동작하도록 설정
        start_epoch = checkpoint['epoch'] + 1

        # Scheduler의 last_epoch를 설정하여 이어서 진행
        # 주의: scheduler.step()이 호출될 때마다 last_epoch가 증가
        # 새로운 T_max 기준으로 진행 비율 계산
        if hasattr(scheduler, 'T_max'):
            # 기존 진행된 step 수 추정 (epoch * batches_per_epoch)
            # 여기서는 scheduler를 새로 시작하되, lr을 적절히 설정
            print(f"   새로운 Scheduler 사용 (T_max: {scheduler.T_max})")

    start_epoch = checkpoint['epoch'] + 1
    last_loss = checkpoint['loss']
    print(f"Epoch {checkpoint['epoch']+1}부터 이어서 학습 시작")
    print(f"이전 손실: {last_loss:.4f}")

    # 현재 lr 출력
    current_lr = optimizer.param_groups[0]['lr']
    print(f"현재 Learning Rate: {current_lr}")

    return start_epoch, last_loss


def contrastive_loss(notice_emb, company_emb, temperature=0.07):
    """Simple InfoNCE loss (in-batch negatives)"""
    sim_matrix = torch.mm(notice_emb, company_emb.t()) / temperature
    labels = torch.arange(len(notice_emb), device=notice_emb.device)
    return torch.nn.functional.cross_entropy(sim_matrix, labels)


def calculate_metrics(notice_emb, company_emb, temperature=0.07):
    """
    학습/검증 메트릭 계산

    Returns:
        dict: {
            'loss': float,
            'accuracy': float,
            'pos_sim': float (positive similarity),
            'sim_gap': float (similarity gap)
        }
    """
    # Similarity matrix
    sim_matrix = torch.mm(notice_emb, company_emb.t()) / temperature
    labels = torch.arange(len(notice_emb), device=notice_emb.device)

    # Loss
    loss = torch.nn.functional.cross_entropy(sim_matrix, labels)

    # Accuracy (correct predictions)
    predictions = torch.argmax(sim_matrix, dim=1)
    accuracy = (predictions == labels).float().mean()

    # Positive similarity (diagonal elements - matching pairs)
    pos_sim = torch.diagonal(sim_matrix).mean()

    # Negative similarity (off-diagonal elements - non-matching pairs)
    batch_size = len(notice_emb)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=notice_emb.device)
    neg_sim = sim_matrix[mask].mean()

    # Similarity gap (positive - negative)
    sim_gap = pos_sim - neg_sim

    return {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'pos_sim': pos_sim.item(),
        'sim_gap': sim_gap.item()
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("Two-Tower V2 모델 학습 (EmbeddingBagCollection)")
    print("=" * 80)

    # 기본값 저장 (resume 시 비교용)
    import sys
    arg_defaults = {}
    parser = parse_args.__wrapped__ if hasattr(parse_args, '__wrapped__') else None

    # 명령줄에서 명시적으로 지정된 인자 찾기
    explicitly_set_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            arg_name = arg.lstrip('--').replace('-', '_')
            explicitly_set_args.add(arg_name)

    # Config 설정
    config = {
        # 데이터 설정
        "batch_size": args.batch_size,
        "test_split": args.test_split,
        "shuffle_seed": args.shuffle_seed,
        "pair_limit": args.pair_limit,

        # DataLoader 설정
        "num_workers": 0,  # V2는 항상 0 (GPU gather)
        "pin_memory": args.pin_memory,
        "shuffle": args.shuffle,
        "streaming": args.streaming,
        "chunk_size": args.chunk_size,
        "feature_chunksize": args.feature_chunksize,
        "test_mode": args.test_mode,
        "use_parquet": args.use_parquet,
        "parquet_dir": args.parquet_dir,

        # 모델 아키텍처
        "categorical_embedding_dim": args.categorical_embedding_dim,
        "notice_dense_input_dim": 256,
        "company_dense_input_dim": 128,
        "tower_hidden_dims": [256, 128],
        "final_embedding_dim": args.final_embedding_dim,
        "dropout_rate": args.dropout_rate,
        "temperature": args.temperature,

        # 학습 설정
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "log_interval": args.log_interval,
        "resume": args.resume,

        # 모델 저장/로딩
        "output_dir": args.output_dir,
        "save_best": args.save_best,
        "save_final": args.save_final,

        # CUDA 최적화
        "enable_tf32": args.enable_tf32,
        "enable_cudnn_benchmark": args.enable_cudnn_benchmark,
        "enable_torch_compile": args.enable_torch_compile,
        "enable_amp": args.enable_amp,

        # 시스템 설정
        "gpu_optimization": "V2: EBC + GPU gather + ID-only + torch.compile + AMP",
        "metadata_path": "meta/metadata.csv"
    }

    # Resume 처리
    if config.get("resume"):
        checkpoint_path = Path(config["resume"])
        if checkpoint_path.exists():
            print(f"\n🔄 이어학습 모드: {checkpoint_path}")
            output_dir_resume = checkpoint_path.parent
            config_json_path = output_dir_resume / "config.json"

            if config_json_path.exists():
                print(f"📋 기존 학습 설정 로드: {config_json_path}")
                with open(config_json_path, "r", encoding="utf-8") as f:
                    saved_config = json.load(f)

                # Config 병합: 명령줄에서 명시한 것은 유지, 나머지는 기존 값 복원
                overridden_params = []
                for key in saved_config:
                    if key not in ["resume", "output_dir", "timestamp", "total_params", "trainable_params", "num_epochs"]:
                        if key in config:
                            # 명령줄에서 명시적으로 지정된 경우 유지
                            if key in explicitly_set_args:
                                overridden_params.append(f"{key}: {saved_config[key]} → {config[key]}")
                            else:
                                config[key] = saved_config[key]

                print(f"✅ 기존 하이퍼파라미터 복원 완료")
                if overridden_params:
                    print(f"📝 명령줄에서 재정의된 파라미터:")
                    for param in overridden_params:
                        print(f"   - {param}")
            else:
                print(f"⚠️ config.json을 찾을 수 없습니다. 명령줄 인자 사용")
        else:
            print(f"⚠️ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    print(f"\n🔧 설정된 하이퍼파라미터:")
    pair_limit_str = f"{config['pair_limit']:,}" if config['pair_limit'] is not None else "전체"
    print(f"   - Data Source: {'Parquet (' + config['parquet_dir'] + ')' if config['use_parquet'] else 'Database'}")
    print(f"   - Pair Limit: {pair_limit_str}")
    print(f"   - Batch Size: {config['batch_size']}")
    print(f"   - Embedding Dim: {config['categorical_embedding_dim']} → {config['final_embedding_dim']}")
    print(f"   - Hidden Dims: {config['tower_hidden_dims']}")
    print(f"   - Learning Rate: {config['learning_rate']}")
    print(f"   - GPU Optimization: torch.compile={config['enable_torch_compile']}, AMP={config['enable_amp']}")
    print(f"   - Temperature: {config['temperature']}")

    # CUDA 최적화
    if config["enable_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if config["enable_cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n디바이스: {device}")
    print("CUDA 가속 최적화 활성화: TF32, cuDNN benchmark\n")

    # 데이터베이스 연결
    print("데이터베이스 연결 중...")
    db = DatabaseConnector()
    schema = build_torchrec_schema_from_meta(
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv"
    )

    print(f"Notice 피처: {len(schema.notice.categorical)}개 범주형, {len(schema.notice.numeric)}개 수치형")
    print(f"Company 피처: {len(schema.company.categorical)}개 범주형, {len(schema.company.numeric)}개 수치형")

    # DataLoader 생성 (V2)
    print("\nDataLoader 생성 중... (PairLoaderV2 - EmbeddingBagCollection용)")
    train_loader, test_loader, metadata = create_pair_dataloaders(
        db_engine=db.engine,
        schema=schema,
        batch_size=config["batch_size"],
        pair_limit=config["pair_limit"],
        test_split=config["test_split"],
        shuffle=config["shuffle"],
        shuffle_seed=config["shuffle_seed"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        streaming=config["streaming"],
        chunk_size=config["chunk_size"],
        feature_chunksize=config["feature_chunksize"],
        device=device,
        test_mode=config["test_mode"],
        use_parquet=config["use_parquet"],
        parquet_dir=config["parquet_dir"],
    )

    # Streaming 모드에서는 len() 호출 불가
    try:
        train_batches = len(train_loader)
        print(f"Train 배치 수: {train_batches}")
    except TypeError:
        print(f"Train 배치 수: streaming")

    if test_loader is not None:
        try:
            test_batches = len(test_loader)
            print(f"Test 배치 수: {test_batches}")
        except TypeError:
            print(f"Test 배치 수: streaming")
    else:
        print(f"Test 배치 수: None")

    # GPU 상주 Feature Tables
    print("\nGPU 상주 Feature Tables 생성 중...")
    notice_store = metadata["notice_store"]
    company_store = metadata["company_store"]

    # FP16으로 상주 (AMP 친화적)
    notice_dense_table = torch.from_numpy(notice_store['dense_projected']).to(
        device, dtype=torch.float16, non_blocking=True
    )
    company_dense_table = torch.from_numpy(company_store['dense_projected']).to(
        device, dtype=torch.float16, non_blocking=True
    )
    notice_cat_table = torch.from_numpy(notice_store['categorical']).to(
        device, dtype=torch.long, non_blocking=True
    )
    company_cat_table = torch.from_numpy(company_store['categorical']).to(
        device, dtype=torch.long, non_blocking=True
    )

    print(f"✅ GPU 상주 완료:")
    print(f"   Notice dense: {notice_dense_table.shape} ({notice_dense_table.dtype})")
    print(f"   Company dense: {company_dense_table.shape} ({company_dense_table.dtype})")
    print(f"   Notice cat: {notice_cat_table.shape}")
    print(f"   Company cat: {company_cat_table.shape}")

    # EBC Metadata
    print("\nEBC Metadata 로딩...")
    notice_meta = get_notice_feature_metadata(
        metadata_path="meta/metadata.csv",
        embedding_dim=config["categorical_embedding_dim"],
        add_unk_token=True
    )
    company_meta = get_company_feature_metadata(
        metadata_path="meta/metadata.csv",
        embedding_dim=config["categorical_embedding_dim"],
        add_unk_token=True
    )

    notice_keys = notice_meta["feature_names"]
    company_keys = company_meta["feature_names"]

    print(f"   Notice keys: {len(notice_keys)}개")
    print(f"   Company keys: {len(company_keys)}개")

    # 모델 생성 (V2)
    print("\n모델 생성 중...")

    notice_tower = NoticeTower(
        metadata_path="meta/metadata.csv",
        categorical_embedding_dim=config["categorical_embedding_dim"],
        dense_input_dim=config["notice_dense_input_dim"],
        tower_hidden_dims=config["tower_hidden_dims"],
        final_embedding_dim=config["final_embedding_dim"],
        dropout_rate=config["dropout_rate"],
        device=device,
        use_fp16=False,  # AMP 사용하므로 모델은 FP32
    )

    company_tower = CompanyTower(
        metadata_path="meta/metadata.csv",
        categorical_embedding_dim=config["categorical_embedding_dim"],
        dense_input_dim=config["company_dense_input_dim"],
        tower_hidden_dims=config["tower_hidden_dims"],
        final_embedding_dim=config["final_embedding_dim"],
        dropout_rate=config["dropout_rate"],
        device=device,
        use_fp16=False,
    )

    # Wrapper
    class TwoTowerWrapper(torch.nn.Module):
        def __init__(self, notice_tower, company_tower):
            super().__init__()
            self.notice_tower = notice_tower
            self.company_tower = company_tower

        def forward(self, batch):
            notice_emb = self.notice_tower(batch["notice"])
            company_emb = self.company_tower(batch["company"])
            return notice_emb, company_emb

    model = TwoTowerWrapper(notice_tower, company_tower).to(device)

    # torch.compile
    if config["enable_torch_compile"]:
        print("\ntorch.compile 활성화 중...")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        print("✅ Compile 완료 (첫 iteration에서 컴파일됨)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터: {total_params:,}\n")

    # Resume 설정 먼저 확인 (optimizer/scheduler 생성 전에 start_epoch 결정 필요)
    best_val_loss = float('inf')
    start_epoch = 0
    checkpoint_to_load = None
    resumed_from = None  # 이어학습 원본 정보

    if config.get("resume"):
        checkpoint_path = Path(config["resume"])
        if checkpoint_path.exists():
            # 체크포인트에서 epoch 정보만 먼저 확인
            checkpoint_info = torch.load(checkpoint_path, map_location='cpu')
            start_epoch = checkpoint_info['epoch'] + 1
            best_val_loss = checkpoint_info['loss']
            checkpoint_to_load = checkpoint_path
            resumed_from = str(checkpoint_path.absolute())

            # 새로운 디렉토리 생성 (이어학습도 별도 디렉토리)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(config["output_dir"]) / f"{timestamp}_resumed"
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"📂 Resume 준비: Epoch {start_epoch}부터 재개 예정")
            print(f"   원본 체크포인트: {checkpoint_path}")
            print(f"   새 모델 저장 경로: {output_dir}")
        else:
            print(f"⚠️ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
            print("새로운 학습을 시작합니다.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(config["output_dir"]) / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config["output_dir"]) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"모델 저장 경로: {output_dir}")

    # 이어학습 시 총 epoch 계산 (start_epoch + 추가 학습할 epoch)
    total_epochs = start_epoch + config["num_epochs"] if config.get("resume") else config["num_epochs"]
    print(f"📊 학습 계획: Epoch {start_epoch+1} ~ {total_epochs} (총 {config['num_epochs']} epochs 추가)")

    # Optimizer & Loss
    print("\nOptimizer & Loss 설정...")
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Streaming 모드에서는 len(train_loader)가 없으므로 적절한 T_max 추정
    if config["streaming"]:
        # pair_limit와 batch_size로 추정
        if config["pair_limit"]:
            estimated_batches = config["pair_limit"] // config["batch_size"]
        else:
            estimated_batches = 1000  # 기본값
        # T_max는 전체 학습 기간 기준 (이어학습 시 total_epochs 사용)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs * estimated_batches)
        print(f"   Estimated batches per epoch: {estimated_batches}")
        print(f"   Scheduler T_max: {total_epochs} epochs * {estimated_batches} batches = {total_epochs * estimated_batches}")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs * len(train_loader))
        print(f"   Scheduler T_max: {total_epochs} epochs * {len(train_loader)} batches = {total_epochs * len(train_loader)}")

    scaler = torch.amp.GradScaler("cuda", enabled=config["enable_amp"])

    print(f"   Optimizer: AdamW (lr={config['learning_rate']})")
    print(f"   Scheduler: CosineAnnealingLR")
    print(f"   AMP: {config['enable_amp']}")
    print(f"   torch.compile: {config['enable_torch_compile']}\n")

    # 체크포인트 로드 (optimizer, scheduler 생성 후)
    if checkpoint_to_load is not None:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, checkpoint_to_load, scheduler,
            config_lr=config["learning_rate"]
        )
        print(f"✅ 모델, Optimizer, Scheduler 로드 완료. Epoch {start_epoch}부터 재개")

    # Config 저장 (이어학습 시에도 저장)
    config_save = config.copy()
    config_save["timestamp"] = timestamp
    config_save["total_params"] = total_params
    config_save["trainable_params"] = total_params
    config_save["total_epochs"] = total_epochs
    config_save["start_epoch"] = start_epoch

    # 이어학습 정보 추가
    if resumed_from is not None:
        config_save["resumed_from"] = resumed_from
        config_save["resumed_epoch"] = start_epoch

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_save, f, indent=2, ensure_ascii=False)
    print(f"학습 설정 저장: {output_dir / 'config.json'}")

    # 학습 루프
    console = Console()
    console.print("\n" + "=" * 80)
    console.print("[bold green]학습 시작[/bold green]")
    console.print("=" * 80)

    # Rich Progress 설정
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[batch_count]}/{task.fields[total_estimate]} batches"),
        TextColumn("•"),
        TextColumn("[yellow]Loss: {task.fields[loss]:.4f}"),
        TextColumn("•"),
        TextColumn("[blue]Acc: {task.fields[accuracy]:.3f}"),
        TextColumn("•"),
        TextColumn("[magenta]Pos-Sim: {task.fields[pos_sim]:.3f}"),
        TextColumn("•"),
        TextColumn("[red]Sim-Gap: {task.fields[sim_gap]:.3f}"),
        TextColumn("•"),
        TextColumn("[green]{task.fields[throughput]:.1f} batch/s"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        console=console,
    )

    # Progress를 전체 학습 루프에서 한 번만 시작
    progress.start()

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_pos_sim = 0.0
        epoch_sim_gap = 0.0
        epoch_start_time = time.time()
        batch_times = deque(maxlen=config["log_interval"])  # 고정 크기 deque 사용
        batch_count = 0  # Streaming 모드를 위한 batch 카운터

        # Progress bar에 표시할 현재 메트릭 (마지막 계산값 유지)
        last_acc = 0.0
        last_pos_sim = 0.0
        last_sim_gap = 0.0

        console.print(f"\n[bold cyan]Epoch {epoch+1}/{total_epochs}[/bold cyan]")

        # Epoch 설정 (shuffle용)
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        # Total batches 계산: pair_limit / batch_size
        if config["streaming"]:
            # Streaming: train split 고려 (test_split만큼 빼기)
            total_pairs = metadata.get("total_pairs")
            if total_pairs is not None:
                train_pairs = int(total_pairs * (1 - config["test_split"]))
                total_batches = train_pairs // config["batch_size"]
                console.print(f"   📊 Total pairs: {total_pairs:,} → Train pairs: {train_pairs:,} → Train batches: {total_batches:,}")
            else:
                total_batches = None
                console.print(f"   ⚠️ total_pairs를 가져올 수 없음 (metadata: {metadata.keys()})")
        else:
            # Non-streaming: len() 사용
            try:
                total_batches = len(train_loader)
            except:
                total_batches = None

        task = progress.add_task(
            f"[cyan]Training Epoch {epoch+1}/{total_epochs}",
            total=total_batches,
            batch_count=0,
            total_estimate=total_batches if total_batches else "?",
            loss=0.0,
            accuracy=0.0,
            pos_sim=0.0,
            sim_gap=0.0,
            throughput=0.0
        )

        for step, batch in enumerate(train_loader):
            batch_count += 1
            step_start_time = time.time()

            # Unpack
            notice_idx_cpu, company_idx_cpu = batch

            # H2D (인덱스만)
            notice_idx = notice_idx_cpu.to(device, non_blocking=True)
            company_idx = company_idx_cpu.to(device, non_blocking=True)

            # GPU gather
            notice_dense_b = torch.index_select(notice_dense_table, dim=0, index=notice_idx)
            company_dense_b = torch.index_select(company_dense_table, dim=0, index=company_idx)
            notice_cat_b = torch.index_select(notice_cat_table, dim=0, index=notice_idx)
            company_cat_b = torch.index_select(company_cat_table, dim=0, index=company_idx)

            # GPU-KJT
            notice_kjt = create_kjt_from_batch_gpu(notice_cat_b, notice_keys, device)
            company_kjt = create_kjt_from_batch_gpu(company_cat_b, company_keys, device)

            # Batch 구성
            batch_gpu = {
                "notice": {"dense": notice_dense_b, "kjt": notice_kjt},
                "company": {"dense": company_dense_b, "kjt": company_kjt},
            }

            # Forward + Loss
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config["enable_amp"]):
                notice_emb, company_emb = model(batch_gpu)
                loss = contrastive_loss(notice_emb, company_emb, temperature=config["temperature"])

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping (NaN 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # scheduler.step()은 optimizer.step()이 실행된 경우에만
            # scaler가 gradient를 skip하지 않았으면 scheduler도 step
            scheduler.step()

            # Loss 저장 (매 배치)
            import math
            current_loss = loss.item()
            if not math.isnan(current_loss) and not math.isinf(current_loss):
                epoch_loss += current_loss
            else:
                # NaN/Inf 발생 시 경고 및 디버깅 정보
                if batch_count <= 10 or batch_count % 100 == 0:
                    console.print(f"⚠️ Batch {batch_count}: Loss is {'NaN' if math.isnan(current_loss) else 'Inf'}")
                    # 임베딩 통계 확인
                    with torch.no_grad():
                        notice_norm = torch.norm(notice_emb, p=2, dim=1)
                        company_norm = torch.norm(company_emb, p=2, dim=1)
                        sim_matrix = torch.mm(notice_emb, company_emb.t()) / config["temperature"]
                        console.print(f"   Notice norm: min={notice_norm.min():.4f}, max={notice_norm.max():.4f}, mean={notice_norm.mean():.4f}")
                        console.print(f"   Company norm: min={company_norm.min():.4f}, max={company_norm.max():.4f}, mean={company_norm.mean():.4f}")
                        console.print(f"   Similarity matrix: min={sim_matrix.min():.4f}, max={sim_matrix.max():.4f}, mean={sim_matrix.mean():.4f}")
                current_loss = 0.0

            # Metrics 계산 (100배치마다 또는 처음 10개)
            if batch_count <= 10 or batch_count % 100 == 0:
                with torch.no_grad():
                    metrics = calculate_metrics(notice_emb, company_emb, temperature=config["temperature"])
                    last_acc = metrics['accuracy']
                    last_pos_sim = metrics['pos_sim']
                    last_sim_gap = metrics['sim_gap']

                    # Epoch 누적
                    epoch_acc += last_acc
                    epoch_pos_sim += last_pos_sim
                    epoch_sim_gap += last_sim_gap

            # 배치 시간 측정
            step_end_time = time.time()
            batch_times.append(step_end_time - step_start_time)

            # Throughput 계산 (deque가 자동으로 크기 제한)
            if len(batch_times) >= config["log_interval"]:
                avg_batch_time = sum(batch_times) / len(batch_times)
                batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
            else:
                batches_per_sec = 0.0

            # Rich progress 업데이트 (주기적으로만 업데이트)
            if batch_count % config["log_interval"] == 0 or batch_count <= 10:
                progress.update(
                    task,
                    advance=config["log_interval"] if batch_count > 10 else 1,
                    batch_count=batch_count,
                    loss=current_loss,
                    accuracy=last_acc,
                    pos_sim=last_pos_sim,
                    sim_gap=last_sim_gap,
                    throughput=batches_per_sec
                )
            elif batch_count % 100 == 0:
                # 100배치마다 간단히 업데이트
                progress.update(
                    task,
                    advance=100,
                    batch_count=batch_count,
                    loss=current_loss,
                    accuracy=last_acc,
                    pos_sim=last_pos_sim,
                    sim_gap=last_sim_gap,
                    throughput=batches_per_sec
                )

        # Epoch 완료
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count  # batch_count 사용 (streaming 호환)

        # Metrics는 100배치마다 계산했으므로 샘플 수로 나눔
        metric_sample_count = (batch_count // 100) + min(batch_count, 10)  # 100배치마다 + 처음 10개
        avg_epoch_acc = epoch_acc / metric_sample_count if metric_sample_count > 0 else 0.0
        avg_epoch_pos_sim = epoch_pos_sim / metric_sample_count if metric_sample_count > 0 else 0.0
        avg_epoch_sim_gap = epoch_sim_gap / metric_sample_count if metric_sample_count > 0 else 0.0

        # Epoch 전체 throughput
        total_batches = batch_count  # batch_count 사용
        total_samples = total_batches * config["batch_size"]
        epoch_batches_per_sec = total_batches / epoch_duration
        epoch_samples_per_sec = total_samples / epoch_duration

        console.print(f"\n[bold green]✅ Epoch {epoch+1} 완료[/bold green]")
        console.print(f"   Loss: {avg_epoch_loss:.4f} | Acc: {avg_epoch_acc:.3f} | "
              f"Pos-Sim: {avg_epoch_pos_sim:.3f} | Sim-Gap: {avg_epoch_sim_gap:.3f}")
        console.print(f"   시간: {epoch_duration:.2f}s | "
              f"Throughput: {epoch_batches_per_sec:.2f} batch/s ({epoch_samples_per_sec:.0f} samples/s)")

        # Validation
        avg_val_loss = avg_epoch_loss
        avg_val_acc = avg_epoch_acc
        avg_val_pos_sim = avg_epoch_pos_sim
        avg_val_sim_gap = avg_epoch_sim_gap

        if test_loader is not None:
            model.eval()
            val_losses = []
            val_accs = []
            val_pos_sims = []
            val_sim_gaps = []

            # Progress bar에 표시할 현재 메트릭 (마지막 계산값 유지)
            val_last_acc = 0.0
            val_last_pos_sim = 0.0
            val_last_sim_gap = 0.0

            # Validation total batches
            val_total_batches = None
            if not config["streaming"]:
                try:
                    val_total_batches = len(test_loader)
                except:
                    pass

            # Streaming 모드에서는 test split 고려한 total batches 추정
            if config["streaming"] and val_total_batches is None:
                total_pairs = metadata.get("total_pairs")
                if total_pairs is not None:
                    test_pairs = int(total_pairs * config["test_split"])
                    val_total_batches = test_pairs // config["batch_size"]

            with torch.no_grad():
                val_task = progress.add_task(
                    "[magenta]Validation",
                    total=val_total_batches,
                    batch_count=0,
                    total_estimate=val_total_batches if val_total_batches else "?",
                    loss=0.0,
                    accuracy=0.0,
                    pos_sim=0.0,
                    sim_gap=0.0,
                    throughput=0.0
                )

                val_batch_count = 0
                val_batch_times = deque(maxlen=10)  # 고정 크기 deque 사용

                for batch in test_loader:
                    val_batch_start = time.time()
                    val_batch_count += 1

                    notice_idx_cpu, company_idx_cpu = batch
                    notice_idx = notice_idx_cpu.to(device, non_blocking=True)
                    company_idx = company_idx_cpu.to(device, non_blocking=True)

                    notice_dense_b = torch.index_select(notice_dense_table, dim=0, index=notice_idx)
                    company_dense_b = torch.index_select(company_dense_table, dim=0, index=company_idx)
                    notice_cat_b = torch.index_select(notice_cat_table, dim=0, index=notice_idx)
                    company_cat_b = torch.index_select(company_cat_table, dim=0, index=company_idx)

                    notice_kjt = create_kjt_from_batch_gpu(notice_cat_b, notice_keys, device)
                    company_kjt = create_kjt_from_batch_gpu(company_cat_b, company_keys, device)

                    batch_gpu = {
                        "notice": {"dense": notice_dense_b, "kjt": notice_kjt},
                        "company": {"dense": company_dense_b, "kjt": company_kjt},
                    }

                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config["enable_amp"]):
                        notice_emb, company_emb = model(batch_gpu)
                        loss = contrastive_loss(notice_emb, company_emb, temperature=config["temperature"])

                    current_loss = loss.item()

                    # NaN 체크
                    import math
                    if math.isnan(current_loss) or math.isinf(current_loss):
                        if val_batch_count <= 10 or val_batch_count % 1000 == 0:
                            console.print(f"\n⚠️ Validation Batch {val_batch_count}: Loss is {'NaN' if math.isnan(current_loss) else 'Inf'}")
                            with torch.no_grad():
                                notice_norm = torch.norm(notice_emb, p=2, dim=1)
                                company_norm = torch.norm(company_emb, p=2, dim=1)
                                sim_matrix = torch.mm(notice_emb, company_emb.t()) / config["temperature"]
                                console.print(f"   Notice norm: min={notice_norm.min():.4f}, max={notice_norm.max():.4f}, mean={notice_norm.mean():.4f}")
                                console.print(f"   Company norm: min={company_norm.min():.4f}, max={company_norm.max():.4f}, mean={company_norm.mean():.4f}")
                                console.print(f"   Similarity matrix: min={sim_matrix.min():.4f}, max={sim_matrix.max():.4f}, mean={sim_matrix.mean():.4f}")
                    else:
                        val_losses.append(loss.detach())

                    # Metrics 계산 (100배치마다 또는 처음 10개)
                    if val_batch_count <= 10 or val_batch_count % 100 == 0:
                        metrics = calculate_metrics(notice_emb, company_emb, temperature=config["temperature"])
                        val_last_acc = metrics['accuracy']
                        val_last_pos_sim = metrics['pos_sim']
                        val_last_sim_gap = metrics['sim_gap']

                        val_accs.append(val_last_acc)
                        val_pos_sims.append(val_last_pos_sim)
                        val_sim_gaps.append(val_last_sim_gap)

                    # 배치 시간 측정
                    val_batch_end = time.time()
                    val_batch_times.append(val_batch_end - val_batch_start)

                    # Throughput 계산 (deque가 자동으로 크기 제한)
                    if len(val_batch_times) >= 10:
                        avg_batch_time = sum(val_batch_times) / len(val_batch_times)
                        batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
                    else:
                        batches_per_sec = 0.0

                    # Progress 업데이트
                    progress.update(
                        val_task,
                        advance=1,
                        batch_count=val_batch_count,
                        loss=current_loss,
                        accuracy=val_last_acc,
                        pos_sim=val_last_pos_sim,
                        sim_gap=val_last_sim_gap,
                        throughput=batches_per_sec
                    )

            # val_losses가 비어있지 않은 경우에만 계산
            if len(val_losses) > 0:
                avg_val_loss = float(torch.stack(val_losses).mean().cpu())
                avg_val_acc = sum(val_accs) / len(val_accs) if len(val_accs) > 0 else 0.0
                avg_val_pos_sim = sum(val_pos_sims) / len(val_pos_sims) if len(val_pos_sims) > 0 else 0.0
                avg_val_sim_gap = sum(val_sim_gaps) / len(val_sim_gaps) if len(val_sim_gaps) > 0 else 0.0

                console.print(f"   [bold magenta]Validation ({val_batch_count} batches)[/bold magenta]")
                console.print(f"   Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.3f} | "
                      f"Pos-Sim: {avg_val_pos_sim:.3f} | Sim-Gap: {avg_val_sim_gap:.3f}")
            else:
                console.print(f"   [bold yellow]⚠️ Validation 데이터 없음 - 0 batches processed[/bold yellow]")

            # Validation task 제거
            progress.remove_task(val_task)

        # Training task 제거
        progress.remove_task(task)

        # 체크포인트 저장
        epoch_metrics = {
            'train_loss': avg_epoch_loss,
            'train_acc': avg_epoch_acc,
            'train_pos_sim': avg_epoch_pos_sim,
            'train_sim_gap': avg_epoch_sim_gap,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc,
            'val_pos_sim': avg_val_pos_sim,
            'val_sim_gap': avg_val_sim_gap,
            'throughput_batch_s': epoch_batches_per_sec,
            'throughput_sample_s': epoch_samples_per_sec,
        }

        save_checkpoint(model, optimizer, epoch, avg_val_loss, output_dir, metrics=epoch_metrics, scheduler=scheduler)

        # 최고 성능 모델 저장
        if config["save_best"] and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, output_dir, is_best=True, metrics=epoch_metrics, scheduler=scheduler)

        # Epoch별 결과를 CSV에 저장
        print(f"   Epoch {epoch+1} 결과 기록 중...")
        epoch_hyperparams = {
            "batch_size": config["batch_size"],
            "model_params": total_params,
            "embedding_dim": config["categorical_embedding_dim"],
            "final_embedding_dim": config["final_embedding_dim"],
            "hidden_dims": config["tower_hidden_dims"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "dropout_rate": config["dropout_rate"],
            "temperature": config["temperature"],
            "epochs": f"{epoch+1}/{total_epochs}",  # 현재 epoch / 전체 epochs 표시
            "train_batches": total_batches,
            "test_batches": "streaming" if config["streaming"] else 0,
            "gpu_optimization": config["gpu_optimization"]
        }

        epoch_metrics_for_csv = {
            "train_loss": avg_epoch_loss,
            "train_acc": avg_epoch_acc,
            "train_pos_sim": avg_epoch_pos_sim,
            "train_sim_gap": avg_epoch_sim_gap,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "val_pos_sim": avg_val_pos_sim,
            "val_sim_gap": avg_val_sim_gap,
            "throughput_batch_s": epoch_batches_per_sec,
            "throughput_sample_s": epoch_samples_per_sec,
        }

        save_training_results(epoch_hyperparams, epoch_metrics_for_csv)
        print(f"   ✅ Epoch {epoch+1} 결과 저장 완료")

        print("=" * 80)

    # Progress 종료
    progress.stop()

    print("🎉 학습 완료!")

    # 최종 모델 저장
    if config["save_final"]:
        final_metrics = {
            'train_loss': avg_epoch_loss,
            'train_acc': avg_epoch_acc,
            'train_pos_sim': avg_epoch_pos_sim,
            'train_sim_gap': avg_epoch_sim_gap,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc,
            'val_pos_sim': avg_val_pos_sim,
            'val_sim_gap': avg_val_sim_gap,
            'throughput_batch_s': epoch_batches_per_sec,
            'throughput_sample_s': epoch_samples_per_sec,
        }
        save_checkpoint(model, optimizer, total_epochs-1, 0.0, output_dir, is_final=True, metrics=final_metrics, preprocessor=metadata.get("preprocessor"), scheduler=scheduler)

    print("\n=== 학습 완료 ===")


if __name__ == "__main__":
    main()
