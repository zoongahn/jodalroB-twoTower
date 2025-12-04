#!/usr/bin/env python3
"""
Two-Tower Model Training Script - V2 (EmbeddingBagCollection)

GPU 최적화 파이프라인:
- PairLoaderV2: ID-only DataLoader (H2D 최소화)
- EmbeddingBagCollection: GPU 최적화 임베딩
- GPU gather + KJT 생성
- torch.compile + AMP + TF32

클라우드 환경용 수정버전 (H100pcle) 

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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
import time
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse
import json
import math  # NaN/inf 체크용

# Project imports
from database.database_connector import DatabaseConnector
from preprocess.torchrec.schema import build_torchrec_schema_from_meta
from src.towers.pairs.pair_loader import create_pair_dataloaders
from src.towers.tower import NoticeTower, CompanyTower
from src.towers.kjt_utils import create_kjt_from_batch_gpu
from src.towers.embedding_config import get_notice_feature_metadata, get_company_feature_metadata
from src.evaluation.evaluator import TwoTowerEvaluator



# =====================================================================
# TwoTowerWrapper를 모듈 최상단에 정의 (pickle 가능)
# =====================================================================
class TwoTowerWrapper(torch.nn.Module):
    """
    학습/추론 공용 Wrapper

    입력 batch:
        {
            "notice": {"dense": Tensor, "kjt": KeyedJaggedTensor},
            "company": {"dense": Tensor, "kjt": KeyedJaggedTensor},
        }

    출력:
        notice_emb: (B, D)
        company_emb: (B, D)
    """

    def __init__(self, notice_tower, company_tower):
        super().__init__()
        self.notice_tower = notice_tower
        self.company_tower = company_tower

    def forward(self, batch):
        notice_emb = self.notice_tower(batch["notice"])
        company_emb = self.company_tower(batch["company"])
        return notice_emb, company_emb


# =====================================================================
# 학습 결과 기록 유틸
# =====================================================================
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
        "val_loss": metrics.get("val_loss", "N/A"),
        "val_acc": metrics.get("val_acc", "N/A"),
        "recall_at_5": metrics.get("recall_at_5", "N/A"),
        "recall_at_10": metrics.get("recall_at_10", "N/A"),
        "mrr": metrics.get("mrr", "N/A"),
        "similarity_gap": metrics.get("similarity_gap", "N/A"),
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


# =====================================================================
# Argument 파서
# =====================================================================
def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="Two-Tower V2 모델 학습 스크립트 (EmbeddingBagCollection)")

    # 데이터 설정
    parser.add_argument("--batch_size", type=int, default=256, help="배치 크기 (H100 권장값)")
    parser.add_argument("--test_split", type=float, default=0.2, help="테스트 데이터 비율")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--pair_limit", type=lambda x: None if x.lower() == 'none' else int(x),
                        default=None, help="학습에 사용할 최대 페어 수 (none이면 전체 데이터 사용)")

    # DataLoader 설정
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory 사용")
    parser.add_argument("--shuffle", action="store_true", default=False, help="데이터 셔플 활성화")
    parser.add_argument("--streaming", action="store_true", default=False, help="스트리밍 모드 사용")
    parser.add_argument("--chunk_size", type=int, default=10000, help="스트리밍 시 pair chunk 크기")
    parser.add_argument("--feature_chunksize", type=int, default=10000, help="피처 로딩 시 chunk 크기")
    parser.add_argument("--test_mode", action="store_true", default=False, help="Test mode: pair_limit에 해당하는 feature만 로딩")

    # 모델 아키텍처
    parser.add_argument("--categorical_embedding_dim", type=int, default=32, help="범주형 임베딩 차원")
    parser.add_argument("--final_embedding_dim", type=int, default=128, help="최종 임베딩 차원")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="드롭아웃 비율")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")

    # 학습 설정
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="가중치 감쇠")
    parser.add_argument("--num_epochs", type=int, default=3, help="학습 에포크 수")
    parser.add_argument("--log_interval", type=int, default=10, help="로그/프로그레스 업데이트 간격 (step 단위)")
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
    parser.add_argument("--enable_cuda_graphs", action="store_true", default=False,
                        help="(현재 비활성화됨) CUDA Graphs 기반 학습 루프 활성화 옵션 (TorchRec KJT와 미호환)")

    return parser.parse_args()


# =====================================================================
# 체크포인트 저장/로딩 유틸
# =====================================================================
def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    save_dir,
    is_best=False,
    is_final=False,
    metrics=None,
    config=None,
):
    """
    체크포인트 및 (최종 시) 전체 모델 번들을 저장한다.

    - 매 epoch: checkpoint_epoch_*.pt (state_dict + optimizer + loss + metrics)
    - 최고 성능: best_model.pt (same format)
    - 최종: final_model.pt + model_weights.pt + full_model_final.pt (번들)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if metrics is not None:
        checkpoint['metrics'] = metrics

    # 일반 epoch 체크포인트 (최종이 아닌 경우)
    if not is_final:
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"   체크포인트 저장: {checkpoint_path}")

    # 최고 성능 모델
    if is_best:
        best_path = save_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"   최고 성능 모델 저장: {best_path}")

    # 최종 모델 + 전체 번들
    if is_final:
        final_path = save_dir / 'final_model.pt'
        torch.save(checkpoint, final_path)
        model_only_path = save_dir / 'model_weights.pt'
        torch.save(model.state_dict(), model_only_path)
        print(f"최종 모델 저장: {final_path}")
        print(f"모델 가중치 저장: {model_only_path}")

        # 🔥 전체 번들 저장 (feature_importance_v2 에서 사용하는 파일)
        bundle = {
            "model": model,        # TwoTowerWrapper (torch.compile 적용된 상태일 수도 있음)
            "config": config or {},
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        full_path = save_dir / 'full_model_final.pt'
        torch.save(bundle, full_path)
        print(f"전체 모델 번들 저장: {full_path}")


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


# =====================================================================
# InfoNCE Loss
# =====================================================================
def contrastive_loss(notice_emb, company_emb, temperature=0.07):
    """Simple InfoNCE loss (in-batch negatives)"""
    sim_matrix = torch.mm(notice_emb, company_emb.t()) / temperature
    labels = torch.arange(len(notice_emb), device=notice_emb.device)
    return torch.nn.functional.cross_entropy(sim_matrix, labels)


# =====================================================================
# main
# =====================================================================
def main():
    args = parse_args()

    print("=" * 80)
    print("Two-Tower V2 모델 학습 (EmbeddingBagCollection)")
    print("=" * 80)

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
        "enable_cuda_graphs": args.enable_cuda_graphs,

        # 시스템 설정
        "gpu_optimization": "V2: EBC + GPU gather + ID-only + torch.compile + AMP",
        "metadata_path": "meta/metadata.csv"
    }

    # Resume 처리 (기존 config.json 병합)
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

                # Config 병합 (resume, output_dir, num_epochs 제외)
                for key in saved_config:
                    if key not in ["resume", "output_dir", "timestamp", "total_params", "trainable_params", "num_epochs"]:
                        if key in config:
                            config[key] = saved_config[key]

                print(f"✅ 기존 하이퍼파라미터 복원 완료")
            else:
                print(f"⚠️ config.json을 찾을 수 없습니다. 명령줄 인자 사용")
        else:
            print(f"⚠️ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    print(f"\n🔧 설정된 하이퍼파라미터:")
    pair_limit_str = f"{config['pair_limit']:,}" if config['pair_limit'] is not None else "전체"
    print(f"   - Pair Limit: {pair_limit_str}")
    print(f"   - Batch Size: {config['batch_size']}")
    print(f"   - Embedding Dim: {config['categorical_embedding_dim']} → {config['final_embedding_dim']}")
    print(f"   - Hidden Dims: {config['tower_hidden_dims']}")
    print(f"   - Learning Rate: {config['learning_rate']}")
    print(f"   - GPU Optimization: torch.compile={config['enable_torch_compile']}, AMP={config['enable_amp']}, CUDA Graphs(강제 비활성)={config['enable_cuda_graphs']}")
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

    # ⚠️ CUDA Graphs 강제 비활성화 (TorchRec KJT와 호환 문제)
    if config["enable_cuda_graphs"]:
        print("⚠️ 현재 TorchRec KeyedJaggedTensor가 CUDA Graphs capture와 호환되지 않으므로 CUDA Graphs를 비활성화합니다.")
    use_cuda_graphs = False

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
    )

    # Streaming 모드에서는 len() 호출 불가
    try:
        train_batches = len(train_loader)
        print(f"Train 배치 수: {train_batches}")
    except TypeError:
        print(f"Train 배치 수: streaming")
        train_batches = None

    if test_loader is not None:
        try:
            test_batches = len(test_loader)
            print(f"Test 배치 수: {test_batches}")
        except TypeError:
            print(f"Test 배치 수: streaming")
            test_batches = None
    else:
        print(f"Test 배치 수: None")
        test_batches = None

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

    model = TwoTowerWrapper(notice_tower, company_tower).to(device)

    # torch.compile
    if config["enable_torch_compile"]:
        print("\ntorch.compile 활성화 중...")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        print("✅ Compile 완료 (첫 iteration에서 컴파일됨)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터: {total_params:,}\n")

    # Optimizer & Loss
    print("Optimizer & Loss 설정...")
    adamw_kwargs = {}
    if torch.cuda.is_available():
        adamw_kwargs["fused"] = True
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        **adamw_kwargs,
    )

    # Streaming 모드에서는 len(train_loader)가 없으므로 적절한 T_max 추정
    if config["streaming"]:
        if config["pair_limit"]:
            estimated_batches = config["pair_limit"] // config["batch_size"]
        else:
            estimated_batches = 1000  # 기본값
        scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"] * estimated_batches)
        print(f"   Estimated batches per epoch: {estimated_batches}")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"] * (train_batches or 1))

    # GradScaler (CUDA Graphs는 강제 off)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config["enable_amp"]
    )

    print(f"   Optimizer: AdamW (lr={config['learning_rate']})")
    print(f"   Scheduler: CosineAnnealingLR")
    print(f"   AMP: {config['enable_amp']} (GradScaler enabled={scaler.is_enabled()})")
    print(f"   torch.compile: {config['enable_torch_compile']}\n")

    # Output directory 설정
    best_val_loss = float('inf')
    start_epoch = 0

    if config.get("resume"):
        checkpoint_path = Path(config["resume"])
        if checkpoint_path.exists():
            start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
            output_dir = checkpoint_path.parent
            print(f"✅ 모델 로드 완료. Epoch {start_epoch}부터 재개")
            print(f"   기존 모델 저장 경로: {output_dir}")
        else:
            print(f"⚠️ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
            print("새로운 학습을 시작합니다.")
            start_epoch = 0
            best_val_loss = float('inf')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(config["output_dir"]) / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config["output_dir"]) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"모델 저장 경로: {output_dir}")

    # Config 저장
    if not config.get("resume"):
        config_save = config.copy()
        config_save["timestamp"] = timestamp
        config_save["total_params"] = total_params
        config_save["trainable_params"] = total_params
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
        TextColumn("[green]{task.fields[throughput]:.1f} batch/s"),
        TimeElapsedColumn(),
        console=console,
    )

    progress.start()

    last_epoch_avg_loss = None
    last_epoch_val_loss = None
    last_epoch_batches_per_sec = None
    last_epoch_samples_per_sec = None

    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        batch_times = []
        batch_count = 0  # Streaming 모드를 위한 batch 카운터

        console.print(f"\n[bold cyan]Epoch {epoch+1}/{config['num_epochs']}[/bold cyan]")

        # Epoch 설정 (shuffle용)
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        # Total batches 계산
        if config["streaming"]:
            train_pairs = int(config["pair_limit"] * (1 - config["test_split"])) if config["pair_limit"] else 0
            total_batches = train_pairs // config["batch_size"] if train_pairs > 0 else None
        else:
            total_batches = train_batches

        task = progress.add_task(
            f"[cyan]Training Epoch {epoch+1}",
            total=total_batches,
            batch_count=0,
            total_estimate=total_batches if total_batches else "?",
            loss=0.0,
            throughput=0.0
        )

        for step, batch in enumerate(train_loader):
            batch_count += 1
            step_start_time = time.time()

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

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config["enable_amp"]):
                notice_emb, company_emb = model(batch_gpu)
                loss = contrastive_loss(notice_emb, company_emb, temperature=config["temperature"])

            # Backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            current_loss = loss.item() if not (math.isnan(loss.item()) or math.isinf(loss.item())) else 0.0

            # Loss 저장 (NaN 체크)
            if not math.isnan(current_loss) and not math.isinf(current_loss):
                epoch_loss += current_loss
            else:
                current_loss = 0.0

            # 배치 시간 측정
            step_end_time = time.time()
            batch_times.append(step_end_time - step_start_time)

            # Throughput 계산 (최근 log_interval 개 기준)
            if len(batch_times) >= config["log_interval"]:
                recent_batches = batch_times[-config["log_interval"]:]
                avg_batch_time = sum(recent_batches) / len(recent_batches)
                batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
            else:
                batches_per_sec = 0.0

            # Rich progress 업데이트 (희소화)
            if (
                batch_count % config["log_interval"] == 0
                or (total_batches and batch_count == total_batches)
            ):
                progress.update(
                    task,
                    advance=config["log_interval"] if total_batches else 1,
                    batch_count=batch_count,
                    loss=current_loss,
                    throughput=batches_per_sec,
                )

        # Epoch 완료
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count  # batch_count 사용 (streaming 호환)

        # Epoch 전체 throughput
        total_batches = batch_count  # batch_count 사용
        total_samples = total_batches * config["batch_size"]
        epoch_batches_per_sec = total_batches / epoch_duration
        epoch_samples_per_sec = total_samples / epoch_duration

        last_epoch_avg_loss = avg_epoch_loss
        last_epoch_batches_per_sec = epoch_batches_per_sec
        last_epoch_samples_per_sec = epoch_samples_per_sec

        console.print(f"\n[bold green]✅ Epoch {epoch+1} 완료[/bold green] - Avg Loss: {avg_epoch_loss:.4f}")
        console.print(
            f"   시간: {epoch_duration:.2f}s | "
            f"Throughput: {epoch_batches_per_sec:.2f} batch/s ({epoch_samples_per_sec:.0f} samples/s)"
        )

        # Validation
        avg_val_loss = avg_epoch_loss
        if test_loader is not None:
            model.eval()
            val_losses = []

            # Validation total batches
            if not config["streaming"]:
                val_total_batches = test_batches
            else:
                if config["pair_limit"]:
                    test_pairs = int(config["pair_limit"] * config["test_split"])
                    val_total_batches = test_pairs // config["batch_size"] if test_pairs > 0 else None
                else:
                    val_total_batches = None

            with torch.no_grad():
                val_task = progress.add_task(
                    "[magenta]Validation",
                    total=val_total_batches,
                    batch_count=0,
                    total_estimate=val_total_batches if val_total_batches else "?",
                    loss=0.0,
                    throughput=0.0
                )

                val_batch_count = 0
                val_batch_times = []

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

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config["enable_amp"]):
                        notice_emb, company_emb = model(batch_gpu)
                        loss = contrastive_loss(notice_emb, company_emb, temperature=config["temperature"])

                    current_loss = loss.item()
                    val_losses.append(loss.detach())

                    # 배치 시간 측정
                    val_batch_end = time.time()
                    val_batch_times.append(val_batch_end - val_batch_start)

                    # Throughput 계산 (최근 10개 기준)
                    if len(val_batch_times) >= 10:
                        recent_batches = val_batch_times[-10:]
                        avg_batch_time = sum(recent_batches) / len(recent_batches)
                        batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
                    else:
                        batches_per_sec = 0.0

                    # Progress 업데이트 (희소화)
                    if (
                        val_batch_count % config["log_interval"] == 0
                        or (val_total_batches and val_batch_count == val_total_batches)
                    ):
                        progress.update(
                            val_task,
                            advance=1,
                            batch_count=val_batch_count,
                            loss=current_loss,
                            throughput=batches_per_sec,
                        )

            if len(val_losses) > 0:
                avg_val_loss = float(torch.stack(val_losses).mean().cpu())
                last_epoch_val_loss = avg_val_loss
                console.print(f"   [bold magenta]Val Loss: {avg_val_loss:.4f}[/bold magenta] ({val_batch_count} batches)")
            else:
                console.print(f"   [bold yellow]⚠️ Validation 데이터 없음 - 0 batches processed[/bold yellow]")

            progress.remove_task(val_task)
        else:
            last_epoch_val_loss = avg_val_loss

        progress.remove_task(task)

        # 체크포인트 저장
        epoch_metrics = {
            'train_loss': avg_epoch_loss,
            'val_loss': avg_val_loss,
            'throughput_batch_s': epoch_batches_per_sec,
            'throughput_sample_s': epoch_samples_per_sec,
        }
        
        save_checkpoint(
            model,
            optimizer,
            epoch,
            avg_val_loss,
            output_dir,
            metrics=epoch_metrics,
            config=config
        )

        # 최고 성능 모델 저장
        if config["save_best"] and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_val_loss,
                output_dir,
                is_best=True,
                metrics=epoch_metrics,
                config=config
            )

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
            "epochs": f"{epoch+1}/{config['num_epochs']}",
            "train_batches": total_batches,
            "test_batches": "streaming" if config["streaming"] else (test_batches or 0),
            "gpu_optimization": config["gpu_optimization"]
        }

        epoch_metrics_for_csv = {
            "train_loss": avg_epoch_loss,
            "val_loss": avg_val_loss,
            "throughput_batch_s": epoch_batches_per_sec,
            "throughput_sample_s": epoch_samples_per_sec,
        }

        save_training_results(epoch_hyperparams, epoch_metrics_for_csv)
        print(f"   ✅ Epoch {epoch+1} 결과 저장 완료")

        print("=" * 80)

    # Progress 종료
    progress.stop()

    print("🎉 학습 완료!")

    # 최종 모델 및 전체 번들 저장
    if config["save_final"]:
        final_metrics = {
            'train_loss': last_epoch_avg_loss,
            'val_loss': last_epoch_val_loss,
            'throughput_batch_s': last_epoch_batches_per_sec,
            'throughput_sample_s': last_epoch_samples_per_sec,
        }
        save_checkpoint(
            model,
            optimizer,
            config["num_epochs"]-1,
            last_epoch_val_loss if last_epoch_val_loss is not None else 0.0,
            output_dir,
            is_final=True,
            metrics=final_metrics,
            config=config
        )

    print("\n=== 학습 완료 ===")


if __name__ == "__main__":
    main()


