#!/usr/bin/env python3
"""
공통 학습 로직 모듈
train.py와 train_for_optuna.py에서 공유하는 함수들
"""

import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

# Project imports
from data.database_connector import DatabaseConnector
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders
from src.towers.two_tower_train_task import create_two_tower_train_task
from src.evaluation.evaluator import TwoTowerEvaluator


def setup_cuda_optimizations(enable_tf32=True, enable_cudnn_benchmark=True):
    """
    CUDA 가속 최적화 설정

    Args:
        enable_tf32: TF32 활성화 여부
        enable_cudnn_benchmark: cuDNN benchmark 활성화 여부
    """
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


def build_schema(metadata_path="meta/metadata.csv"):
    """
    TorchRec 스키마 구축

    Args:
        metadata_path: 메타데이터 CSV 파일 경로

    Returns:
        schema: TorchRec 스키마 객체
    """
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": metadata_path
    }
    schema = build_torchrec_schema_from_meta(**schema_config)
    return schema


def build_dataloaders(db_engine, schema, config):
    """
    데이터로더 생성 로직

    Args:
        db_engine: 데이터베이스 엔진
        schema: TorchRec 스키마
        config: 데이터로더 설정 딕셔너리

    Returns:
        train_loader, test_loader
    """
    train_loader, test_loader = create_unified_bid_dataloaders(
        db_engine=db_engine,
        schema=schema,
        batch_size=config["batch_size"],
        test_split=config.get("test_split", 0.2),
        shuffle_seed=config.get("shuffle_seed", 42),
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
        streaming=config.get("streaming", False),
        load_all_features=config.get("load_all_features", True),
        chunk_size=config.get("chunk_size", 1000000),
        feature_chunksize=config.get("feature_chunksize", 1000),
        use_preprocessor=config.get("use_preprocessor", True),
        test_mode=config.get("test_mode", True),
        pair_limit=config.get("pair_limit", 5000000),
    )
    return train_loader, test_loader


def build_model(schema, config, device):
    """
    모델 생성 로직

    Args:
        schema: TorchRec 스키마
        config: 모델 설정 딕셔너리
        device: 디바이스 (cuda/cpu)

    Returns:
        train_task: Two-Tower 학습 태스크
    """
    notice_categorical_keys = schema.notice.categorical
    company_categorical_keys = schema.company.categorical

    train_task = create_two_tower_train_task(
        notice_categorical_keys=notice_categorical_keys,
        company_categorical_keys=company_categorical_keys,
        metadata_path=config.get("metadata_path", "meta/metadata.csv"),
        categorical_embedding_dim=config.get("categorical_embedding_dim", 32),
        notice_dense_input_dim=config.get("notice_dense_input_dim", 256),
        company_dense_input_dim=config.get("company_dense_input_dim", 128),
        tower_hidden_dims=config.get("tower_hidden_dims", [512, 256]),
        final_embedding_dim=config.get("final_embedding_dim", 128),
        dropout_rate=config.get("dropout_rate", 0.1),
        temperature=config.get("temperature", 1.0),
        loss_type=config.get("loss_type", "cross_entropy"),
        device=device
    )

    # torch.compile 최적화 (선택적)
    if config.get("enable_torch_compile", False):
        compile_mode = config.get("compile_mode", "reduce-overhead")
        train_task = torch.compile(train_task, mode=compile_mode, fullgraph=False)

    return train_task


def create_optimizer_and_scheduler(train_task, train_loader, config):
    """
    옵티마이저 및 스케줄러 생성

    Args:
        train_task: 학습 태스크
        train_loader: 학습 데이터로더
        config: 학습 설정 딕셔너리

    Returns:
        optimizer, scheduler
    """
    optimizer = optim.Adam(
        train_task.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5)
    )

    # Learning Rate Warmup 스케줄러
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = max(1, int(len(train_loader) * config.get("warmup_ratio", 0.05)))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return optimizer, scheduler


def create_async_transfer_fn(device):
    """
    비동기 GPU 전송 함수 생성

    Args:
        device: 디바이스 (cuda/cpu)

    Returns:
        transfer_fn: 비동기 전송 함수
    """
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

    return _to_device_async, prefetch_stream


def train_one_epoch(train_task, train_loader, optimizer, scheduler, device, config, verbose=True):
    """
    1 에포크 학습

    Args:
        train_task: 학습 태스크
        train_loader: 학습 데이터로더
        optimizer: 옵티마이저
        scheduler: 학습률 스케줄러
        device: 디바이스
        config: 학습 설정 딕셔너리
        verbose: 출력 여부

    Returns:
        avg_loss, avg_acc
    """
    train_task.train()
    train_losses = []
    train_accuracies = []

    # 비동기 전송 함수 생성
    to_device_async, prefetch_stream = create_async_transfer_fn(device)

    # Progress bar 설정
    if verbose:
        train_pbar = tqdm(
            train_loader,
            desc="Training",
            unit="batch",
            position=0,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}',
            ncols=100,
            mininterval=0.5,
            leave=False
        )
        info_bar = tqdm(
            total=0,
            position=1,
            bar_format='{desc}',
            ncols=100,
            leave=False
        )
        data_iter = train_pbar
    else:
        data_iter = train_loader

    step_count = 0
    log_interval = config.get("log_interval", 20)

    for batch in data_iter:
        # 비동기 GPU 전송
        batch = to_device_async(batch, device)
        if torch.cuda.is_available():
            torch.cuda.current_stream().wait_stream(prefetch_stream)

        # Forward pass
        optimizer.zero_grad()
        result = train_task(batch, return_metrics=True)

        loss = result["loss"]
        accuracy = result["accuracy"]

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        step_count += 1

        # 메트릭 저장
        train_losses.append(loss.item())
        train_accuracies.append(accuracy.item())

        # Progress bar 업데이트
        if verbose and step_count % log_interval == 0:
            loss_val = loss.item()
            accuracy_val = accuracy.item()
            pos_sim = result.get("positive_similarity_mean", torch.tensor(0.0)).item()
            neg_sim = result.get("negative_similarity_mean", torch.tensor(0.0)).item()
            sim_gap = result.get("similarity_gap", torch.tensor(0.0)).item()
            z_gap = sim_gap / max(abs(neg_sim) + 1e-8, 1e-8)

            info_str = f"📊 Loss: {loss_val:.3f} | Acc: {accuracy_val:.3f} | Pos: {pos_sim:.3f} | Neg: {neg_sim:.3f} | Z-gap: {z_gap:.2f} | Batch: {config['batch_size']}"
            info_bar.set_description_str(info_str)

    # 진행바 종료
    if verbose:
        train_pbar.close()
        info_bar.close()

    avg_loss = sum(train_losses) / len(train_losses)
    avg_acc = sum(train_accuracies) / len(train_accuracies)

    return avg_loss, avg_acc


def validate(train_task, test_loader, device, config, verbose=True):
    """
    검증 수행

    Args:
        train_task: 학습 태스크
        test_loader: 검증 데이터로더
        device: 디바이스
        config: 설정 딕셔너리
        verbose: 출력 여부

    Returns:
        avg_loss, avg_acc
    """
    if test_loader is None:
        return None, None

    train_task.eval()
    val_losses = []
    val_accuracies = []

    # Progress bar 설정
    if verbose:
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
        data_iter = val_pbar
    else:
        data_iter = test_loader

    with torch.no_grad():
        for batch in data_iter:
            result = train_task(batch, return_metrics=True)

            loss = result["loss"]
            accuracy = result["accuracy"]

            val_losses.append(loss.item())
            val_accuracies.append(accuracy.item())

            # Validation 정보 표시
            if verbose:
                val_loss = loss.item()
                val_acc = accuracy.item()
                val_pos_sim = result.get("positive_similarity_mean", torch.tensor(0.0)).item()
                val_neg_sim = result.get("negative_similarity_mean", torch.tensor(0.0)).item()
                val_sim_gap = result.get("similarity_gap", torch.tensor(0.0)).item()
                val_z_gap = val_sim_gap / max(abs(val_neg_sim) + 1e-8, 1e-8)

                info_str = f"🔍 Loss: {val_loss:.3f} | Acc: {val_acc:.3f} | Pos: {val_pos_sim:.3f} | Neg: {val_neg_sim:.3f} | Z-gap: {val_z_gap:.2f}"
                val_info_bar.set_description_str(info_str)

    # 진행바 종료
    if verbose:
        val_pbar.close()
        val_info_bar.close()

    avg_loss = sum(val_losses) / len(val_losses)
    avg_acc = sum(val_accuracies) / len(val_accuracies)

    return avg_loss, avg_acc


def evaluate_comprehensive(train_task, test_loader, device, verbose=True):
    """
    종합 평가 수행

    Args:
        train_task: 학습 태스크
        test_loader: 테스트 데이터로더
        device: 디바이스
        verbose: 출력 여부

    Returns:
        metrics: 평가 메트릭 딕셔너리
    """
    evaluator = TwoTowerEvaluator(device=device)

    if test_loader is not None:
        test_metrics = evaluator.evaluate_comprehensive(train_task, test_loader, verbose=verbose)
        return test_metrics
    else:
        return {}


def save_checkpoint(model, optimizer, epoch, loss, save_dir, is_best=False, is_final=False):
    """
    체크포인트 저장

    Args:
        model: 모델
        optimizer: 옵티마이저
        epoch: 에포크 번호
        loss: 손실값
        save_dir: 저장 디렉토리
        is_best: 최고 성능 모델 여부
        is_final: 최종 모델 여부
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # 최종 모델 저장
    if is_final:
        final_path = save_dir / 'final_model.pt'
        torch.save(checkpoint, final_path)

        model_only_path = save_dir / 'model_weights.pt'
        torch.save(model.state_dict(), model_only_path)
        print(f"최종 모델 저장: {final_path}")
        print(f"모델 가중치 저장: {model_only_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    체크포인트 불러오기

    Args:
        model: 모델
        optimizer: 옵티마이저
        checkpoint_path: 체크포인트 파일 경로

    Returns:
        start_epoch, last_loss
    """
    print(f"체크포인트 불러오는 중: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    last_loss = checkpoint['loss']

    print(f"Epoch {checkpoint['epoch']+1}부터 이어서 학습 시작")
    print(f"이전 손실: {last_loss:.4f}")

    return start_epoch, last_loss
