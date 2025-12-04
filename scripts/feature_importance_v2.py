#!/usr/bin/env python3
"""
Two-Tower V2 (EmbeddingBagCollection) - 퍼뮤테이션 기반 변수 중요도 분석 스크립트 (AP 기반 + 시각화)

[데이터 파이프라인 개요]

DB 스키마:

1) Pair 테이블 (입찰-회사 매칭 관계)
   - 예: step1.notice_company_pairs 또는 schema.pair.table 로 정의된 테이블
   - 컬럼:
       · bidntceno  : 공고 번호
       · bidntceord : 공고 차수
       · bizno      : 회사 사업자번호

2) 전처리된 Feature 테이블 (핵심!)
   - Notice feature 테이블:   step1.notice_preprocessed
   - Company feature 테이블:  step1.company_preprocessed

   이 테이블들에는 "이미 전처리된" 수치/범주형 feature 가 들어있고,
   FeaturePreprocessor / build_feature_store 계층에서 이 테이블들을 읽어
   다음과 같은 구조의 feature_store 를 구성한다고 가정한다:

   notice_store = {
       "ids": [(bidntceno, bidntceord), ...],          # 공고 ID 튜플 리스트
       "dense_projected": np.ndarray [N_notice, Dd],   # 전처리/투영된 dense feature
       "categorical": np.ndarray [N_notice, Dc],       # 범주형 인덱스 테이블
       ...
   }

   company_store = {
       "ids": [bizno, ...],                            # 회사 ID 리스트
       "dense_projected": np.ndarray [N_company, Dd],
       "categorical": np.ndarray [N_company, Dc],
       ...
   }

이 스크립트는 train_v2.py 의 로직을 재사용하여:

1) create_pair_dataloaders(...) 를 통해
   - PairLoaderV2: (notice_idx, company_idx) 를 반환하는 DataLoader
   - metadata["notice_store"], metadata["company_store"] 로부터
     전처리된 feature 테이블을 GPU 상주 텐서로 로딩

2) 학습된 Two-Tower 모델 (NoticeTower + CompanyTower)을 체크포인트에서 로딩
   - full_model_final.pt 번들 ({"model": ..., "config": ...}) 대응

3) 원본 feature로 전체 데이터를 한 번 평가:
   - in-batch retrieval 가정 하의 average precision (AP, 사실상 MRR과 동일)

4) 특정 범주형 feature 하나를 선택해서:
   - 해당 feature column만 전체 row 기준으로 무작위 셔플 (permutation)
   - 같은 방식으로 AP를 다시 계산
   - baseline 대비 AP 감소량 ΔAP = base_ap - shuffled_ap 을 "변수 중요도"로 해석
   - 이를 num_repeats 번 반복해서 평균/표준편차 계산

5) 선택된 모든 feature 에 대해 반복:
   - 결과를 CSV로 저장
   - 상위 K개 feature 에 대해 bar chart (PNG) 시각화 생성
   - 전체 ΔAP 분포 히스토그램도 생성
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import pandas as pd
import matplotlib.pyplot as plt

# --- Project imports (train_v2.py 와 동일한 경로 사용) ---------------------------
from database.database_connector import DatabaseConnector
from preprocess.torchrec.schema import build_torchrec_schema_from_meta
from src.towers.pairs.pair_loader import create_pair_dataloaders
from src.towers.tower import NoticeTower, CompanyTower
from src.towers.kjt_utils import create_kjt_from_batch_gpu
from src.towers.embedding_config import (
    get_notice_feature_metadata,
    get_company_feature_metadata,
)


# =====================================================================
# 0. TwoTowerWrapper (번들 언피클 + 실제 모델 둘 다에서 사용)
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

    중요:
    - train_v2.py 에서도 동일한 이름/구조로 정의되어 있고
      full_model_final.pt 번들 안에서도 이 클래스 이름이 기록되어 있음.
    - 여기서도 같은 이름으로 정의해 주면,
      torch.load(full_model_final.pt) 시 언피클러가
      scripts.feature_importance_v2.TwoTowerWrapper 를 찾을 수 있다.
    """

    def __init__(self, notice_tower=None, company_tower=None):
        super().__init__()
        self.notice_tower = notice_tower
        self.company_tower = company_tower

    def forward(self, batch):
        if self.notice_tower is None or self.company_tower is None:
            raise RuntimeError("TwoTowerWrapper 내부 타워가 초기화되지 않았습니다.")
        notice_emb = self.notice_tower(batch["notice"])
        company_emb = self.company_tower(batch["company"])
        return notice_emb, company_emb


# =====================================================================
# 1. Argument 파서
# =====================================================================
def parse_args():
    """
    커맨드 라인 인자 정의.

    - 체크포인트 / config.json / metadata.csv 경로
    - 사용할 pair 수 / batch 크기
    - target_table(notice/company) 및 셔플할 피처 목록
    - 반복 횟수(num_repeats)
    - 출력 CSV 및 시각화 옵션
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-Tower V2 퍼뮤테이션 기반 Feature Importance (EmbeddingBagCollection, AP + 시각화)"
    )

    # [필수] 학습된 모델 체크포인트
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="학습된 모델 체크포인트 경로 (full_model_final.pt 번들 권장)",
    )

    # [선택] config.json (없으면 체크포인트 디렉토리에서 자동 탐색)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="학습 시 저장된 config.json 경로 (기본: 체크포인트 디렉토리의 config.json 사용)",
    )

    # 메타데이터 경로 (train_v2.py 와 동일 기본값)
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="meta/metadata.csv",
        help="metadata.csv 경로 (기본: meta/metadata.csv)",
    )

    # 평가 데이터량 및 배치 설정
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="평가 시 배치 크기",
    )
    parser.add_argument(
        "--pair_limit",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=100000,
        help="분석에 사용할 최대 pair 수 (None 이면 전체 데이터 사용)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.0,
        help="create_pair_dataloaders 의 test_split (0이면 train 전체 사용)",
    )

    # 디바이스
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="사용할 디바이스 (cuda / cpu)",
    )
    parser.add_argument(
        "--enable_amp",
        action="store_true",
        default=False,
        help="평가 시 AMP(FP16/bfloat16) 사용 여부 (cuda 에서만 의미 있음)",
    )

    # 타겟 테이블 및 셔플할 피처
    parser.add_argument(
        "--target_table",
        type=str,
        default="notice",
        choices=["notice", "company"],
        help="어느 쪽 범주형 피처를 셔플할지 선택 (notice / company)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help=(
            "셔플할 피처 이름 리스트 (콤마 구분). "
            "예: 'dminsttcd,ntceinsttcd'. "
            "미지정 시 target_table 의 모든 categorical feature 사용"
        ),
    )

    # InfoNCE temperature (config.json 에 없으면 여기 값 사용)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="InfoNCE temperature (기본: config.json 의 값을 사용, 없으면 0.07)",
    )

    # 퍼뮤테이션 반복 횟수
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=10,
        help="각 feature 에 대해 셔플/평가를 반복할 횟수 (기본: 10)",
    )

    # 결과 저장 경로
    parser.add_argument(
        "--output",
        type=str,
        default="output/analysis/feature_importance_ap.csv",
        help="변수 중요도 결과 CSV 경로",
    )

    # 시각화 옵션
    parser.add_argument(
        "--plot_top_k",
        type=int,
        default=10,
        help="시각화 시 상위 몇 개 feature 를 그릴지 (기본: 10개)",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        default=False,
        help="셋하면 PNG 시각화 파일을 생성하지 않음",
    )

    return parser.parse_args()


# =====================================================================
# 2. Config / Dataloader / Model 빌드 유틸
# =====================================================================
def load_config(checkpoint_path: str, config_path: Optional[str] = None) -> Dict:
    """
    train_v2.py 에서 저장한 config.json 을 로드한다.

    - 기본 동작: checkpoint_path 의 상위 디렉토리에서 config.json 찾기
    - --config 명시 시 해당 경로 사용
    """
    ckpt_dir = Path(checkpoint_path).parent

    if config_path is None:
        config_path = ckpt_dir / "config.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"config.json을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"\n📋 학습 설정 로드 완료: {config_path}")
    return config


def build_dataloaders_and_feature_tables(
    db_engine,
    metadata_path: str,
    batch_size: int,
    pair_limit: Optional[int],
    test_split: float,
    device: torch.device,
):
    """
    DB에서 pair + feature store 를 로드하고,
    GPU 상주 dense / categorical 테이블을 만드는 함수.
    """
    # 1) TorchRec 스키마 생성 (metadata.csv 기반)
    schema = build_torchrec_schema_from_meta(
        pair_notice_id_cols=["bidntceno", "bidntceord"],  # 공고 키 컬럼
        pair_company_id_cols=["bizno"],                   # 회사 키 컬럼
        metadata_path=metadata_path,
    )

    print(f"\n📐 TorchRec 스키마 로드 완료")
    print(f"   Notice 피처: {len(schema.notice.categorical)}개 범주형, {len(schema.notice.numeric)}개 수치형")
    print(f"   Company 피처: {len(schema.company.categorical)}개 범주형, {len(schema.company.numeric)}개 수치형")

    # 2) PairLoaderV2 기반 DataLoader 생성
    print("\n🧵 DataLoader 생성 중... (PairLoaderV2 - EmbeddingBagCollection용)")
    train_loader, test_loader, metadata = create_pair_dataloaders(
        db_engine=db_engine,
        schema=schema,
        batch_size=batch_size,
        pair_limit=pair_limit,
        test_split=test_split,
        shuffle=False,            # 중요도 분석에서는 순서 고정
        shuffle_seed=42,
        num_workers=0,            # V2는 항상 0 (GPU gather)
        pin_memory=True,
        streaming=False,
        chunk_size=10000,
        feature_chunksize=10000,
        feature_limit=None,
        device=device,
        test_mode=False,          # pair_limit 에 해당하는 feature 모두 로딩
    )

    # 3) Feature store 를 GPU 텐서로 상주시킴
    print("\n📦 GPU 상주 Feature Tables 생성 중...")

    notice_store = metadata["notice_store"]
    company_store = metadata["company_store"]

    # 분석 스크립트에서는 dtype mismatch 방지를 위해 FP32로 상주
    notice_dense_table = torch.from_numpy(notice_store["dense_projected"]).to(
        device, dtype=torch.float32, non_blocking=True
    )
    company_dense_table = torch.from_numpy(company_store["dense_projected"]).to(
        device, dtype=torch.float32, non_blocking=True
    )

    # 범주형 인덱스 테이블 (int64)
    notice_cat_table = torch.from_numpy(notice_store["categorical"]).to(
        device, dtype=torch.long, non_blocking=True
    )
    company_cat_table = torch.from_numpy(company_store["categorical"]).to(
        device, dtype=torch.long, non_blocking=True
    )

    print(f"   Notice dense 테이블:  {notice_dense_table.shape} ({notice_dense_table.dtype})")
    print(f"   Company dense 테이블: {company_dense_table.shape} ({company_dense_table.dtype})")
    print(f"   Notice categorical:  {notice_cat_table.shape}")
    print(f"   Company categorical: {company_cat_table.shape}\n")

    return (
        train_loader,
        test_loader,
        schema,
        notice_dense_table,
        company_dense_table,
        notice_cat_table,
        company_cat_table,
    )


def build_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict,
    metadata_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    NoticeTower / CompanyTower 를 생성하고, 체크포인트의 가중치를 로드한다.

    - full_model_final.pt 번들 ({"model": ..., "config": ...}) 또는
      일반 checkpoint ({"model_state_dict": ...}) 모두 지원
    """
    categorical_embedding_dim = config.get("categorical_embedding_dim", 32)
    notice_dense_input_dim = config.get("notice_dense_input_dim", 256)
    company_dense_input_dim = config.get("company_dense_input_dim", 128)
    tower_hidden_dims = config.get("tower_hidden_dims", [256, 128])
    final_embedding_dim = config.get("final_embedding_dim", 128)
    dropout_rate = config.get("dropout_rate", 0.2)

    print("\n🧠 모델 생성 중...")
    print(f"   categorical_embedding_dim = {categorical_embedding_dim}")
    print(f"   notice_dense_input_dim    = {notice_dense_input_dim}")
    print(f"   company_dense_input_dim   = {company_dense_input_dim}")
    print(f"   tower_hidden_dims         = {tower_hidden_dims}")
    print(f"   final_embedding_dim       = {final_embedding_dim}")
    print(f"   dropout_rate              = {dropout_rate}")

    # (1) 새 모델 아키텍처 생성
    notice_tower = NoticeTower(
        metadata_path=metadata_path,
        categorical_embedding_dim=categorical_embedding_dim,
        dense_input_dim=notice_dense_input_dim,
        tower_hidden_dims=tower_hidden_dims,
        final_embedding_dim=final_embedding_dim,
        dropout_rate=dropout_rate,
        device=device,
        use_fp16=False,
    )

    company_tower = CompanyTower(
        metadata_path=metadata_path,
        categorical_embedding_dim=categorical_embedding_dim,
        dense_input_dim=company_dense_input_dim,
        tower_hidden_dims=tower_hidden_dims,
        final_embedding_dim=final_embedding_dim,
        dropout_rate=dropout_rate,
        device=device,
        use_fp16=False,
    )

    model = TwoTowerWrapper(notice_tower=notice_tower, company_tower=company_tower).to(device)

    # (2) 체크포인트 로드
    print(f"\n📂 체크포인트 로드: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = None

    # 1) full_model_final.pt 같이 {"model": ..., "config": ...} 형태
    if isinstance(ckpt, dict) and "model" in ckpt:
        print("   ▶ Detected bundle checkpoint with 'model' key")
        state_dict = ckpt["model"].state_dict()

    # 2) {"model_state_dict": ...} 형태 (best_model.pt / final_model.pt 등)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print("   ▶ Detected checkpoint with 'model_state_dict'")
        state_dict = ckpt["model_state_dict"]

    # 3) 그냥 state_dict 라고 가정
    elif isinstance(ckpt, dict):
        print("   ▶ Detected raw state_dict checkpoint")
        state_dict = ckpt

    else:
        raise ValueError(f"알 수 없는 체크포인트 형식입니다: type={type(ckpt)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing keys: {missing}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {unexpected}")
    print("✅ 체크포인트 로드 완료")

    model.eval()
    return model


# =====================================================================
# 3. AP 평가 유틸
# =====================================================================
@torch.no_grad()
def evaluate_avg_ap(
    model: torch.nn.Module,
    data_loader,
    notice_dense_table: torch.Tensor,
    company_dense_table: torch.Tensor,
    notice_cat_table: torch.Tensor,
    company_cat_table: torch.Tensor,
    notice_keys: List[str],
    company_keys: List[str],
    device: torch.device,
    temperature: float,
    enable_amp: bool = False,
) -> float:
    """
    test_loader 전체에 대해 average precision (AP) 평균 계산.

    - in-batch retrieval 가정 (각 notice i 의 정답 company는 i번째)
    - 단일 positive라 AP는 1/rank 형태 (실질적으로 MRR과 동일)
    """
    model.eval()

    total_ap = 0.0
    total_queries = 0
    batch_count = 0

    for batch in data_loader:
        batch_count += 1

        notice_idx_cpu, company_idx_cpu = batch
        notice_idx = notice_idx_cpu.to(device, non_blocking=True)
        company_idx = company_idx_cpu.to(device, non_blocking=True)

        # GPU 상주 테이블에서 gather (FP32 보장)
        notice_dense_b = torch.index_select(notice_dense_table, dim=0, index=notice_idx).float()
        company_dense_b = torch.index_select(company_dense_table, dim=0, index=company_idx).float()
        notice_cat_b = torch.index_select(notice_cat_table, dim=0, index=notice_idx)
        company_cat_b = torch.index_select(company_cat_table, dim=0, index=company_idx)

        # GPU-KJT 생성
        notice_kjt = create_kjt_from_batch_gpu(notice_cat_b, notice_keys, device)
        company_kjt = create_kjt_from_batch_gpu(company_cat_b, company_keys, device)

        batch_gpu = {
            "notice": {"dense": notice_dense_b, "kjt": notice_kjt},
            "company": {"dense": company_dense_b, "kjt": company_kjt},
        }

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(enable_amp and device.type == "cuda"),
        ):
            notice_emb, company_emb = model(batch_gpu)
            # [B, D] x [D, B] -> [B, B]
            sim_matrix = torch.mm(notice_emb, company_emb.t()) / temperature

        B = sim_matrix.size(0)
        if B == 0:
            continue

        target = torch.arange(B, device=device)  # 정답 인덱스: i -> company i

        # similarity 내림차순 정렬 후 rank 계산
        _, indices = sim_matrix.sort(dim=1, descending=True)  # [B, B]
        ranks = torch.empty_like(indices)
        ranks.scatter_(
            dim=1,
            index=indices,
            src=torch.arange(B, device=device).unsqueeze(0).expand_as(indices),
        )
        pos_ranks = ranks[torch.arange(B, device=device), target] + 1  # 1-based rank

        # AP (single positive) = 1 / rank
        ap_batch = 1.0 / pos_ranks.float()  # [B]
        total_ap += ap_batch.sum().item()
        total_queries += B

    avg_ap = total_ap / total_queries if total_queries > 0 else float("nan")
    print(f"[EVAL] avg AP = {avg_ap:.6f} ({batch_count} batches, {total_queries} queries)")
    return avg_ap


# =====================================================================
# 4. 시각화 유틸
# =====================================================================
def plot_feature_importance_bar(
    df: pd.DataFrame,
    metric_col: str,
    err_col: Optional[str],
    top_k: int,
    output_png: Path,
    title: str,
):
    """
    변수 중요도 결과 DataFrame 을 바 차트로 시각화하여 PNG 로 저장한다.

    Args:
        df: feature importance 결과 DataFrame
        metric_col: 시각화에 사용할 중요도 지표 컬럼명 (예: 'mean_delta_ap')
        err_col: 에러바(표준편차) 컬럼명 (예: 'std_delta_ap') 또는 None
        top_k: 상위 몇 개 feature 를 그릴지
        output_png: 저장할 PNG 파일 경로
        title: 그래프 제목
    """
    if df.empty:
        print(f"⚠️ 시각화 대상 데이터가 없습니다: {metric_col}")
        return

    df_sorted = df.sort_values(by=metric_col, ascending=False).head(top_k)

    x = range(len(df_sorted))
    y = df_sorted[metric_col].values
    yerr = df_sorted[err_col].values if err_col and err_col in df_sorted.columns else None

    plt.figure(figsize=(max(8, len(df_sorted) * 0.5), 6))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, df_sorted["feature"], rotation=45, ha="right")
    plt.tight_layout()
    plt.title(title)
    plt.ylabel(metric_col)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"📈 시각화 저장 완료 (bar): {output_png}")


def plot_delta_histogram(
    df: pd.DataFrame,
    delta_col: str,
    output_png: Path,
    title: str,
):
    """
    전체 ΔAP 분포 히스토그램을 저장한다.

    Args:
        df: feature importance 결과 DataFrame
        delta_col: 'all_delta_ap' 같이 리스트 형태가 들어있는 컬럼명
        output_png: 저장할 PNG 파일 경로
        title: 그래프 제목
    """
    if df.empty or delta_col not in df.columns:
        print(f"⚠️ 히스토그램 대상 데이터가 없습니다: {delta_col}")
        return

    # all_delta_ap 컬럼은 리스트이므로 모두 펼쳐서 1D 리스트로 만듦
    all_deltas = []
    for lst in df[delta_col]:
        if isinstance(lst, (list, tuple)):
            all_deltas.extend(lst)

    if not all_deltas:
        print("⚠️ ΔAP 값이 비어 있어 히스토그램을 그릴 수 없습니다.")
        return

    plt.figure(figsize=(7, 5))
    plt.hist(all_deltas, bins=20)
    plt.tight_layout()
    plt.title(title)
    plt.xlabel("ΔAP (base_ap - shuffled_ap)")
    plt.ylabel("count")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"📈 시각화 저장 완료 (hist): {output_png}")


# =====================================================================
# 5. 메인 로직
# =====================================================================
def main():
    args = parse_args()

    # 디바이스 설정
    device = torch.device(
        "cuda:0" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"\n🖥️  사용 디바이스: {device}")

    # 1) config 로드 및 temperature 설정
    config = load_config(args.checkpoint, args.config)
    temperature = args.temperature if args.temperature is not None else config.get(
        "temperature", 0.07
    )
    print(f"🔥 사용 temperature = {temperature}")

    # 2) DB 연결 및 DataLoader + Feature 테이블 로딩
    print("\n🔌 DB 연결 중...")
    db = DatabaseConnector()
    engine = db.engine
    print("✅ DB 연결 완료")

    (
        train_loader,
        test_loader,
        schema,
        notice_dense_table,
        company_dense_table,
        notice_cat_table,
        company_cat_table,
    ) = build_dataloaders_and_feature_tables(
        db_engine=engine,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        pair_limit=args.pair_limit,
        test_split=args.test_split,
        device=device,
    )

    # 평가에 사용할 loader 선택:
    eval_loader = test_loader if test_loader is not None else train_loader

    # 3) Feature metadata 로부터 categorical feature key 리스트 획득
    notice_meta = get_notice_feature_metadata(
        metadata_path=args.metadata_path,
        embedding_dim=config.get("categorical_embedding_dim", 32),
        add_unk_token=True,
    )
    company_meta = get_company_feature_metadata(
        metadata_path=args.metadata_path,
        embedding_dim=config.get("categorical_embedding_dim", 32),
        add_unk_token=True,
    )

    notice_keys = notice_meta["feature_names"]
    company_keys = company_meta["feature_names"]

    print(f"\n🔑 Notice categorical keys : {notice_keys}")
    print(f"🔑 Company categorical keys: {company_keys}")

    # 4) 모델 생성 + 체크포인트 로드
    model = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        metadata_path=args.metadata_path,
        device=device,
    )

    # 5) 셔플 대상 feature 리스트 결정
    if args.target_table == "notice":
        all_keys = notice_keys
    else:
        all_keys = company_keys

    if args.features:
        target_features = [f.strip() for f in args.features.split(",") if f.strip()]
        for f in target_features:
            if f not in all_keys:
                raise ValueError(
                    f"지정한 피처 '{f}' 가 {args.target_table} 의 categorical feature 목록에 없습니다.\n"
                    f"유효한 값: {all_keys}"
                )
    else:
        # 전체 피처 대상
        target_features = list(all_keys)
        print(
            f"\n🎯 features 미지정 → {args.target_table} 의 모든 categorical feature 에 대해 중요도 계산:"
        )
        print(f"   {target_features}")

    # 6) Baseline 평가 (셔플 없이 원본 feature 그대로, AP 기준)
    print("\n============================================")
    print(" [1/2] Baseline 평가 (원본 feature, 셔플 없음, AP)")
    print("============================================")

    base_ap = evaluate_avg_ap(
        model=model,
        data_loader=eval_loader,
        notice_dense_table=notice_dense_table,
        company_dense_table=company_dense_table,
        notice_cat_table=notice_cat_table,
        company_cat_table=company_cat_table,
        notice_keys=notice_keys,
        company_keys=company_keys,
        device=device,
        temperature=temperature,
        enable_amp=args.enable_amp,
    )

    print(f"\n📊 Baseline 결과:")
    print(f"   - avg AP = {base_ap:.6f}")

    # 7) Feature 별 퍼뮤테이션 중요도 계산 (AP 기준, num_repeats 반복)
    print("\n============================================")
    print(" [2/2] Feature 별 퍼뮤테이션 중요도 계산 시작 (AP 기반)")
    print("============================================")

    results = []

    for feat in target_features:
        print(f"\n--- 🔀 Feature 셔플: {args.target_table}.{feat} ---")

        if args.target_table == "notice":
            cat_table_base = notice_cat_table
            num_rows, _ = cat_table_base.shape
            col_idx = all_keys.index(feat)
        else:
            cat_table_base = company_cat_table
            num_rows, _ = cat_table_base.shape
            col_idx = all_keys.index(feat)

        deltas = []
        shuffled_aps = []

        for r in range(args.num_repeats):
            print(f"   · repeat {r+1}/{args.num_repeats} ...", end="", flush=True)

            cat_table_shuffled = cat_table_base.clone()
            col = cat_table_shuffled[:, col_idx]

            perm = torch.randperm(num_rows, device=device)
            col_shuffled = col[perm]
            cat_table_shuffled[:, col_idx] = col_shuffled

            if args.target_table == "notice":
                notice_cat_current = cat_table_shuffled
                company_cat_current = company_cat_table
            else:
                notice_cat_current = notice_cat_table
                company_cat_current = cat_table_shuffled

            shuffle_ap = evaluate_avg_ap(
                model=model,
                data_loader=eval_loader,
                notice_dense_table=notice_dense_table,
                company_dense_table=company_dense_table,
                notice_cat_table=notice_cat_current,
                company_cat_table=company_cat_current,
                notice_keys=notice_keys,
                company_keys=company_keys,
                device=device,
                temperature=temperature,
                enable_amp=args.enable_amp,
            )

            delta = float(base_ap - shuffle_ap)  # AP 감소량 (양수일수록 중요)
            deltas.append(delta)
            shuffled_aps.append(float(shuffle_ap))
            print(f" ΔAP = {delta:.6f}")

        if deltas:
            mean_delta = float(sum(deltas) / len(deltas))
            std_delta = float(torch.tensor(deltas).std(unbiased=False).item()) if len(deltas) > 1 else 0.0
        else:
            mean_delta, std_delta = float("nan"), 0.0

        print(f"   → mean ΔAP={mean_delta:.6f}, std={std_delta:.6f}")

        results.append(
            {
                "target_table": args.target_table,
                "feature": feat,
                "feature_index": col_idx,
                "base_ap": base_ap,
                "mean_delta_ap": mean_delta,
                "std_delta_ap": std_delta,
                "num_repeats": args.num_repeats,
                "all_delta_ap": deltas,
                "all_shuffled_ap": shuffled_aps,
            }
        )

    # 8) 결과 DataFrame 정리 및 CSV 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    if not df.empty:
        df.sort_values(by="mean_delta_ap", ascending=False, inplace=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Feature importance(AP) 결과 CSV 저장: {output_path}")

    print(f"\n🔝 상위 {min(10, len(df))}개 feature (mean_delta_ap 기준):")
    if not df.empty:
        print(df.head(10).to_string(index=False))
    else:
        print("   (데이터 없음)")

    # 9) 시각화 (bar + histogram)
    if not args.no_plots and not df.empty:
        print("\n📊 시각화 생성 중...")

        # (1) mean_delta_ap 기준 상위 K개 bar chart (에러바 = std_delta_ap)
        png_bar = output_path.with_suffix(".mean_delta_ap.png")
        plot_feature_importance_bar(
            df=df,
            metric_col="mean_delta_ap",
            err_col="std_delta_ap",
            top_k=args.plot_top_k,
            output_png=png_bar,
            title=f"{args.target_table} - Feature Importance (ΔAP)",
        )

        # (2) 전체 ΔAP 분포 히스토그램
        png_hist = output_path.with_suffix(".delta_ap_hist.png")
        plot_delta_histogram(
            df=df,
            delta_col="all_delta_ap",
            output_png=png_hist,
            title=f"{args.target_table} - ΔAP Distribution",
        )
    elif args.no_plots:
        print("\n(옵션에 의해 PNG 시각화는 생성하지 않음: --no_plots)")
    else:
        print("\n(시각화 대상 데이터 없음)")

    # 10) DB 커넥션 정리
    db.close()
    print("\n🎉 변수 중요도(AP 기반) 분석 + 시각화 완료!")


if __name__ == "__main__":
    main()
