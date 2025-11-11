#!/usr/bin/env python3
"""
벡터 DB UMAP 시각화 스크립트

벡터 DB의 전체 Notice/Company 임베딩을 UMAP으로 2D/3D 차원 축소하여 시각화합니다.

주요 기능:
1. 전체 임베딩 UMAP 시각화 (기본: 전체 데이터)
2. 진행률 표시 및 배치 처리
3. 대화형 Plotly 시각화 (2D/3D)
4. HTML 파일로 저장

사용법:
    # Notice 전체 2D 시각화
    python scripts/visualize_vectors_umap.py --db output/vector/embeddings_v1 --type notice

    # Company 전체 3D 시각화
    python scripts/visualize_vectors_umap.py --db output/vector/embeddings_v1 --type company --n_components 3

    # 둘 다 시각화
    python scripts/visualize_vectors_umap.py --db output/vector/embeddings_v1 --type both

    # 샘플링 버전 (빠른 테스트)
    python scripts/visualize_vectors_umap.py --db output/vector/embeddings_v1 --type notice --max_samples 5000
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorize import EmbeddingStore

# Check for required packages and prefer GPU version
USE_GPU = False
try:
    from cuml.manifold import UMAP
    import cupy as cp
    USE_GPU = True
    print("✓ cuML (GPU) UMAP detected - using GPU acceleration")
except ImportError:
    try:
        import umap
        UMAP = umap.UMAP
        print("⚠️  Using CPU UMAP (install cuml-cu12 for GPU acceleration)")
    except ImportError:
        print("❌ UMAP not installed. Install with: pip install umap-learn")
        sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError:
    print("❌ Plotly not installed. Install with: pip install plotly")
    sys.exit(1)


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="벡터 DB UMAP 시각화 (전체 데이터)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # Notice 전체 2D 시각화
  python test/visualize_vectors_umap.py --db output/vector/embeddings_v1 --type notice

  # Company 전체 3D 시각화
  python test/visualize_vectors_umap.py --db output/vector/embeddings_v1 --type company --n_components 3

  # 샘플링 (빠른 테스트)
  python test/visualize_vectors_umap.py --db output/vector/embeddings_v1 --type both --max_samples 5000
        """
    )

    # 필수 인자
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="벡터 DB 경로 (접두사, 예: output/vector/embeddings_v1)"
    )

    # 시각화 옵션
    parser.add_argument(
        "--type",
        type=str,
        choices=["notice", "company", "both"],
        default="notice",
        help="시각화할 임베딩 타입 (기본값: notice)"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="최대 샘플 수 (None=전체 데이터 사용)"
    )

    parser.add_argument(
        "--n_components",
        type=int,
        choices=[2, 3],
        default=2,
        help="UMAP 차원 (2D 또는 3D, 기본값: 2)"
    )

    # UMAP 파라미터
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors 파라미터 (기본값: 15)"
    )

    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist 파라미터 (기본값: 0.1)"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="UMAP distance metric (기본값: cosine)"
    )

    # 출력 옵션
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 HTML 파일 경로 (기본값: 자동 생성)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42)"
    )

    return parser.parse_args()


def extract_all_embeddings(store: EmbeddingStore, index_type: str) -> Tuple[np.ndarray, list]:
    """Faiss 인덱스에서 전체 임베딩 추출"""
    if index_type == "notice":
        index = store.notice_index
        ids = store.notice_ids
        label = "Notice"
    else:
        index = store.company_index
        ids = store.company_ids
        label = "Company"

    n_total = index.ntotal
    dimension = store.dimension

    print(f"\n{'='*80}")
    print(f"{label} 임베딩 추출")
    print(f"{'='*80}")
    print(f"  총 개수: {n_total:,}")
    print(f"  차원: {dimension}")

    embeddings = np.zeros((n_total, dimension), dtype='float32')

    # 배치 단위로 추출
    batch_size = 10000
    for i in tqdm(range(0, n_total, batch_size), desc=f"  {label} 추출"):
        end_idx = min(i + batch_size, n_total)
        for j in range(i, end_idx):
            embeddings[j] = index.reconstruct(j)

    print(f"✓ 추출 완료: {embeddings.shape}")
    return embeddings, ids


def sample_if_needed(
    embeddings: np.ndarray,
    ids: list,
    max_samples: Optional[int],
    seed: int = 42
) -> Tuple[np.ndarray, list]:
    """필요시 샘플링"""
    if max_samples is None or max_samples >= len(embeddings):
        return embeddings, ids

    print(f"\n  ⚠️  샘플링 적용: {len(embeddings):,} → {max_samples:,}")
    np.random.seed(seed)
    indices = np.random.choice(len(embeddings), max_samples, replace=False)
    indices = sorted(indices)

    return embeddings[indices], [ids[i] for i in indices]


def apply_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42
) -> np.ndarray:
    """UMAP 차원 축소"""
    print(f"\n{'='*80}")
    print(f"UMAP 차원 축소 ({'GPU' if USE_GPU else 'CPU'})")
    print(f"{'='*80}")
    print(f"  입력: {embeddings.shape}")
    print(f"  출력: {n_components}D")
    print(f"  n_neighbors: {n_neighbors}, min_dist: {min_dist}, metric: {metric}")

    if USE_GPU:
        # GPU version (cuML)
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            verbose=True
        )
        print("\n  🚀 GPU UMAP 실행 중...")
        reduced = reducer.fit_transform(embeddings)
        # Convert cupy array to numpy
        if hasattr(reduced, 'get'):
            reduced = reduced.get()
    else:
        # CPU version (umap-learn)
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            verbose=True
        )
        print("\n  UMAP 실행 중...")
        reduced = reducer.fit_transform(embeddings)

    print(f"\n✓ UMAP 완료: {reduced.shape}")
    return reduced


def create_plot(
    embeddings: np.ndarray,
    ids: list,
    label: str,
    color: str,
    n_components: int
):
    """Scatter plot 생성"""
    n_samples = len(embeddings)

    # Hover text
    if n_samples > 10000:
        hover_text = [f"{label} #{i}" for i in range(n_samples)]
    else:
        hover_text = [f"{label}: {id}" for id in ids]

    if n_components == 2:
        return go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers',
            name=label,
            marker=dict(
                size=3 if n_samples > 10000 else 4,
                color=color,
                opacity=0.5 if n_samples > 10000 else 0.6,
                line=dict(width=0)
            ),
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        )
    else:
        return go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            name=label,
            marker=dict(
                size=2 if n_samples > 10000 else 3,
                color=color,
                opacity=0.4 if n_samples > 10000 else 0.5,
                line=dict(width=0)
            ),
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        )


def main():
    args = parse_args()

    print("=" * 80)
    print("벡터 DB UMAP 시각화")
    print("=" * 80)
    print(f"DB: {args.db}")
    print(f"타입: {args.type}")
    print(f"샘플: {args.max_samples if args.max_samples else '전체'}")
    print(f"차원: {args.n_components}D")

    # 경고
    if args.type == "both" and args.max_samples is None:
        print("\n⚠️  경고: 전체 데이터 시각화는 시간이 오래 걸립니다!")
        print("   샘플링 권장: --max_samples 10000")

    # Load vector DB
    print("\n벡터 DB 로딩...")
    store = EmbeddingStore()
    store.load(args.db)
    print("✓ 로딩 완료")

    traces = []

    # Notice 처리
    if args.type in ["notice", "both"]:
        embeddings, ids = extract_all_embeddings(store, "notice")
        embeddings, ids = sample_if_needed(embeddings, ids, args.max_samples, args.seed)
        reduced = apply_umap(embeddings, args.n_components, args.n_neighbors,
                           args.min_dist, args.metric, args.seed)
        traces.append(create_plot(reduced, ids, "Notice", "#1f77b4", args.n_components))

    # Company 처리
    if args.type in ["company", "both"]:
        embeddings, ids = extract_all_embeddings(store, "company")
        embeddings, ids = sample_if_needed(embeddings, ids, args.max_samples, args.seed)
        reduced = apply_umap(embeddings, args.n_components, args.n_neighbors,
                           args.min_dist, args.metric, args.seed)
        traces.append(create_plot(reduced, ids, "Company", "#ff7f0e", args.n_components))

    # Create figure
    print(f"\n{'='*80}")
    print("Plotly 시각화 생성")
    print(f"{'='*80}")

    fig = go.Figure(data=traces)

    if args.n_components == 2:
        fig.update_layout(
            title=f"UMAP 2D - {args.type.capitalize()} Embeddings",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode='closest',
            width=1400,
            height=900,
            template="plotly_white"
        )
    else:
        fig.update_layout(
            title=f"UMAP 3D - {args.type.capitalize()} Embeddings",
            scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
            hovermode='closest',
            width=1400,
            height=900,
            template="plotly_white"
        )

    # Save
    if args.output is None:
        suffix = f"{args.max_samples}" if args.max_samples else "full"
        args.output = f"output/visualizations/umap_{args.type}_{args.n_components}d_{suffix}.html"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))

    print(f"\n✓ 저장: {output_path}")
    print(f"  크기: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 80)
    print("✓ 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
