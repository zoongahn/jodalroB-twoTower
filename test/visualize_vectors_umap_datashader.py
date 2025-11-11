#!/usr/bin/env python3
"""
Datashader 기반 대용량 벡터 DB UMAP 시각화

Datashader를 사용하여 수백만 개의 임베딩을 빠르게 시각화합니다.
래스터 기반 렌더링으로 브라우저 부하 없이 즉시 표시됩니다.

주요 기능:
1. 대용량 데이터(2.1M+) 빠른 렌더링
2. Datashader 래스터 이미지 생성
3. 정적 이미지(PNG) 및 대화형 HTML
4. 컬러맵으로 밀도 시각화

사용법:
    # Notice 전체 2D 시각화
    python test/visualize_vectors_umap_datashader.py --db output/vector/embeddings_v1 --type notice

    # Company 전체 시각화
    python test/visualize_vectors_umap_datashader.py --db output/vector/embeddings_v1 --type company

    # 둘 다 시각화
    python test/visualize_vectors_umap_datashader.py --db output/vector/embeddings_v1 --type both
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorize import EmbeddingStore

# Check for required packages
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
        print("⚠️  Using CPU UMAP (install cuml-cu11 for GPU acceleration)")
    except ImportError:
        print("❌ UMAP not installed. Install with: pip install umap-learn")
        sys.exit(1)

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import export_image
    from datashader.colors import inferno, viridis
except ImportError:
    print("❌ Datashader not installed. Install with: pip install datashader")
    sys.exit(1)

try:
    import holoviews as hv
    from holoviews.operation.datashader import datashade, dynspread
    hv.extension('bokeh')
    HOLOVIEWS_AVAILABLE = True
except ImportError:
    print("⚠️  Holoviews not available (optional). Install with: pip install holoviews bokeh")
    HOLOVIEWS_AVAILABLE = False


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Datashader 기반 대용량 벡터 DB UMAP 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # Notice 전체 시각화 (2.1M)
  python test/visualize_vectors_umap_datashader.py --db output/vector/embeddings_v1 --type notice

  # Company 전체 시각화 (800K)
  python test/visualize_vectors_umap_datashader.py --db output/vector/embeddings_v1 --type company

  # 둘 다 시각화
  python test/visualize_vectors_umap_datashader.py --db output/vector/embeddings_v1 --type both
        """
    )

    # 필수 인자
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="벡터 DB 경로"
    )

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
        help="최대 샘플 수 (None=전체)"
    )

    # UMAP 파라미터
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (기본값: 15)"
    )

    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (기본값: 0.1)"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="UMAP metric (기본값: cosine)"
    )

    # 시각화 옵션
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="이미지 너비 (기본값: 1200)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="이미지 높이 (기본값: 900)"
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        choices=["viridis", "inferno", "plasma", "magma"],
        help="컬러맵 (기본값: viridis)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 경로 (기본값: 자동 생성)"
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

    print(f"\n  ⚠️  샘플링: {len(embeddings):,} → {max_samples:,}")
    np.random.seed(seed)
    indices = np.random.choice(len(embeddings), max_samples, replace=False)
    indices = sorted(indices)
    return embeddings[indices], [ids[i] for i in indices]


def apply_umap(
    embeddings: np.ndarray,
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
    print(f"  출력: 2D")
    print(f"  n_neighbors: {n_neighbors}, min_dist: {min_dist}, metric: {metric}")

    if USE_GPU:
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            verbose=True
        )
        print("\n  🚀 GPU UMAP 실행 중...")
        reduced = reducer.fit_transform(embeddings)
        if hasattr(reduced, 'get'):
            reduced = reduced.get()
    else:
        reducer = UMAP(
            n_components=2,
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


def create_datashader_image(
    df: pd.DataFrame,
    width: int,
    height: int,
    cmap: str,
    output_path: Path
):
    """Datashader로 이미지 생성"""
    print(f"\n{'='*80}")
    print(f"Datashader 렌더링")
    print(f"{'='*80}")
    print(f"  데이터: {len(df):,} 포인트")
    print(f"  해상도: {width}x{height}")
    print(f"  컬러맵: {cmap}")

    # Create canvas
    canvas = ds.Canvas(plot_width=width, plot_height=height)

    # Aggregate points
    print("\n  집계 중...")
    if 'category' in df.columns:
        agg = canvas.points(df, 'x', 'y', ds.count_cat('category'))
    else:
        agg = canvas.points(df, 'x', 'y')

    # Render image
    print("  렌더링 중...")
    cmap_dict = {
        'viridis': viridis,
        'inferno': inferno,
        'plasma': ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],
        'magma': ['#000004', '#721f81', '#b73779', '#f1605d', '#fcfdbf']
    }

    img = tf.shade(agg, cmap=cmap_dict.get(cmap, viridis))
    img = tf.set_background(img, "black")

    # Save PNG
    png_path = output_path.with_suffix('.png')
    export_image(img, str(png_path.stem), export_path=str(png_path.parent))

    print(f"\n✓ PNG 저장: {png_path}")
    print(f"  크기: {png_path.stat().st_size / 1024 / 1024:.1f} MB")

    return img


def create_interactive_html(
    df: pd.DataFrame,
    width: int,
    height: int,
    cmap: str,
    output_path: Path
):
    """대화형 HTML 생성 (Holoviews)"""
    if not HOLOVIEWS_AVAILABLE:
        print("⚠️  Holoviews 없음 - HTML 생략")
        return

    print(f"\n{'='*80}")
    print(f"대화형 HTML 생성")
    print(f"{'='*80}")

    # Create holoviews points
    points = hv.Points(df, kdims=['x', 'y'])

    # Apply datashader
    shaded = datashade(points, cmap=cmap, width=width, height=height)
    shaded = dynspread(shaded, threshold=0.5, max_px=3)

    # Save HTML
    html_path = output_path.with_suffix('.html')
    hv.save(shaded, html_path, backend='bokeh')

    print(f"✓ HTML 저장: {html_path}")
    print(f"  크기: {html_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    args = parse_args()

    print("=" * 80)
    print("Datashader 기반 대용량 벡터 DB UMAP 시각화")
    print("=" * 80)
    print(f"DB: {args.db}")
    print(f"타입: {args.type}")
    print(f"샘플: {args.max_samples if args.max_samples else '전체'}")

    # Load vector DB
    print("\n벡터 DB 로딩...")
    store = EmbeddingStore()
    store.load(args.db)
    print("✓ 로딩 완료")

    all_data = []

    # Notice 처리
    if args.type in ["notice", "both"]:
        embeddings, ids = extract_all_embeddings(store, "notice")
        embeddings, ids = sample_if_needed(embeddings, ids, args.max_samples, args.seed)
        reduced = apply_umap(embeddings, args.n_neighbors, args.min_dist, args.metric, args.seed)

        df = pd.DataFrame({'x': reduced[:, 0], 'y': reduced[:, 1], 'category': 'Notice'})
        all_data.append(df)

    # Company 처리
    if args.type in ["company", "both"]:
        embeddings, ids = extract_all_embeddings(store, "company")
        embeddings, ids = sample_if_needed(embeddings, ids, args.max_samples, args.seed)
        reduced = apply_umap(embeddings, args.n_neighbors, args.min_dist, args.metric, args.seed)

        df = pd.DataFrame({'x': reduced[:, 0], 'y': reduced[:, 1], 'category': 'Company'})
        all_data.append(df)

    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert category to categorical type
    if 'category' in combined_df.columns:
        combined_df['category'] = combined_df['category'].astype('category')

    # Auto-generate output path
    if args.output is None:
        suffix = f"{args.max_samples}" if args.max_samples else "full"
        args.output = f"output/visualizations/umap_{args.type}_datashader_{suffix}"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    create_datashader_image(combined_df, args.width, args.height, args.cmap, output_path)

    if HOLOVIEWS_AVAILABLE:
        create_interactive_html(combined_df, args.width, args.height, args.cmap, output_path)

    print("\n" + "=" * 80)
    print("✓ 완료!")
    print("=" * 80)
    print(f"\n생성된 파일:")
    print(f"  PNG: {output_path.with_suffix('.png')}")
    if HOLOVIEWS_AVAILABLE:
        print(f"  HTML: {output_path.with_suffix('.html')}")


if __name__ == "__main__":
    main()
