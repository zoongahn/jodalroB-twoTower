"""
임베딩 분석 스크립트: 벡터 내부값 보기 + 시각화 + 품질점검

사용법:
    python test/analyze_embeddings.py

기능:
    1. Faiss 인덱스에서 벡터 추출
    2. 양성/음성 코사인 유사도 분포 비교
    3. Recall@k 계산
    4. UMAP 2D 시각화 (Plotly)
    5. 이웃 탐색 예시
"""
import sys
import os
import numpy as np
import faiss

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vectorize import EmbeddingStore


# ============================================================================
# 1) Faiss 인덱스에서 벡터 꺼내기
# ============================================================================

def describe(index):
    """인덱스 정보 출력"""
    print(f"Index type: {type(index).__name__}")
    print(f"  Dimension: {index.d}")
    print(f"  Total vectors: {index.ntotal}")


def get_vectors_cpu(index):
    """
    인덱스에 저장된 원시 벡터를 numpy로 꺼낸다.
    - Flat 계열: index.xb (float32, [N, d])
    - IVF/HNSW 등: reconstruct(_batch) 지원되면 그걸 사용
    - IDMap이면 ids도 함께

    Returns:
        xb: np.ndarray [N, d], 벡터들
        ids: np.ndarray [N] or None, ID 매핑이 있으면
    """
    index = faiss.downcast_index(index)

    # 1) IDMap 여부
    ids = None
    if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        # 원본 인덱스
        base = faiss.downcast_index(index.index)
        ids = faiss.vector_to_array(index.id_map).reshape(-1)
        index = base

    # 2) Flat 계열
    if hasattr(index, "xb"):  # IndexFlatIP, IndexFlatL2
        xb = faiss.rev_swig_ptr(
            index.xb,
            index.ntotal * index.d
        ).reshape(index.ntotal, index.d).copy()
        return xb, ids

    # 3) reconstruct 지원 (IVF/HNSW 등)
    if hasattr(index, "reconstruct_batch"):
        I = list(range(index.ntotal))
        xb = index.reconstruct_batch(I)  # np.float32 [N, d]
        return xb, ids

    if hasattr(index, "reconstruct"):
        xb = [index.reconstruct(i) for i in range(index.ntotal)]
        xb = np.vstack(xb).astype("float32")
        return xb, ids

    raise RuntimeError(
        "이 인덱스는 원시 벡터를 직접 노출하지 않습니다. "
        "인덱스 빌드 시 별도 보관을 권장합니다."
    )


# ============================================================================
# 2) 양성/음성 분포 + Recall@k
# ============================================================================

def analyze_quality(X, labels, sample_size=5000):
    """
    양성/음성 코사인 유사도 분포 비교 및 Recall@k 계산

    Args:
        X: np.ndarray [N, d], 벡터들
        labels: np.ndarray [N], 각 벡터의 라벨 (같은 라벨 = positive pair)
        sample_size: 샘플링할 쿼리 개수
    """
    print("\n" + "=" * 60)
    print("품질 분석: 양성/음성 유사도 분포 & Recall@k")
    print("=" * 60)

    # 코사인 정규화
    Xn = X / np.maximum(1e-12, np.linalg.norm(X, axis=1, keepdims=True))

    # 샘플링
    rng = np.random.default_rng(0)
    n_sample = min(sample_size, len(Xn))
    idx = rng.choice(len(Xn), size=n_sample, replace=False)
    A = Xn[idx]  # [M, d]
    S = A @ Xn.T  # [M, N], cosine similarity

    lab = labels[idx]
    pos_scores, neg_scores = [], []

    for i, ql in enumerate(lab):
        # 같은 라벨(자기 자신 포함) = positive
        mask_pos = (labels == ql)
        pos = S[i, mask_pos]
        neg = S[i, ~mask_pos]

        # 자기 자신 제거
        pos = pos[pos < 0.9999] if pos.size > 1 else pos
        if pos.size:
            pos_scores.extend(pos.tolist())
        if neg.size:
            neg_scores.extend(
                rng.choice(neg, size=min(100, neg.size), replace=False)
            )

    pos_mean = np.mean(pos_scores) if pos_scores else 0.0
    neg_mean = np.mean(neg_scores) if neg_scores else 0.0

    print(f"\n코사인 유사도 분포 (샘플 {n_sample}개):")
    print(f"  Positive mean: {pos_mean:.4f} (std: {np.std(pos_scores):.4f})")
    print(f"  Negative mean: {neg_mean:.4f} (std: {np.std(neg_scores):.4f})")
    print(f"  차이 (Pos - Neg): {pos_mean - neg_mean:.4f}")

    if pos_mean - neg_mean > 0.2:
        print("  ✓ 양성/음성 분리가 양호합니다 (차이 > 0.2)")
    elif pos_mean - neg_mean > 0.1:
        print("  △ 양성/음성 분리가 보통입니다 (0.1 < 차이 < 0.2)")
    else:
        print("  ✗ 양성/음성 분리가 약합니다 (차이 < 0.1)")

    # Recall@k 계산
    print("\nRecall@k 계산:")
    for k in [1, 5, 10, 20]:
        index_search = faiss.IndexFlatIP(Xn.shape[1])
        index_search.add(Xn)  # 코사인 = 정규화 + 내적
        D, I = index_search.search(A, k)  # [M, k]

        hit = 0
        for i, ql in enumerate(lab):
            if np.any(labels[I[i]] == ql):
                hit += 1

        recall = hit / len(lab)
        print(f"  Recall@{k:2d}: {recall:.4f}")

    return pos_scores, neg_scores


# ============================================================================
# 3) 이웃 탐색/점검
# ============================================================================

def show_neighbors(query_idx, Xn, ids, labels, topk=10):
    """
    특정 벡터의 이웃들을 출력

    Args:
        query_idx: 쿼리 벡터 인덱스
        Xn: 정규화된 벡터들 [N, d]
        ids: 벡터의 ID 리스트
        labels: 벡터의 라벨 리스트
        topk: 출력할 이웃 개수
    """
    q = Xn[query_idx:query_idx+1]
    sims = (q @ Xn.T).ravel()
    nn = np.argsort(-sims)[:topk+1]  # 자기 자신 포함

    print(f"\n쿼리: idx={query_idx}, id={ids[query_idx]}, label={labels[query_idx]}")
    print("Top neighbors:")
    for j in nn:
        marker = "★" if labels[j] == labels[query_idx] else " "
        print(f"  {marker} idx={j:6d}  id={ids[j]:15s}  sim={sims[j]:.4f}  label={labels[j]}")


# ============================================================================
# 4) UMAP 2D 시각화
# ============================================================================

def visualize_umap(X, labels, ids, max_samples=10000, output_path=None):
    """
    UMAP으로 2D 투영 후 Plotly로 시각화

    Args:
        X: np.ndarray [N, d]
        labels: 라벨 배열
        ids: ID 배열
        max_samples: 최대 샘플 개수 (속도/메모리)
        output_path: HTML 저장 경로 (선택)
    """
    try:
        import umap
        import plotly.express as px
    except ImportError as e:
        print(f"\nUMAP 시각화를 위해 추가 패키지가 필요합니다:")
        print("  pip install umap-learn plotly")
        return

    print("\n" + "=" * 60)
    print("UMAP 2D 시각화")
    print("=" * 60)

    # 샘플링
    n = min(max_samples, len(X))
    rng = np.random.default_rng(1)
    idx = rng.choice(len(X), size=n, replace=False)
    X_sample = X[idx]
    y_sample = labels[idx]
    id_sample = ids[idx]

    print(f"\n샘플 개수: {n} / {len(X)}")

    # 정규화
    X2 = X_sample / np.maximum(1e-12, np.linalg.norm(X_sample, axis=1, keepdims=True))

    # UMAP
    print("UMAP 투영 중...")
    U = umap.UMAP(
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42
    ).fit_transform(X2)

    print(f"UMAP 결과 shape: {U.shape}")

    # Plotly scatter
    import pandas as pd
    df = pd.DataFrame({
        'x': U[:, 0],
        'y': U[:, 1],
        'label': y_sample.astype(str),
        'id': id_sample
    })

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='label',
        hover_data=['id'],
        title="Embedding UMAP (cosine)",
        width=1200,
        height=800
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))

    if output_path:
        fig.write_html(output_path)
        print(f"\n시각화 저장: {output_path}")

    fig.show()

    return U


# ============================================================================
# 5) 메인 실행
# ============================================================================

def main():
    print("=" * 60)
    print("임베딩 분석: 벡터 내부값 + 시각화 + 품질점검")
    print("=" * 60)

    # 설정
    EMBEDDING_PATH = "data/vector_db/embeddings"
    VISUALIZE = True  # UMAP 시각화 여부

    # Step 1: EmbeddingStore 로드
    print(f"\n[1] EmbeddingStore 로드: {EMBEDDING_PATH}_*")
    store = EmbeddingStore()
    store.load(EMBEDDING_PATH)
    print(store)

    # Step 2: 공고 임베딩 분석
    print("\n" + "=" * 60)
    print("공고(Notice) 임베딩 분석")
    print("=" * 60)

    describe(store.notice_index)
    X_notice, _ = get_vectors_cpu(store.notice_index)
    ids_notice = np.array(store.notice_ids)

    print(f"\n추출된 벡터:")
    print(f"  Shape: {X_notice.shape}")
    print(f"  Dtype: {X_notice.dtype}")
    print(f"  Mean: {X_notice.mean():.6f}, Std: {X_notice.std():.6f}")
    print(f"  Min: {X_notice.min():.6f}, Max: {X_notice.max():.6f}")

    # 샘플 벡터 출력
    print(f"\n샘플 벡터 (notice_0000):")
    print(f"  First 20 values: {X_notice[0, :20]}")
    print(f"  L2 norm: {np.linalg.norm(X_notice[0]):.6f}")

    # 라벨 생성 (더미: notice ID의 앞부분을 라벨로 사용)
    # 실제로는 positive pair 정보로부터 생성해야 함
    labels_notice = np.array([nid.split('_')[0] for nid in ids_notice])

    # 품질 분석
    pos_scores, neg_scores = analyze_quality(X_notice, labels_notice)

    # 이웃 탐색 예시
    print("\n" + "=" * 60)
    print("이웃 탐색 예시")
    print("=" * 60)
    Xn_notice = X_notice / np.maximum(1e-12, np.linalg.norm(X_notice, axis=1, keepdims=True))
    show_neighbors(0, Xn_notice, ids_notice, labels_notice, topk=10)
    show_neighbors(10, Xn_notice, ids_notice, labels_notice, topk=10)

    # Step 3: 업체 임베딩 분석
    print("\n\n" + "=" * 60)
    print("업체(Company) 임베딩 분석")
    print("=" * 60)

    describe(store.company_index)
    X_company, _ = get_vectors_cpu(store.company_index)
    ids_company = np.array(store.company_ids)

    print(f"\n추출된 벡터:")
    print(f"  Shape: {X_company.shape}")
    print(f"  Dtype: {X_company.dtype}")
    print(f"  Mean: {X_company.mean():.6f}, Std: {X_company.std():.6f}")
    print(f"  Min: {X_company.min():.6f}, Max: {X_company.max():.6f}")

    # 샘플 벡터 출력
    print(f"\n샘플 벡터 (company_0000):")
    print(f"  First 20 values: {X_company[0, :20]}")
    print(f"  L2 norm: {np.linalg.norm(X_company[0]):.6f}")

    labels_company = np.array([cid.split('_')[0] for cid in ids_company])

    # 품질 분석
    pos_scores, neg_scores = analyze_quality(X_company, labels_company)

    # 이웃 탐색 예시
    print("\n" + "=" * 60)
    print("이웃 탐색 예시")
    print("=" * 60)
    Xn_company = X_company / np.maximum(1e-12, np.linalg.norm(X_company, axis=1, keepdims=True))
    show_neighbors(0, Xn_company, ids_company, labels_company, topk=10)

    # Step 4: UMAP 시각화 (공고 + 업체 통합)
    if VISUALIZE:
        print("\n" + "=" * 60)
        print("통합 UMAP 시각화 (공고 + 업체)")
        print("=" * 60)

        # 공고와 업체 임베딩 합치기
        X_combined = np.vstack([X_notice, X_company])

        # 타입 라벨 생성 (Notice vs Company)
        type_labels = np.array(['Notice'] * len(X_notice) + ['Company'] * len(X_company))

        # ID 합치기
        ids_combined = np.concatenate([ids_notice, ids_company])

        # 샘플링 (Notice와 Company 비율 유지)
        max_notice_samples = min(500, len(X_notice))
        max_company_samples = min(250, len(X_company))

        rng = np.random.default_rng(42)
        notice_idx = rng.choice(len(X_notice), size=max_notice_samples, replace=False)
        company_idx = rng.choice(len(X_company), size=max_company_samples, replace=False) + len(X_notice)

        combined_idx = np.concatenate([notice_idx, company_idx])

        X_sample = X_combined[combined_idx]
        type_sample = type_labels[combined_idx]
        id_sample = ids_combined[combined_idx]

        print(f"\n샘플 개수:")
        print(f"  Notice: {max_notice_samples}")
        print(f"  Company: {max_company_samples}")
        print(f"  Total: {len(combined_idx)}")

        # 정규화
        X2 = X_sample / np.maximum(1e-12, np.linalg.norm(X_sample, axis=1, keepdims=True))

        # UMAP
        try:
            import umap
            import plotly.express as px
            import pandas as pd

            print("UMAP 투영 중...")
            U = umap.UMAP(
                n_neighbors=15,
                min_dist=0.05,
                metric="cosine",
                random_state=42
            ).fit_transform(X2)

            print(f"UMAP 결과 shape: {U.shape}")

            # Plotly scatter
            df = pd.DataFrame({
                'x': U[:, 0],
                'y': U[:, 1],
                'type': type_sample,
                'id': id_sample
            })

            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='type',
                hover_data=['id'],
                title="Embedding UMAP: Notice (파랑) vs Company (빨강)",
                width=1400,
                height=900,
                color_discrete_map={'Notice': '#1f77b4', 'Company': '#ff7f0e'}
            )
            fig.update_traces(marker=dict(size=6, opacity=0.7))

            output_path = "data/vector_db/combined_umap.html"
            fig.write_html(output_path)
            print(f"\n시각화 저장: {output_path}")

            fig.show()

        except ImportError:
            print("UMAP 시각화를 위해 추가 패키지가 필요합니다:")
            print("  pip install umap-learn plotly")

    # Step 5: 내보내기
    print("\n" + "=" * 60)
    print("벡터 내보내기 (선택)")
    print("=" * 60)
    print("필요시 아래 명령으로 numpy로 저장 가능:")
    print(f"  np.save('data/vector_db/notice_embeddings.npy', X_notice)")
    print(f"  np.save('data/vector_db/company_embeddings.npy', X_company)")

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
