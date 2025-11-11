"""
KJT 생성 유틸리티 - GPU 최적화 (Production Ready)

기존 방식의 문제점:
1. CPU에서 KJT 생성 → from_lengths_sync() 호출 (CPU-GPU 동기화)
2. Python dict lookup → NumPy stack 반복
3. Collate function에서 매 배치마다 재생성

개선 방향:
1. GPU에서 직접 KJT 생성 (stride=B 명시로 배치 차원 보장)
2. Torch gather operations 사용 (NumPy 제거)
3. FP16 dense + 파이썬 루프 최소화
4. DDP/멀티 GPU 대비 local_rank 기반 device
5. 멀티-핫 필드 지원
"""

import torch
from torchrec import KeyedJaggedTensor
from typing import List, Dict, Optional, Callable
import numpy as np


def create_kjt_from_batch_gpu(
    cat_indices: torch.Tensor,
    keys: List[str],
    device: Optional[torch.device] = None,
    num_embeddings_per_key: Optional[List[int]] = None,
    remap: Optional[str] = "mod"
) -> KeyedJaggedTensor:
    """
    GPU에서 배치 단위 KJT 생성 (최적화 버전)

    CRITICAL:
    - from_lengths_sync(..., stride=B) 사용으로 배치 차원 명시
    - 직접 생성자 대신 sync API 사용 → EBC가 stride 정확히 해석

    Args:
        cat_indices: [B, K] categorical feature indices (이미 GPU에 있음)
        keys: K개 feature names
        device: 디바이스 (None이면 cat_indices의 device 사용)

    Returns:
        KeyedJaggedTensor (GPU에 있음)
    """
    if device is None:
        device = cat_indices.device

    B, K = cat_indices.shape
    assert K == len(keys), f"Keys length {len(keys)} != tensor shape {K}"

    # Key-major 순서로 변환
    # [B, K] → [K, B] → [K*B] flatten
    values = cat_indices.t().reshape(-1).to(device=device, dtype=torch.long)  # [K*B]

    # Lengths: 각 feature마다 B개 값 (모두 1)
    lengths = torch.ones(K * B, device=device, dtype=torch.int32)

    # CRITICAL: stride=B 명시 (EBC가 배치 차원을 정확히 해석)
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=values,
        lengths=lengths,
        stride=B  # 🔴 배치 크기 명시 (필수!)
    )

    return kjt


def create_kjt_from_multihot_gpu(
    values_list: List[torch.Tensor],
    lengths_list: List[torch.Tensor],
    keys: List[str],
    batch_size: int,
    device: torch.device
) -> KeyedJaggedTensor:
    """
    멀티-핫 필드 지원 KJT 생성

    단일 값이 아닌 가변 길이 값 리스트를 갖는 필드 처리
    (예: tags=[1,3,5], categories=[2,7])

    Args:
        values_list: 각 필드의 values (concat 아님, 필드별 텐서)
        lengths_list: 각 필드의 lengths (B개 합=values 수)
        keys: feature names
        batch_size: 배치 크기
        device: GPU device

    Returns:
        KeyedJaggedTensor (멀티-핫 지원)

    Example:
        # 2개 필드, 배치 크기 3
        values_list = [
            torch.tensor([1,2, 3,4,5, 6], device='cuda'),  # field1: [2, 3, 1] 길이
            torch.tensor([7,8,9, 10, 11,12,13], device='cuda')  # field2: [3, 1, 3] 길이
        ]
        lengths_list = [
            torch.tensor([2,3,1], device='cuda', dtype=torch.int32),
            torch.tensor([3,1,3], device='cuda', dtype=torch.int32)
        ]
        kjt = create_kjt_from_multihot_gpu(values_list, lengths_list, ['f1','f2'], 3, device)
    """
    # Key-major concat
    all_values = torch.cat(values_list, dim=0).to(device=device, dtype=torch.long)
    all_lengths = torch.cat(lengths_list, dim=0).to(device=device, dtype=torch.int32)

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=all_values,
        lengths=all_lengths,
        stride=batch_size
    )

    return kjt


def create_kjt_from_store_gpu(
    indices: torch.Tensor,
    feature_store: Dict[str, np.ndarray],
    keys: List[str],
    device: torch.device
) -> KeyedJaggedTensor:
    """
    Feature store에서 직접 KJT 생성 (GPU 최적화)

    기존 방식:
    1. indices → NumPy indexing → stack → from_lengths_sync()

    새 방식:
    1. indices (GPU) → torch.index_select (GPU) → KJT (GPU)

    Args:
        indices: [B] 배치 인덱스 (GPU에 있음)
        feature_store: categorical features (NumPy array)
        keys: feature names
        device: GPU device

    Returns:
        KeyedJaggedTensor (GPU)
    """
    # Feature store를 GPU tensor로 변환 (한 번만)
    if not isinstance(feature_store['categorical'], torch.Tensor):
        cat_tensor = torch.from_numpy(feature_store['categorical']).long().to(device)
    else:
        cat_tensor = feature_store['categorical'].to(device)

    # GPU index_select (매우 빠름!)
    batch_cat = torch.index_select(cat_tensor, dim=0, index=indices.long())  # [B, K]

    # KJT 생성
    return create_kjt_from_batch_gpu(batch_cat, keys, device)


# 전역 캐시: 중복 메모리 방지
_COLLATE_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def create_ebc_collate_fn_v2(
    notice_store: Dict,
    company_store: Dict,
    notice_keys: List[str],
    company_keys: List[str],
    device: torch.device,
    use_fp16: bool = True,
    cache_key: Optional[str] = None
) -> Callable:
    """
    EmbeddingBagCollection용 최적화된 collate function

    기존 방식 (create_safe_pair_collate_fn):
    - CPU NumPy indexing
    - CPU에서 KJT 생성
    - from_lengths_sync() 호출 (CPU-GPU sync)
    - float32 dense (AMP 사용 시 캐스팅 오버헤드)

    새 방식 (V2):
    - GPU에서 index_select
    - GPU에서 KJT 생성 (stride=B 명시)
    - FP16 dense (AMP 친화적, 메모리/대역폭 절약)
    - 파이썬 루프 최소화
    - 전역 캐시로 중복 메모리 방지

    Args:
        notice_store: Notice feature store (NumPy)
        company_store: Company feature store (NumPy)
        notice_keys: Notice feature names
        company_keys: Company feature names
        device: GPU device (DDP 시 local_rank 기반)
        use_fp16: True면 dense를 FP16으로 (AMP 친화적)
        cache_key: 캐시 키 (None이면 캐싱 안 함)

    Returns:
        collate function
    """
    global _COLLATE_CACHE

    # 캐시 확인 (train/test 중복 방지)
    if cache_key and cache_key in _COLLATE_CACHE:
        print(f"✅ Collate cache hit: {cache_key} (메모리 중복 방지)")
        cached = _COLLATE_CACHE[cache_key]
        notice_cat_gpu = cached['notice_cat']
        company_cat_gpu = cached['company_cat']
        notice_dense_gpu = cached['notice_dense']
        company_dense_gpu = cached['company_dense']
    else:
        # 1회 GPU 상주 (FP16으로 메모리/대역폭 절약)
        dtype_dense = torch.float16 if use_fp16 else torch.float32

        notice_cat_gpu = torch.from_numpy(notice_store['categorical']).to(
            device=device, dtype=torch.long
        )
        company_cat_gpu = torch.from_numpy(company_store['categorical']).to(
            device=device, dtype=torch.long
        )

        notice_dense_gpu = torch.from_numpy(notice_store['dense_projected']).to(
            device=device, dtype=dtype_dense
        )
        company_dense_gpu = torch.from_numpy(company_store['dense_projected']).to(
            device=device, dtype=dtype_dense
        )

        # 캐싱
        if cache_key:
            _COLLATE_CACHE[cache_key] = {
                'notice_cat': notice_cat_gpu,
                'company_cat': company_cat_gpu,
                'notice_dense': notice_dense_gpu,
                'company_dense': company_dense_gpu
            }
            print(f"✅ Collate cache created: {cache_key}")

        print(f"✅ Collate function 초기화 완료:")
        print(f"   Notice cat: {notice_cat_gpu.shape} (GPU, {notice_cat_gpu.dtype})")
        print(f"   Company cat: {company_cat_gpu.shape} (GPU, {company_cat_gpu.dtype})")
        print(f"   Notice dense: {notice_dense_gpu.shape} (GPU, {notice_dense_gpu.dtype})")
        print(f"   Company dense: {company_dense_gpu.shape} (GPU, {company_dense_gpu.dtype})")

    def collate(batch: List[Dict]) -> Dict:
        """
        배치 collate (완전 GPU 파이프라인)

        Input:
            batch: [{"notice_idx": int, "company_idx": int}, ...]

        Output:
            {
                "notice": {"dense": Tensor[B, D], "kjt": KeyedJaggedTensor},
                "company": {"dense": Tensor[B, D], "kjt": KeyedJaggedTensor}
            }
        """
        # CRITICAL: 파이썬 루프 최소화
        # 이상적: Dataset이 텐서 리턴 → default_collate → .to(device, non_blocking=True)
        # 현재 구조 유지 시: 한 번만 변환
        nid = torch.tensor([b['notice_idx'] for b in batch],
                          device=device, dtype=torch.long)
        cid = torch.tensor([b['company_idx'] for b in batch],
                          device=device, dtype=torch.long)

        # GPU index_select (매우 빠름!)
        n_dense = torch.index_select(notice_dense_gpu, dim=0, index=nid)  # [B, Dn]
        c_dense = torch.index_select(company_dense_gpu, dim=0, index=cid)  # [B, Dc]

        n_cat = torch.index_select(notice_cat_gpu, dim=0, index=nid)  # [B, Fn]
        c_cat = torch.index_select(company_cat_gpu, dim=0, index=cid)  # [B, Fc]

        # KJT 생성 (stride=B 명시)
        n_kjt = create_kjt_from_batch_gpu(n_cat, notice_keys, device)
        c_kjt = create_kjt_from_batch_gpu(c_cat, company_keys, device)

        return {
            "notice": {"dense": n_dense, "kjt": n_kjt},
            "company": {"dense": c_dense, "kjt": c_kjt}
        }

    return collate


def clear_collate_cache(cache_key: Optional[str] = None):
    """
    Collate 캐시 초기화

    Args:
        cache_key: 특정 키만 삭제 (None이면 전체 삭제)
    """
    global _COLLATE_CACHE
    if cache_key:
        if cache_key in _COLLATE_CACHE:
            del _COLLATE_CACHE[cache_key]
            print(f"✅ Collate cache cleared: {cache_key}")
    else:
        _COLLATE_CACHE.clear()
        print(f"✅ Collate cache cleared: all")


def benchmark_kjt_creation():
    """KJT 생성 방식 비교 벤치마크"""
    import time

    # 더미 데이터 생성
    B = 256  # 배치 크기
    K = 32   # 피처 개수
    vocab_size = 1000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    keys = [f"feat_{i}" for i in range(K)]

    # 방법 1: CPU NumPy stack + from_lengths_sync (기존)
    print("\n=== 방법 1: CPU NumPy (기존 방식) ===")
    cat_np = np.random.randint(0, vocab_size, (B, K))

    times_cpu = []
    for _ in range(100):
        start = time.perf_counter()

        # NumPy stack
        values_list = []
        lengths_list = []
        for k in range(K):
            values_list.append(cat_np[:, k])
            lengths_list.append(np.ones(B, dtype=np.int32))

        all_values = np.concatenate(values_list)
        all_lengths = np.concatenate(lengths_list)

        # CPU에서 KJT 생성
        values_t = torch.from_numpy(all_values).long()
        lengths_t = torch.from_numpy(all_lengths).int()

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=keys,
            values=values_t,
            lengths=lengths_t,
            stride=B
        )

        # GPU로 이동
        if device.type == 'cuda':
            kjt = kjt.to(device)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = (time.perf_counter() - start) * 1000
        times_cpu.append(elapsed)

    print(f"평균: {np.mean(times_cpu):.3f} ms")
    print(f"표준편차: {np.std(times_cpu):.3f} ms")

    # 방법 2: GPU direct (새 방식)
    print("\n=== 방법 2: GPU Direct (새 방식 - stride=B) ===")
    cat_gpu = torch.from_numpy(cat_np).long().to(device)

    times_gpu = []
    for _ in range(100):
        start = time.perf_counter()

        kjt = create_kjt_from_batch_gpu(cat_gpu, keys, device)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = (time.perf_counter() - start) * 1000
        times_gpu.append(elapsed)

    print(f"평균: {np.mean(times_gpu):.3f} ms")
    print(f"표준편차: {np.std(times_gpu):.3f} ms")

    # 비교
    print("\n=== 성능 비교 ===")
    speedup = np.mean(times_cpu) / np.mean(times_gpu)
    print(f"속도 향상: {speedup:.2f}x")
    print(f"절감 시간: {np.mean(times_cpu) - np.mean(times_gpu):.3f} ms/batch")


if __name__ == "__main__":
    print("=== KJT 생성 유틸리티 테스트 ===\n")

    # 1. 기본 테스트
    print("1. GPU KJT 생성 테스트 (stride=B 명시)")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B = 4
    K = 3
    keys = ["feat_a", "feat_b", "feat_c"]

    # 더미 데이터
    cat_batch = torch.randint(0, 100, (B, K), device=device)
    print(f"   입력: {cat_batch.shape} on {device}")

    kjt = create_kjt_from_batch_gpu(cat_batch, keys, device)
    print(f"   출력 KJT keys: {kjt.keys()}")
    print(f"   출력 KJT stride: {kjt.stride()} (배치 크기: {B})")
    assert kjt.stride() == B, f"Stride mismatch: {kjt.stride()} != {B}"
    print(f"   ✅ 생성 성공 (stride 검증 OK)\n")

    # 2. Feature store 기반 테스트
    print("2. Feature store 기반 KJT 생성")

    # 더미 feature store
    N = 100  # 전체 샘플 수
    feature_store = {
        'categorical': np.random.randint(0, 50, (N, K))
    }

    # 배치 인덱스
    batch_indices = torch.randint(0, N, (B,), device=device)
    print(f"   Batch indices: {batch_indices}")

    kjt2 = create_kjt_from_store_gpu(batch_indices, feature_store, keys, device)
    print(f"   출력 KJT keys: {kjt2.keys()}")
    print(f"   출력 KJT stride: {kjt2.stride()}")
    print(f"   ✅ 생성 성공\n")

    # 3. 멀티-핫 필드 테스트
    print("3. 멀티-핫 필드 KJT 생성")
    B_multi = 3
    values_list = [
        torch.tensor([1, 2, 3, 4, 5, 6], device=device),  # field1: 길이 [2, 3, 1]
        torch.tensor([7, 8, 9, 10, 11, 12, 13], device=device)  # field2: 길이 [3, 1, 3]
    ]
    lengths_list = [
        torch.tensor([2, 3, 1], device=device, dtype=torch.int32),
        torch.tensor([3, 1, 3], device=device, dtype=torch.int32)
    ]
    kjt_multi = create_kjt_from_multihot_gpu(
        values_list, lengths_list, ["f1", "f2"], B_multi, device
    )
    print(f"   출력 KJT keys: {kjt_multi.keys()}")
    print(f"   출력 KJT stride: {kjt_multi.stride()}")
    print(f"   출력 KJT values: {kjt_multi.values()}")
    print(f"   ✅ 멀티-핫 생성 성공\n")

    # 4. 벤치마크
    if device.type == 'cuda':
        print("4. 성능 벤치마크")
        benchmark_kjt_creation()
    else:
        print("4. 성능 벤치마크 (CUDA 없음 - 스킵)")

    print("\n✅ 모든 테스트 완료!")
