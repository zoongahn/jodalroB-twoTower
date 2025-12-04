# DataLoader 병목 분석: 일회성 vs 배치별 작업

## 목차
1. [일회성 초기화 작업](#일회성-초기화-작업)
2. [배치마다 반복되는 작업](#배치마다-반복되는-작업)
3. [현재 병목 위치](#현재-병목-위치)
4. [최적화 방향](#최적화-방향)

---

## ⚡ 일회성 초기화 작업 (학습 시작 시 1회만)

### 1. 데이터 준비 단계 (`create_unified_bid_dataloaders` 호출 시)

**Test Mode 기준:**

```python
# unified_bid_data_loader.py:853-950
_create_test_mode_dataloaders()
├─ 1. DB에서 pair 로딩 (1회)
│   └─ SELECT bidntceno, bidntceord, bizno FROM pairs LIMIT 1000000
│
├─ 2. 고유 ID 추출 (1회)
│   ├─ notice_ids = set(zip(...))  # 5,154개
│   └─ company_ids = set(...)      # 95,744개
│
├─ 3. 선택적 피처 로딩 (1회, DB I/O)
│   ├─ _load_features_for_test_mode(notice)  # 5,154 rows
│   └─ _load_features_for_test_mode(company) # 95,744 rows
│
├─ 4. ID 매핑 딕셔너리 생성 (1회)
│   ├─ notice_id_to_idx = {(bidntceno, ord): idx}
│   └─ company_id_to_idx = {bizno: idx}
│
└─ 5. FeaturePreprocessor로 전처리 (1회, GPU 연산)
    ├─ Notice: (5154, raw) → (5154, 256) dense_projected
    └─ Company: (95744, raw) → (95744, 128) dense_projected
```

**메모리 상주 데이터:**
- `notice_store['dense_projected']`: (5154, 256) numpy array
- `company_store['dense_projected']`: (95744, 128) numpy array
- `notice_store['categorical']`: (5154, 32) numpy array
- `company_store['categorical']`: (95744, 6) numpy array

---

### 2. Dataset 초기화 (`UnifiedBidDataset.__init__`)

```python
# unified_bid_data_loader.py:58-191
UnifiedBidDataset.__init__()
├─ 1. Streaming 메타데이터 계산 (1회)
│   ├─ self.total_count = 800,000 (train)
│   ├─ self.num_chunks = 80 (chunk_size=10000)
│   └─ self.chunk_order = [0, 1, 2, ...] (shuffle용)
│
├─ 2. 공유 피처 스토어 참조 (1회)
│   ├─ self.notice_store = shared_stores['preprocessed']['notice']
│   └─ self.company_store = shared_stores['preprocessed']['company']
│
└─ 3. ID 매핑 복사 (1회)
    ├─ self.notice_id_to_idx (5154개)
    └─ self.company_id_to_idx (95744개)
```

---

## 🔄 배치마다 반복되는 작업 (3125 batches × 학습 내내)

### 1. Epoch 시작 시 (1 epoch당 1회)

```python
# unified_bid_data_loader.py:387-416
dataset.set_epoch(epoch)
└─ Chunk 순서 shuffle (streaming + shuffle=True일 때)
    ├─ self.chunk_order = rng.permutation(80)
    └─ self.chunk_cache.clear()
```

---

### 2. 배치 로딩 주기 (`DataLoader` iteration)

**매 배치(256 samples)마다:**

```python
for batch in train_loader:  # 3125번 반복
    # === 워커 프로세스 (병렬 실행 가능) ===
    for i in range(256):  # 배치 크기만큼
        __getitem__(idx)
        ├─ 1. Streaming: chunk 로딩 (필요 시)
        │   └─ _load_chunk(physical_chunk_id)
        │       └─ DB SQL: SELECT ... OFFSET ... LIMIT 10000  # 청크당 1회
        │
        ├─ 2. Pair ID → notice/company key 변환
        │   ├─ bidntceno = str(chunk_arrays['bidntceno'][local_idx])
        │   └─ company_key = bizno
        │
        ├─ 3. ID 매핑 딕셔너리 조회
        │   ├─ notice_idx = self.notice_id_to_idx[(bidntceno, ord)]  # O(1)
        │   └─ company_idx = self.company_id_to_idx[bizno]          # O(1)
        │
        └─ 4. 인덱스만 반환 (현재 구현)
            └─ return {"notice_idx": ni, "company_idx": ci}

    # === 메인 프로세스 (단일 스레드) ===
    collate_fn(batch_list)  # 256개 샘플
    ├─ 5. NumPy array 인덱싱 (256번 반복) ⚠️ 병목!
    │   for item in batch:  # 256번
    │       n_dense_list.append(notice_store['dense_projected'][ni])  # numpy 복사
    │       c_dense_list.append(company_store['dense_projected'][ci])
    │       n_cat_list.append(notice_store['categorical'][ni])
    │       c_cat_list.append(company_store['categorical'][ci])
    │
    ├─ 6. NumPy stack → Torch (CPU) ⚠️ 병목!
    │   ├─ np.stack(n_dense_list)  # (256, 256)
    │   ├─ torch.from_numpy(...)   # NumPy → Torch 복사
    │   └─ 총 4번 stack + 4번 from_numpy
    │
    ├─ 7. KJT 생성 (CPU) ⚠️ 병목!
    │   ├─ _build_batch_kjt(notice_cat, 32 keys)  # Python 루프
    │   └─ _build_batch_kjt(company_cat, 6 keys)
    │
    └─ 8. GPU 전송 (비동기지만 collate 끝난 후)
        ├─ .to(device, non_blocking=True)
        └─ 총 4번 H2D 전송
```

---

## 🔴 현재 병목 위치 (배치마다 반복)

| 작업 | 실행 위치 | 병렬화 | 비용 | 빈도 |
|------|----------|--------|------|------|
| **5. NumPy 인덱싱 (256×4=1024번)** | 메인 프로세스 | ❌ | **HIGH** | 매 배치 |
| **6. NumPy stack + from_numpy (×4)** | 메인 프로세스 | ❌ | **MEDIUM** | 매 배치 |
| **7. KJT 생성 (Python 루프)** | 메인 프로세스 | ❌ | **HIGH** | 매 배치 |
| 8. GPU 전송 | 메인 프로세스 | ⚠️ (non_blocking) | LOW | 매 배치 |

**👆 5~7번이 메인 프로세스에서 순차 실행되므로 GPU가 대기!**

### 상세 분석

#### 병목 1: NumPy 인덱싱 (5번)
```python
# collate_fn 내부 (메인 프로세스)
for item in batch:  # 256번 반복
    ni = item["notice_idx"]
    ci = item["company_idx"]

    # 매번 numpy array 인덱싱 → 메모리 복사 발생
    n_dense_list.append(dataset.notice_store['dense_projected'][ni])
    c_dense_list.append(dataset.company_store['dense_projected'][ci])
    n_cat_list.append(dataset.notice_store['categorical'][ni])
    c_cat_list.append(dataset.company_store['categorical'][ci])
```

**문제점:**
- 메인 프로세스가 256번 반복하며 numpy 인덱싱
- 워커 프로세스들은 유휴 상태 (인덱스만 반환했으므로)
- GIL(Global Interpreter Lock)로 인한 병렬화 불가

#### 병목 2: NumPy → Torch 변환 (6번)
```python
# 4번의 stack + from_numpy
notice_dense  = torch.from_numpy(np.stack(n_dense_list, axis=0)).float()
company_dense = torch.from_numpy(np.stack(c_dense_list, axis=0)).float()
notice_cat_np  = np.stack(n_cat_list, axis=0)
company_cat_np = np.stack(c_cat_list, axis=0)
```

**문제점:**
- NumPy stack은 새 메모리 할당 + 복사
- from_numpy는 메모리 공유지만 타입 변환 시 복사 발생
- 배치 크기가 클수록 오버헤드 증가

#### 병목 3: KJT 생성 (7번)
```python
def _build_batch_kjt(cat_batch: torch.Tensor, keys: List[str]):
    B, K = cat_batch.shape

    values_by_key = []
    lengths_by_key = []

    for k in range(K):  # 32번 또는 6번 반복
        values_by_key.append(cat_batch[:, k])
        lengths_by_key.append(torch.ones(B, dtype=torch.int32))

    all_values = torch.cat(values_by_key, dim=0)
    all_lengths = torch.cat(lengths_by_key, dim=0)

    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=all_values,
        lengths=all_lengths
    )
```

**문제점:**
- Python for 루프 (K=32 또는 6)
- 매 배치마다 Python object 생성 오버헤드
- KeyedJaggedTensor 내부에서 추가 검증/변환

---

## ✅ 최적화 방향

### 현재 구조
```
워커 프로세스 (병렬 8개) → 인덱스만 반환 {"notice_idx": 123, ...}
                           ↓
메인 프로세스 (단일)     → NumPy 인덱싱 (256×4번)
                         → NumPy stack (4번)
                         → from_numpy (4번)
                         → KJT 생성 (2번)
                           ↓
                         GPU 대기! ⏸️
```

### 개선 후 구조
```
워커 프로세스 (병렬 8개) → NumPy 인덱싱
                         → torch.Tensor 변환
                         → return Tensor
                           ↓
메인 프로세스 (단일)     → torch.stack만 (가벼움)
                         → KJT 생성 (배치 단위 1회)
                           ↓
                         GPU 바로 처리! ⚡
```

### 구체적 변경 사항

#### 1. `__getitem__` 수정 (워커에서 무거운 작업 처리)
```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    # ... (기존 ID 조회 로직) ...

    # ✅ 워커 프로세스에서 NumPy → Torch 변환
    notice_dense = torch.from_numpy(
        self.notice_store['dense_projected'][notice_idx]
    ).to(torch.float32)

    company_dense = torch.from_numpy(
        self.company_store['dense_projected'][company_idx]
    ).to(torch.float32)

    notice_cat = torch.from_numpy(
        self.notice_store['categorical'][notice_idx]
    ).to(torch.long)

    company_cat = torch.from_numpy(
        self.company_store['categorical'][company_idx]
    ).to(torch.long)

    return {
        "notice_dense": notice_dense,    # [D_n] Tensor
        "company_dense": company_dense,  # [D_c] Tensor
        "notice_cat": notice_cat,        # [K_n] Tensor
        "company_cat": company_cat,      # [K_c] Tensor
    }
```

#### 2. `collate_fn` 수정 (메인은 가볍게)
```python
def collate_fn(batch: List[Dict]) -> Dict:
    # ✅ 이미 Tensor이므로 stack만 수행
    notice_dense = torch.stack([b["notice_dense"] for b in batch], dim=0)
    company_dense = torch.stack([b["company_dense"] for b in batch], dim=0)
    notice_cat = torch.stack([b["notice_cat"] for b in batch], dim=0)
    company_cat = torch.stack([b["company_cat"] for b in batch], dim=0)

    # KJT는 배치 단위로 1회만 생성
    notice_kjt = _build_batch_kjt(notice_cat, schema.notice.categorical)
    company_kjt = _build_batch_kjt(company_cat, schema.company.categorical)

    return {
        "notice": {"dense": notice_dense, "kjt": notice_kjt},
        "company": {"dense": company_dense, "kjt": company_kjt},
    }
```

### 기대 효과

| 항목 | 변경 전 | 변경 후 | 개선도 |
|------|---------|---------|--------|
| NumPy 인덱싱 | 메인 (순차) | 워커 (병렬 8개) | **8배** |
| from_numpy | 메인 (4번) | 워커 (병렬) | **8배** |
| IPC 오버헤드 | 낮음 (인덱스만) | 중간 (Tensor) | 약간 증가 |
| 메인 프로세스 부하 | **높음** | **낮음** | **대폭 감소** |
| GPU 활용률 | 40% | 80-95% (예상) | **2배 이상** |

---

## 📝 주의사항

### 1. Pin Memory와의 조합
- `pin_memory=True`일 때 collate에서 반환하는 텐서는 **CPU**에 있어야 함
- DataLoader가 자동으로 pinned memory로 이동
- 이후 `.to(device, non_blocking=True)`가 진짜 비동기로 동작

### 2. IPC 오버헤드
- Tensor 반환 시 공유 메모리 사용으로 복사 최소화
- NumPy 반환 시 피클링/직렬화 오버헤드 큼
- **반드시 `torch.Tensor`로 반환!**

### 3. KJT 생성 전략
- 샘플마다 KJT 생성 ❌ (직렬화 오버헤드)
- 배치 단위로 1회 생성 ✅
- 더 나은 방법: GPU에 feature table 상주 + ID만 전송

---

## 🚀 더 나은 최적화 (선택)

### GPU Feature Table 상주 방식

```python
# 초기화 시 GPU로 전체 테이블 업로드 (1회)
notice_dense_gpu = torch.from_numpy(notice_store['dense_projected']).cuda()
company_dense_gpu = torch.from_numpy(company_store['dense_projected']).cuda()

# __getitem__은 ID만 반환
def __getitem__(self, idx):
    return {
        "notice_id": notice_idx,  # int
        "company_id": company_idx # int
    }

# Forward에서 GPU에서 직접 조회
def forward(self, batch):
    notice_dense = notice_dense_gpu[batch["notice_id"]]  # GPU gather
    company_dense = company_dense_gpu[batch["company_id"]]
```

**장점:**
- H2D 페이로드 최소 (ID만 전송)
- GPU에서 직접 gather → 매우 빠름
- 메인 프로세스 부하 거의 없음

**단점:**
- GPU 메모리 필요
- 현재 5154×256 + 95744×128 ≈ 50MB (충분히 가능)

---

## 결론

**즉시 적용 가능한 최적화:**
1. `__getitem__` → torch.Tensor 반환
2. `collate_fn` → stack만 수행
3. KJT 배치 단위 생성

**예상 성능 향상:**
- GPU 활용률: 40% → 80-95%
- 배치 처리 속도: 23 batch/s → 40-60 batch/s (예상)
- 학습 시간: 약 50% 단축 (예상)
