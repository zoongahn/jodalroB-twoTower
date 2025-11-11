# TorchRec EmbeddingBagCollection 구현 계획

## 📋 목표

기존 범주형 처리 방식을 유지하면서, TorchRec의 `EmbeddingBagCollection` 방식을 병렬로 구현하여 GPU 점유율 개선

---

## 🎯 핵심 전략

### 1. **기존 코드 보존 원칙**
- ✅ 기존 `CatEmbed`, `TwoTowerModel` 코드는 그대로 유지
- ✅ 새로운 모듈은 `V2` suffix로 구분 (예: `CatEmbedV2`, `TwoTowerModelV2`)
- ✅ 학습 스크립트에서 `--use_ebc` 플래그로 방식 선택 가능

### 2. **병목 제거 목표**
| 현재 병목 | TorchRec 방식 해결 |
|----------|------------------|
| ❌ 매 배치마다 `_build_batch_kjt` 호출 (CPU-GPU sync) | ✅ EmbeddingBagCollection에서 내부 처리 |
| ❌ Python dict lookup (GIL 경합) | ✅ GPU 상주 테이블에서 직접 indexing |
| ❌ 순차 처리 (Pipeline 없음) | ✅ Embedding lookup이 GPU kernel로 실행 |

---

## 📐 아키텍처 설계

### **현재 구조 (기존 방식)**

```
DataLoader
  └─> __getitem__: dict lookup (Python)
        └─> collate_fn: NumPy stack + KJT 생성 (CPU)
              └─> Training loop
                    └─> GPU에서 gather + KJT 재생성
                          └─> CatEmbed: Embedding lookup
                                └─> Tower MLP
```

**문제점:**
1. `_build_batch_kjt`가 매 배치마다 호출됨 (CPU-GPU sync)
2. Python dict lookup 오버헤드
3. KJT 생성 시 `from_lengths_sync()` 호출로 동기화 발생

---

### **새로운 구조 (EmbeddingBagCollection 방식)**

```
DataLoader
  └─> __getitem__: 인덱스만 반환 (int)
        └─> collate_fn_v2: LongTensor로 배치 생성
              └─> Training loop
                    └─> GPU에서 직접 KJT 생성 (비동기)
                          └─> EmbeddingBagCollection
                                └─> Pooled embeddings (GPU kernel)
                                      └─> Tower MLP
```

**개선점:**
1. ✅ KJT 생성이 EmbeddingBagCollection 내부에서 GPU kernel로 처리
2. ✅ Python overhead 제거 (dict → direct indexing)
3. ✅ Embedding lookup + pooling이 한 번에 처리 (fused operation)

---

## 🛠️ 구현 상황 (업데이트: 2025-11-08)

### **Phase 1: 아키텍처 설계 및 Config 정의** ✅ **완료**

#### Task 1.1: Metadata Parser 구현 ✅
**파일:** `preprocess/metadata_parser.py` (완료)

**구현 내용:**
- metadata.csv에서 범주형 피처의 vocab_size 자동 추출
- 사용 여부(Y) + 범주형 여부(Y) 필터링
- 동적 범주 추가 대응: 여유있는 num_embeddings 권장 (20~50% 추가)
- UNK 토큰 지원

**실제 분석 결과:**
```
Notice 테이블: 32개 범주형 피처
  - 총 Vocab Size: 59,704
  - 메모리: 8.82 MB (embedding_dim=32)
  - 최대 vocab: dminsttcd (29,910), ntceinsttcd (28,983)

Company 테이블: 6개 범주형 피처
  - 총 Vocab Size: 4,057
  - 메모리: 0.66 MB
  - 최대 vocab: rgnnm (3,670)

총 메모리: 9.48 MB (매우 경량!)
```

**검증:**
- ✅ metadata.csv에서 vocab_size 자동 추출
- ✅ 모든 범주형 피처에 대한 config 생성 확인
- ✅ 여유있는 num_embeddings 권장 (1.2x ~ 2x)
- ✅ UNK 토큰 지원 (+1)

---

#### Task 1.2: EmbeddingBagConfig 생성 모듈 ✅
**파일:** `src/towers/embedding_config.py` (완료)

**구현 내용:**
```python
from torchrec import EmbeddingBagConfig
from torchrec.modules.embedding_configs import PoolingType
from preprocess.metadata_parser import MetadataParser

def create_notice_embedding_configs(
    metadata_path: str = "meta/metadata.csv",
    embedding_dim: int = 32,
    pooling_mode: PoolingType = PoolingType.MEAN,  # CRITICAL: Enum 사용!
    add_unk_token: bool = True
) -> List[EmbeddingBagConfig]

def create_company_embedding_configs(...)
```

**주요 학습 사항:**
- ⚠️ **pooling은 문자열이 아니라 `PoolingType` enum 사용!**
- ⚠️ **`name`과 `feature_names`의 차이 이해 필요**
  - `name`: EmbeddingBagConfig 식별자 (예: `"notice_rentceyn"`)
  - `feature_names`: KJT keys와 매칭되는 실제 피처명 (예: `["rentceyn"]`)

**검증:**
- ✅ Notice/Company별 EmbeddingBagConfig 생성
- ✅ 메모리 사용량 추정 (9.49 MB)
- ✅ 파라미터 수 계산 (2,487,648개)

---

### **Phase 2: CatEmbedV2 모듈 구현** ✅ **완료 (정확성 검증됨)**

#### Task 2.1: EmbeddingBagCollection 기반 임베딩 레이어 ✅
**파일:** `src/towers/cat_embed_v2.py` (완료)

**구현 내용:**
```python
class CatEmbedV2(nn.Module):
    def __init__(self, embedding_configs, device):
        self.ebc = EmbeddingBagCollection(tables=embedding_configs, device=device)
        self.output_dim = sum(cfg.embedding_dim for cfg in embedding_configs)

        # CRITICAL: pooled_features의 key는 feature_names를 사용!
        self.feature_names = [cfg.feature_names[0] for cfg in embedding_configs]

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        pooled_features: KeyedTensor = self.ebc(kjt)

        # KeyedTensor → Tensor 변환
        values_list = []
        for name in self.feature_names:
            values_list.append(pooled_features[name])

        return torch.cat(values_list, dim=1)
```

**발견한 이슈 및 해결:**

**🐛 Issue #1: 출력이 모두 0 (Zero Output)**
- **원인:** `self.feature_names`가 `cfg.name` (예: `"notice_rentceyn"`)을 저장했으나, `pooled_features`의 key는 `cfg.feature_names[0]` (예: `"rentceyn"`)
- **해결:** `self.feature_names = [cfg.feature_names[0] for cfg in embedding_configs]`
- **교훈:** EmbeddingBagCollection의 출력 KeyedTensor는 `feature_names`를 key로 사용!

**정확성 검증 (test/test_ebc_sanity.py):**
- ✅ Test 1: 최소 단위 EBC - 정상 동작 (출력 > 0)
- ✅ Test 2: Weight 초기화 - 정상 (abs_mean: 0.22)
- ✅ Test 3: KJT Keys ↔ Config 매칭 - 정상
- ✅ Test 4: KJT Lengths/Values 정합성 - 정상
- ✅ Test 5: CatEmbedV2 Full Forward - 정상 (출력 범위: [-0.27, +0.26])

**검증:**
- ✅ 더미 KJT로 forward pass 테스트
- ✅ 출력 shape 확인: `[batch_size, sum(embedding_dims)]`
- ✅ GPU에서 정상 동작 (cuda:0)
- ✅ 파라미터 수 일치 (2,313,888개)

---

#### Task 2.2: KJT 생성 유틸리티 최적화 🚧 **진행 중**
**파일:** `src/towers/kjt_utils.py` (예정)

**목표:**
- GPU 상의 카테고리 인덱스로부터 KJT 직접 생성
- CPU-GPU sync 제거
- Python loop 제거 (vectorized operation)

**구현 예정:**
```python
def build_kjt_from_indices(
    cat_indices: torch.Tensor,  # [B, K] on GPU
    keys: List[str],
    device: torch.device
) -> KeyedJaggedTensor:
    """
    GPU에서 직접 KJT 생성 (비동기)

    개선점:
    - from_lengths_sync → KeyedJaggedTensor(...) (비동기)
    - Python loop 제거
    - GPU 텐서 연산만 사용
    """
    B, K = cat_indices.shape

    # Key-major 순서로 재배열 (vectorized)
    values = cat_indices.t().reshape(-1)  # [K*B]
    lengths = torch.ones(K * B, dtype=torch.int32, device=device)

    return KeyedJaggedTensor(
        keys=keys,
        values=values,
        lengths=lengths
    )
```

**검증 예정:**
- [ ] 기존 `_build_batch_kjt`와 동일한 출력
- [ ] CPU-GPU sync 발생 여부 확인
- [ ] 성능 측정: 기존 vs 새로운 방식

---

### **Phase 3: TwoTowerModelV2 구현** ⏳ **대기 중**

**계획:**
- CatEmbedV2를 사용하는 Two-Tower 모델
- 기존 BaseTower 재사용
- KJT를 직접 입력받는 forward signature

---

### **Phase 4: DataLoader 수정** ⏳ **대기 중**

**계획:**
- `create_ebc_collate_fn` 구현
- GPU에서 직접 KJT 생성
- NumPy stack 제거

---

### **Phase 5: Training Script 통합** ⏳ **대기 중**

**계획:**
- `--use_ebc` 플래그 추가
- 기존 방식과 EBC 방식 분기 처리

---

### **Phase 6: 성능 벤치마크** ⏳ **대기 중**

**측정 예정:**
- 배치 처리 시간
- GPU 점유율
- Throughput
- 메모리 사용량

---

## 📊 현재까지 달성한 성과

### **구현 완료:**
1. ✅ **Metadata Parser** - metadata.csv 자동 파싱
2. ✅ **EmbeddingBagConfig 생성** - Notice/Company별 설정
3. ✅ **CatEmbedV2** - EmbeddingBagCollection 기반 임베딩
4. ✅ **정확성 검증** - 5개 테스트 모두 통과

### **주요 학습 사항:**
1. **PoolingType은 Enum 사용** (`PoolingType.MEAN`, not `"mean"`)
2. **name vs feature_names 구분**
   - `name`: Config 식별자
   - `feature_names`: KJT keys와 매칭
3. **KeyedTensor 접근 방법**
   - `pooled_features[feature_name]` (feature_name으로 접근)
4. **동적 범주 대응 전략**
   - 여유있는 num_embeddings (1.2x ~ 2x)
   - UNK 토큰 지원

### **코드 변경 사항:**
- 패키지 이동: `src/torchrec_preprocess/` → `preprocess/`
- 모든 import 경로 업데이트

---

## 🚨 발견한 이슈 및 해결책

### **Issue #1: Zero Output Problem**
**증상:** CatEmbedV2 forward 출력이 모두 0

**근본 원인:**
```python
# 잘못된 코드
self.feature_names = [cfg.name for cfg in embedding_configs]  # "notice_rentceyn"
# pooled_features의 key는 "rentceyn"이므로 매칭 실패!
```

**해결:**
```python
# 올바른 코드
self.feature_names = [cfg.feature_names[0] for cfg in embedding_configs]  # "rentceyn"
```

**진단 과정:**
1. 최소 EBC 테스트 → EBC 자체는 정상
2. Weight 초기화 확인 → 정상
3. Key 매칭 확인 → **불일치 발견!**
4. feature_names 사용으로 수정 → 해결 ✅

---

## ⚠️ 주의사항 (Lessons Learned)

### **1. TorchRec 특수성**
- `EmbeddingBagConfig.pooling`은 **PoolingType enum** 사용
- `KeyedTensor`는 feature_names로 접근
- `from_lengths_sync`는 CPU-GPU sync 발생 → 비동기 버전 사용 필요

### **2. 디버깅 체크리스트**
출력이 0인 경우:
1. ✅ EBC 자체 동작 확인 (최소 예제)
2. ✅ Weight 초기화 확인 (mean, std)
3. ✅ **Key 매칭 확인** (name vs feature_names)
4. ✅ KJT lengths/values 정합성
5. ✅ Device 일치 (CUDA/CPU)

### **3. 동적 범주 문제**
- 새로운 범주 추가 가능성 대비
- num_embeddings를 여유있게 설정 (20~50% 추가)
- UNK 토큰 지원 (+1)
- 주기적인 metadata 업데이트 + 모델 재학습 전략 필요

---

## 📅 업데이트된 일정

| Phase | 작업 | 상태 | 실제 소요 |
|-------|------|------|----------|
| Phase 1 | 설계 및 Config | ✅ 완료 | 1일 |
| Phase 2 | CatEmbedV2 구현 | ✅ 완료 | 1일 (정확성 검증 포함) |
| Phase 3 | TwoTowerModelV2 구현 | ⏳ 대기 | - |
| Phase 4 | DataLoader 수정 | ⏳ 대기 | - |
| Phase 5 | Training Script 통합 | ⏳ 대기 | - |
| Phase 6 | 벤치마크 및 최적화 | ⏳ 대기 | - |

**진행률:** Phase 1~2 완료 (33%)

---

## 🎯 다음 단계

### **우선순위 1: KJT 생성 최적화**
- `kjt_utils.py` 구현
- GPU 텐서 연산만 사용
- 성능 측정

### **우선순위 2: 통합 테스트**
- TwoTowerModelV2 구현
- End-to-end forward pass 검증

### **우선순위 3: 성능 튜닝**
- `torch.compile` 적용
- Cross-batch memory
- 배치/차원 확대

---

## 📚 참고 자료

- [TorchRec Documentation](https://pytorch.org/torchrec/)
- [EmbeddingBagCollection API](https://pytorch.org/torchrec/torchrec.modules.embedding_modules.html)
- [KeyedJaggedTensor Guide](https://pytorch.org/torchrec/torchrec.sparse.html)
- TorchRec GitHub: `examples/retrieval/two_tower_train.py`

---

## 🎯 성공 기준 (업데이트)

1. ✅ 기존 코드 100% 보존
2. ✅ Metadata 자동 파싱
3. ✅ CatEmbedV2 정확성 검증 (5개 테스트 통과)
4. ⏳ `--use_ebc` 플래그로 방식 선택 가능
5. ⏳ GPU 점유율 40%p 이상 증가
6. ⏳ Throughput 30% 이상 증가
7. ⏳ 학습 안정성 및 수렴 동등성 확인

**현재 달성률:** 3/7 (43%)

---

**최종 업데이트:** 2025-11-08 08:40 KST
**작성자:** AI Assistant
**버전:** 2.0 (정확성 검증 완료)
