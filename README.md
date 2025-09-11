# Two-Tower Model for Bid Recommendation

입찰 추천을 위한 Two-Tower 모델 구현 (PyTorch + TorchRec)

## 📋 프로젝트 개요

- **목표**: 공고(Notice)와 기업(Company) 간의 입찰 매칭을 위한 추천 시스템
- **아키텍처**: Two-Tower Neural Network (Retrieval Model)
- **프레임워크**: PyTorch, TorchRec
- **데이터**: 공고 피처 + 기업 피처 + 입찰 페어 데이터

## 🏗 시스템 아키텍처

```
Notice Tower                    Company Tower
    ↓                               ↓
Dense Features                  Dense Features
Categorical Features           Categorical Features
    ↓                               ↓
Feature Projection             Feature Projection
    ↓                               ↓
Hidden Layers [128, 64]        Hidden Layers [128, 64]
    ↓                               ↓
Embedding (64D)                Embedding (64D)
    ↓                               ↓
    └─────────> Cosine Similarity <─────────┘
                      ↓
                  Cross Entropy Loss
```

## 🚀 성능 최적화 이력

### 1. 초기 구현 (Baseline)
- **속도**: ~11 batch/s (순차 처리)
- **GPU 활용률**: 40%
- **방식**: 단순 DataLoader + collate_fn

### 2. 파이프라인 시도들

#### AsyncBatchPreprocessor (실패)
- 비동기 전처리 시도
- 결과: 가짜 파이프라인, 실제 오버랩 없음

#### TrueOverlapPipeline (실패)
- 복잡한 멀티스레딩 구조
- StreamingDataLoaderManager + BatchProcessorPool
- **문제**: 과도한 오버헤드로 4.63 it/s로 느려짐 (5배 감소!)
- **원인**: Python GIL + 불필요한 동기화

### 3. 현재 솔루션 (순차 처리 + DataLoader 최적화)
- **방식**: 단순 순차 처리로 복원
- **최적화**:
  - `num_workers`: 0 → 12 (멀티프로세스 데이터 로딩)
  - `prefetch_factor`: 2 → 4
  - `persistent_workers`: True
  - `pin_memory`: True
- **목표**: 23+ it/s 회복

## 📁 주요 파일 구조

```
jodalroB-twoTower/
├── scripts/
│   └── train.py                    # 메인 학습 스크립트
├── src/
│   ├── towers/
│   │   ├── two_tower_train_task.py # Two-Tower 모델 정의
│   │   └── pairs/
│   │       └── unified_bid_data_loader.py  # 데이터로더
│   ├── torchrec_preprocess/
│   │   ├── feature_preprocessor.py # 피처 전처리 (pre-projection)
│   │   ├── schema.py               # TorchRec 스키마
│   │   └── feature_store.py        # 피처 스토어
│   └── training/
│       ├── pipeline_wrapper.py     # (deprecated) 초기 파이프라인
│       ├── async_batch_preprocessor.py # (deprecated) 비동기 전처리
│       └── true_overlap_pipeline.py    # (deprecated) 복잡한 파이프라인
└── data/
    └── database_connector.py       # DB 연결
```

## 🔧 주요 설정

### 모델 하이퍼파라미터
```python
- categorical_embedding_dim: 32
- tower_hidden_dims: [128, 64]
- final_embedding_dim: 64
- dropout_rate: 0.1
- learning_rate: 1e-3
- batch_size: 256
```

### 데이터 설정
```python
- chunk_size: 1,000,000  # 청크 단위 로딩
- feature_limit: 100,000  # 피처 데이터 제한
- test_split: 0.2
- num_workers: 12         # DataLoader 워커
```

## 📊 성능 메트릭

| 구현 | 속도 (it/s) | GPU 활용률 | 비고 |
|------|------------|-----------|------|
| Baseline | 23 | 40% | 순차 처리 |
| AsyncBatchPreprocessor | ~23 | 40% | 가짜 파이프라인 |
| TrueOverlapPipeline | 4.63 | <20% | 오버헤드 과다 |
| **현재 (최적화)** | **목표: 45+** | **목표: 80%** | DataLoader 최적화 |

## 🎯 핵심 교훈

1. **복잡한 병렬화가 항상 답은 아니다**
   - Python GIL 제약 고려 필요
   - 오버헤드 vs 이익 분석 중요

2. **DataLoader 자체 최적화가 효과적**
   - PyTorch의 내장 멀티프로세싱 활용
   - num_workers 조정이 핵심

3. **프로파일링의 중요성**
   - 실제 병목 지점 파악 필수
   - 추측이 아닌 측정 기반 최적화

## 🚦 실행 방법

```bash
# 학습 실행
python scripts/train.py

# 주요 로그 확인 포인트
- "Training (Sequential)" 진행률 바의 it/s
- 배치 처리 속도
- GPU 메모리 사용량
```

## 📈 다음 단계

1. **CUDA Stream 활용**
   - GPU 전송과 연산 오버랩
   - torch.cuda.Stream() 활용

2. **Mixed Precision Training**
   - FP16/BF16 활용
   - 메모리 절약 + 속도 향상

3. **Distributed Training**
   - 멀티 GPU 활용
   - DDP (DistributedDataParallel)

## 🐛 알려진 이슈

- streaming=True 모드에서 청크 로딩 시 간헐적 지연
- 첫 번째 에포크 시작 시 피처 로딩으로 인한 초기 지연

## 📝 참고 자료

- [PyTorch TorchRec Two Tower Example](https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py)
- [TorchRec Documentation](https://pytorch.org/torchrec/)

---

**Last Updated**: 2025-09-11  
**Status**: 성능 최적화 진행 중 (순차 처리 + DataLoader 최적화)
