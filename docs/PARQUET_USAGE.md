# Parquet 변환 및 사용 가이드

이 가이드는 DB 테이블을 Parquet 파일로 변환하고, 학습 시 Parquet 파일을 사용하는 방법을 설명합니다.

## 개요

Parquet 파일 사용의 장점:
- **빠른 로딩 속도**: DB 쿼리 오버헤드 없이 직접 파일에서 로드
- **압축 저장**: Snappy 압축으로 저장 공간 절약
- **이식성**: DB 연결 없이 데이터 이동 및 공유 가능
- **재현성**: 동일한 데이터셋으로 반복 실험 가능

## Step 1: DB 테이블을 Parquet로 변환

### 기본 사용법

```bash
# 전체 테이블 변환 (기본 위치: data/parquet)
python preprocess/convert_to_parquet.py

# 출력 디렉토리 지정
python preprocess/convert_to_parquet.py --output /path/to/parquet/dir
```

### 옵션

```bash
# 청크 크기 조정 (메모리 사용량 조절)
python preprocess/convert_to_parquet.py --chunksize 1000000

# 일부 데이터만 변환 (테스트용)
python preprocess/convert_to_parquet.py --limit 100000

# 모든 옵션 사용
python preprocess/convert_to_parquet.py \
  --output data/parquet \
  --chunksize 500000 \
  --limit none
```

### 변환되는 테이블

1. **step1.notice_preprocessed** → `notice.parquet`
2. **step1.company_preprocessed** → `company.parquet`
3. **step1.bid_two_tower** → `pairs.parquet`

### 변환 시간 예상

- 전체 데이터: 약 10-30분 (데이터 크기에 따라 다름)
- 테스트 데이터 (limit=100000): 약 1-2분

## Step 2: Parquet 파일로 학습하기

### 기본 사용법

```bash
# Parquet 파일 사용 (기본 위치: data/parquet)
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --use_parquet \
  --batch_size 256 \
  --num_epochs 3

# 커스텀 Parquet 디렉토리 지정
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --use_parquet \
  --parquet_dir /path/to/custom/parquet \
  --batch_size 256 \
  --num_epochs 3
```

### 주요 차이점

#### DB 사용 시 (기존 방식)
```bash
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --batch_size 256 \
  --num_epochs 3
```

#### Parquet 사용 시 (새로운 방식)
```bash
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --use_parquet \
  --parquet_dir data/parquet \
  --batch_size 256 \
  --num_epochs 3
```

### 전체 예제

```bash
# 1. Parquet 변환 (최초 1회만 실행)
python preprocess/convert_to_parquet.py --output data/parquet

# 2. Parquet로 학습
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --use_parquet \
  --parquet_dir data/parquet \
  --batch_size 512 \
  --num_epochs 5 \
  --learning_rate 1e-3 \
  --categorical_embedding_dim 32 \
  --final_embedding_dim 128
```

## 제약사항

### 현재 지원되지 않는 기능

1. **Streaming 모드**: Parquet에서는 아직 streaming 모드를 지원하지 않습니다.
   - `--streaming` 옵션과 `--use_parquet`를 함께 사용하면 오류 발생

```bash
# ❌ 작동하지 않음
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --use_parquet \
  --streaming

# ✅ 올바른 사용
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --use_parquet
```

2. **선택적 feature 로딩**: 현재는 전체 테이블을 로드합니다.
   - 향후 버전에서 `pair_limit`에 따른 선택적 로딩 지원 예정

## 파일 구조

```
data/parquet/
├── notice.parquet      # Notice 피처 (PK: bidntceno, bidntceord)
├── company.parquet     # Company 피처 (PK: bizno)
└── pairs.parquet       # Pair 정보 (bidntceno, bidntceord, bizno)
```

## 데이터 검증

Parquet 파일이 올바르게 생성되었는지 확인:

```python
import pandas as pd

# Notice 데이터 확인
notice_df = pd.read_parquet('data/parquet/notice.parquet')
print(f"Notice rows: {len(notice_df):,}")
print(notice_df.head())

# Company 데이터 확인
company_df = pd.read_parquet('data/parquet/company.parquet')
print(f"Company rows: {len(company_df):,}")
print(company_df.head())

# Pairs 데이터 확인
pairs_df = pd.read_parquet('data/parquet/pairs.parquet')
print(f"Pairs rows: {len(pairs_df):,}")
print(pairs_df.head())
```

## 성능 비교

### 예상 성능 향상

| 항목 | DB 방식 | Parquet 방식 | 개선율 |
|------|---------|--------------|--------|
| 초기 로딩 시간 | 60-120초 | 10-20초 | 5-6배 빠름 |
| 메모리 사용량 | 유사 | 유사 | - |
| 학습 속도 | 기준 | 기준 | 동일 |

### 로딩 시간 비교 (예시)

```
DB 방식:
  Notice features: 100,000 rows (45초)
  Company features: 50,000 rows (30초)
  Total: ~75초

Parquet 방식:
  Notice features: 100,000 rows (8초)
  Company features: 50,000 rows (5초)
  Total: ~13초
```

## 문제 해결

### Parquet 파일을 찾을 수 없음

```
FileNotFoundError: Parquet 파일을 찾을 수 없습니다: data/parquet/notice.parquet
```

**해결 방법:**
1. Parquet 파일이 생성되었는지 확인
   ```bash
   ls -lh data/parquet/
   ```

2. Parquet 변환 실행
   ```bash
   python preprocess/convert_to_parquet.py
   ```

### Streaming 모드 오류

```
NotImplementedError: Streaming mode with Parquet is not yet supported
```

**해결 방법:**
`--streaming` 옵션을 제거하고 실행

```bash
# ❌ 오류 발생
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py --use_parquet --streaming

# ✅ 올바른 사용
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py --use_parquet
```

## 권장 워크플로우

### 개발 단계
1. 소량 데이터로 Parquet 생성 (--limit 사용)
2. Parquet로 빠른 반복 실험
3. 하이퍼파라미터 튜닝

```bash
# 개발용: 소량 데이터 변환
python preprocess/convert_to_parquet.py --limit 50000 --output data/parquet_dev

# 개발용: 빠른 학습
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --use_parquet \
  --parquet_dir data/parquet_dev \
  --batch_size 256 \
  --num_epochs 2
```

### 프로덕션 단계
1. 전체 데이터로 Parquet 생성
2. 최종 하이퍼파라미터로 학습
3. 모델 저장 및 배포

```bash
# 프로덕션용: 전체 데이터 변환
python preprocess/convert_to_parquet.py --output data/parquet_prod

# 프로덕션용: 전체 학습
PYTHONPATH=/data/dev/jodalroB-twoTower python scripts/train.py \
  --use_parquet \
  --parquet_dir data/parquet_prod \
  --batch_size 512 \
  --num_epochs 10 \
  --learning_rate 1e-3
```

## 추가 정보

### 관련 파일
- `preprocess/convert_to_parquet.py`: DB → Parquet 변환 스크립트
- `preprocess/feature_store.py`: Parquet 로더 구현
- `src/towers/pairs/pair_loader.py`: DataLoader에 Parquet 지원 추가
- `scripts/train.py`: 학습 스크립트 (--use_parquet 옵션)

### 향후 개선 사항
- [ ] Streaming 모드 지원
- [ ] 선택적 feature 로딩 (pair_limit 기반)
- [ ] 증분 변환 (변경된 데이터만 업데이트)
- [ ] 멀티프로세싱 기반 병렬 변환
