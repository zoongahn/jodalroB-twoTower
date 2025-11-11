# Two-Tower 모델 예측 사용 가이드

## 개요

학습된 Two-Tower 모델을 사용하여 새로운 Notice에 대해 적합한 Company를 추천하는 방법을 설명합니다.

## 사전 준비

### 1. 모델 학습 완료
```bash
python scripts/train.py --num_epochs 5 --batch_size 256
```

학습 완료 후 다음 파일들이 생성됩니다:
- `output/models/{timestamp}/best_model.pt` - 최고 성능 모델
- `output/models/{timestamp}/final_model.pt` - 최종 모델
- `output/models/{timestamp}/config.json` - 모델 설정

### 2. 벡터 DB 생성
```bash
python scripts/vectorize.py \
    --checkpoint output/models/20251023_055430/best_model.pt \
    --output data/vectorize/embeddings
```

생성되는 파일들:
- `data/vectorize/embeddings_notice.index` - Notice 벡터 인덱스
- `data/vectorize/embeddings_company.index` - Company 벡터 인덱스
- `data/vectorize/embeddings_metadata.pkl` - 메타데이터

## 사용 방법

### 1. 단일 Notice 예측

```bash
python scripts/predict.py \
    --checkpoint output/models/20251023_055430/best_model.pt \
    --vector_db data/vectorize/embeddings \
    --notice "20230106038" "000" \
    --top_k 10
```

**출력 예시:**
```
예측 결과
================================================================================

Notice ID: ('20230106038', '000')

Top-10 추천 Company:
Rank   Company ID           Similarity Score
----------------------------------------------
1      ('1234567890',)      0.856234
2      ('0987654321',)      0.823456
3      ('1111111111',)      0.789012
...
```

### 2. 배치 예측

**입력 CSV 파일 (`notices.csv`):**
```csv
bidntceno,bidntceord
20230106038,000
20230106039,000
20230106040,000
```

**실행:**
```bash
python scripts/predict.py \
    --checkpoint output/models/20251023_055430/best_model.pt \
    --vector_db data/vectorize/embeddings \
    --batch notices.csv \
    --output predictions.csv \
    --top_k 10
```

**출력 CSV (`predictions.csv`):**
```csv
bidntceno,bidntceord,rank,company_id,similarity_score,error
20230106038,000,1,('1234567890',),0.856234,
20230106038,000,2,('0987654321',),0.823456,
20230106038,000,3,('1111111111',),0.789012,
...
```

### 3. Python 코드에서 직접 사용

```python
from data.database_connector import DatabaseConnector
from src.prediction import TwoTowerPredictor

# 데이터베이스 연결
db = DatabaseConnector()

# 예측기 초기화
predictor = TwoTowerPredictor(
    checkpoint_path="output/models/20251023_055430/best_model.pt",
    vector_db_path="data/vectorize/embeddings",
    db_engine=db.engine,
    device="cuda"
)

# 단일 예측
result = predictor.predict_for_notice(
    bidntceno="20230106038",
    bidntceord="000",
    top_k=10
)

print(f"Notice ID: {result['notice_id']}")
for rank, (company_id, score) in enumerate(result['top_k_companies'], 1):
    print(f"{rank}. {company_id}: {score:.4f}")

# 배치 예측
notice_ids = [
    ("20230106038", "000"),
    ("20230106039", "000"),
    ("20230106040", "000")
]

results = predictor.predict_for_notices_batch(
    notice_ids=notice_ids,
    top_k=10,
    show_progress=True
)

db.close()
```

## 고급 옵션

### CPU 사용
GPU가 없는 환경에서는 `--device cpu` 옵션 사용:
```bash
python scripts/predict.py \
    --checkpoint output/models/20251023_055430/best_model.pt \
    --vector_db data/vectorize/embeddings \
    --notice "20230106038" "000" \
    --device cpu
```

### 임베딩 벡터 함께 반환
```python
result = predictor.predict_for_notice(
    bidntceno="20230106038",
    bidntceord="000",
    top_k=10,
    return_embeddings=True  # 임베딩도 반환
)

notice_emb = result['notice_embedding']  # shape: (128,)
company_embs = result['company_embeddings']  # Dict[company_id, embedding]
```

## 아키텍처

### 예측 파이프라인
```
1. Notice 데이터 로드 (DB)
   ↓
2. 전처리 파이프라인 실행
   - Numeric/Text features → Dense projection
   - Categorical features → Embedding indices
   ↓
3. Notice Tower → Notice 임베딩 생성
   ↓
4. Vector DB에서 모든 Company 임베딩 로드
   ↓
5. 코사인 유사도 계산
   ↓
6. Top-K 추출 및 반환
```

### 주요 컴포넌트

#### `ModelLoader` (`src/prediction/model_loader.py`)
- 체크포인트 로드
- Config 복원
- Schema 초기화
- Preprocessor 준비

#### `TwoTowerPredictor` (`src/prediction/predictor.py`)
- Notice 데이터 로드 및 전처리
- 임베딩 생성
- Vector DB 통합
- 유사도 계산 및 Top-K 추천

#### `EmbeddingStore` (`src/vectorize/embedding_store.py`)
- Faiss 기반 벡터 DB
- 효율적인 임베딩 저장/조회
- L2 거리 기반 검색

## 성능 최적화

### GPU 메모리 절약
- 배치 예측 시 한 번에 하나씩 처리
- `return_embeddings=False`로 메모리 사용 최소화

### 속도 향상
- Vector DB를 메모리에 한 번만 로드
- GPU 사용 (`--device cuda`)
- 배치 예측 사용

## 문제 해결

### Q1. "Notice를 찾을 수 없습니다" 에러
**원인:** DB에 해당 Notice가 없음
**해결:** Notice ID를 확인하고 DB에 존재하는지 검증

### Q2. Vector DB 로딩 실패
**원인:** 벡터 DB 파일이 없거나 경로가 잘못됨
**해결:** `scripts/vectorize.py`를 먼저 실행하여 벡터 DB 생성

### Q3. GPU 메모리 부족
**원인:** 임베딩 차원이 크거나 Company 수가 많음
**해결:** `--device cpu` 사용 또는 배치 크기 줄이기

## 예제 시나리오

### 시나리오 1: 실시간 추천 API
```python
from flask import Flask, request, jsonify
from src.prediction import TwoTowerPredictor

app = Flask(__name__)

# 서버 시작 시 예측기 초기화 (한 번만)
predictor = TwoTowerPredictor(...)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    result = predictor.predict_for_notice(
        bidntceno=data['bidntceno'],
        bidntceord=data['bidntceord'],
        top_k=10
    )
    return jsonify(result)
```

### 시나리오 2: 오프라인 배치 추천
```bash
# 1. 추천할 Notice 목록 생성
psql -d mydb -c "COPY (SELECT bidntceno, bidntceord FROM notices WHERE created_at > '2023-01-01') TO '/tmp/notices.csv' CSV HEADER"

# 2. 배치 예측 실행
python scripts/predict.py \
    --checkpoint output/models/best_model.pt \
    --vector_db data/vectorize/embeddings \
    --batch /tmp/notices.csv \
    --output /tmp/predictions.csv

# 3. 결과를 DB에 저장
psql -d mydb -c "COPY recommendations FROM '/tmp/predictions.csv' CSV HEADER"
```

## 참고

- 학습: `scripts/train.py`
- 벡터화: `scripts/vectorize.py`
- 벡터 DB 조회: `scripts/query_vector_db.py`
- 평가: `test/evaluate.py`
