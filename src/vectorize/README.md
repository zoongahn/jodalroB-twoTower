# Vectorize Module

학습된 Two-Tower 모델로부터 임베딩을 추출하고 벡터 DB에 저장하는 모듈입니다.

## 📁 구조

```
src/vectorize/
├── __init__.py                # 패키지 초기화
├── embedding_extractor.py     # 모델로부터 임베딩 추출
├── embedding_store.py          # Faiss 기반 벡터 DB
└── README.md                   # 문서 (이 파일)
```

## 🚀 사용 방법

### 1. 임베딩 추출 및 저장

```bash
# 기본 사용 (GPU)
python scripts/vectorize.py \
    --checkpoint output/models/20251026_040021/best_model.pt \
    --output data/vectorize/embeddings

# 배치 크기 조정 (GPU 메모리에 따라)
python scripts/vectorize.py \
    --checkpoint output/models/20251026_040021/best_model.pt \
    --output data/vectorize/embeddings \
    --batch_size 2048

# CPU 사용
python scripts/vectorize.py \
    --checkpoint output/models/20251026_040021/best_model.pt \
    --output data/vectorize/embeddings \
    --device cpu

# 조용한 모드 (진행 상황 최소화)
python scripts/vectorize.py \
    --checkpoint output/models/20251026_040021/best_model.pt \
    --output data/vectorize/embeddings \
    --quiet
```

### 2. 벡터 DB 로드 및 사용

```python
from src.vectorize import EmbeddingStore

# 벡터 DB 로드
store = EmbeddingStore()
store.load("data/vectorize/embeddings")

print(store)  # 통계 출력

# 단일 임베딩 조회 (복합키 사용)
notice_id = ("12345", "1")  # (bidntceno, bidntceord) 튜플
notice_emb = store.get_notice_embedding(notice_id)
print(f"Notice embedding shape: {notice_emb.shape}")

company_id = ("1234567890",)  # (bizno,) 튜플
company_emb = store.get_company_embedding(company_id)
print(f"Company embedding shape: {company_emb.shape}")

# 배치 조회
notice_ids = [("12345", "1"), ("12345", "2"), ("12345", "3")]
valid_ids, embeddings = store.get_notice_embeddings_batch(notice_ids)
print(f"Retrieved {len(valid_ids)} embeddings: {embeddings.shape}")
```

### 3. 벡터 DB 조회 및 검색

```bash
# 벡터 DB 통계 확인
python scripts/query_vector_db.py --db output/vector/embeddings_v1 --stats

# Notice 임베딩 조회
python scripts/query_vector_db.py --db output/vector/embeddings_v1 --notice "20230106038_000"

# Company 임베딩 조회
python scripts/query_vector_db.py --db output/vector/embeddings_v1 --company "8928101595"

# 유사한 Notice 검색 (Top-10)
python scripts/query_vector_db.py --db output/vector/embeddings_v1 --similar_notices "20230106038_000" --top_k 10

# 유사한 Company 검색 (Top-10)
python scripts/query_vector_db.py --db output/vector/embeddings_v1 --similar_companies "8928101595" --top_k 10

# Notice-Company 간 유사도 계산
python scripts/query_vector_db.py --db output/vector/embeddings_v1 --notice "20230106038_000" --company "8928101595" --similarity
```

### 4. 임베딩 시각화 (UMAP)

```bash
# Notice 임베딩 2D 시각화 (500개 샘플)
python test/visualize_embeddings_umap.py --db output/vector/embeddings_v1 --type notice --n_samples 500

# Company 임베딩 3D 시각화 (300개 샘플)
python test/visualize_embeddings_umap.py --db output/vector/embeddings_v1 --type company --n_samples 300 --n_components 3

# Notice + Company 동시 시각화 (2D)
python test/visualize_embeddings_umap.py --db output/vector/embeddings_v1 --type both --n_samples 300 --output output/visualizations/umap_both.html

# UMAP 파라미터 커스터마이징
python test/visualize_embeddings_umap.py --db output/vector/embeddings_v1 --type notice --n_neighbors 30 --min_dist 0.05 --metric euclidean
```

### 5. 임베딩 분석

```bash
# 저장된 임베딩 분석 (품질 검증)
python test/analyze_embeddings.py
```

## 📊 주요 기능

### EmbeddingExtractor

모델로부터 임베딩을 추출하는 클래스

**주요 메서드:**
- `load_model(schema)`: 모델 체크포인트 로드
- `extract_notice_embeddings(notice_store, batch_size)`: Notice 임베딩 추출
- `extract_company_embeddings(company_store, batch_size)`: Company 임베딩 추출
- `extract_all_embeddings(db_engine, schema)`: 전체 임베딩 추출 (원스톱)

**예제:**
```python
from src.vectorize import EmbeddingExtractor
from data.database_connector import DatabaseConnector
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta

# 초기화
extractor = EmbeddingExtractor(
    checkpoint_path="output/models/best_model.pt",
    device="cuda"
)

# 스키마 로드
schema_config = {
    "pair_notice_id_cols": ["bidntceno", "bidntceord"],
    "pair_company_id_cols": ["bizno"],
    "metadata_path": "meta/metadata.csv"
}
schema = build_torchrec_schema_from_meta(**schema_config)

# DB 연결
db = DatabaseConnector()

# 전체 임베딩 추출
(notice_ids, notice_embs), (company_ids, company_embs) = \
    extractor.extract_all_embeddings(
        db_engine=db.engine,
        schema=schema,
        batch_size=1024
    )

print(f"Notice: {notice_embs.shape}, Company: {company_embs.shape}")
```

### EmbeddingStore

Faiss 기반 벡터 데이터베이스

**주요 메서드:**
- `add_notices(notice_ids, embeddings)`: Notice 임베딩 추가
- `add_companies(company_ids, embeddings)`: Company 임베딩 추가
- `get_notice_embedding(notice_id)`: 단일 Notice 임베딩 조회
- `get_company_embedding(company_id)`: 단일 Company 임베딩 조회
- `get_notice_embeddings_batch(notice_ids)`: 배치 Notice 조회
- `get_company_embeddings_batch(company_ids)`: 배치 Company 조회
- `save(path_prefix)`: 디스크에 저장
- `load(path_prefix)`: 디스크에서 로드
- `get_stats()`: 통계 정보 반환

**예제:**
```python
from src.vectorize import EmbeddingStore
import numpy as np

# 새 벡터 DB 생성
store = EmbeddingStore(dimension=128)

# 임베딩 추가 (복합키 사용)
notice_ids = [("12345", "1"), ("12345", "2")]  # (bidntceno, bidntceord) 튜플
notice_embeddings = np.random.randn(2, 128).astype('float32')
store.add_notices(notice_ids, notice_embeddings)

company_ids = [("1234567890",), ("9876543210",)]  # (bizno,) 튜플
company_embeddings = np.random.randn(2, 128).astype('float32')
store.add_companies(company_ids, company_embeddings)

# 저장
store.save("data/vectorize/my_embeddings")

# 불러오기
new_store = EmbeddingStore()
new_store.load("data/vectorize/my_embeddings")

# 통계
stats = new_store.get_stats()
print(stats)
```

## 🔧 고급 사용법

### 커스텀 전처리 파이프라인

```python
from src.vectorize import EmbeddingExtractor, EmbeddingStore
from src.torchrec_preprocess.feature_preprocessor import FeaturePreprocessor

# 1. 임베딩 추출기 초기화
extractor = EmbeddingExtractor(
    checkpoint_path="output/models/best_model.pt",
    device="cuda"
)

# 2. 커스텀 전처리
preprocessor = FeaturePreprocessor(
    schema=schema,
    device="cuda",
    num_proj_dim=128,
    text_proj_dim=128
)

preprocessed_stores = preprocessor.preprocess_all(
    db_engine=db.engine,
    feature_chunksize=10000
)

# 3. 임베딩 추출
notice_ids, notice_embs = extractor.extract_notice_embeddings(
    preprocessed_stores['notice'],
    batch_size=2048
)

# 4. 벡터 DB에 저장 (복합키 그대로 사용)
store = EmbeddingStore(dimension=128)
# notice_ids는 이미 (bidntceno, bidntceord) 튜플 리스트
store.add_notices(notice_ids, notice_embs)
store.save("data/vectorize/custom")
```

### 증분 업데이트

기존 벡터 DB에 새 임베딩 추가:

```python
from src.vectorize import EmbeddingStore

# 기존 DB 로드
store = EmbeddingStore()
store.load("data/vectorize/embeddings")

print(f"기존: {store.get_stats()}")

# 새 임베딩 추가 (복합키 사용)
new_notice_ids = [("99999", "1"), ("99999", "2")]
new_embeddings = extractor.extract_notice_embeddings(new_notice_store)
store.add_notices(new_notice_ids, new_embeddings)

# 저장 (기존 파일 덮어쓰기)
store.save("data/vectorize/embeddings")

print(f"업데이트 후: {store.get_stats()}")
```

## 📝 저장 형식

벡터 DB 저장 시 생성되는 파일:

```
data/vectorize/embeddings_notice.index     # Faiss Notice 인덱스
data/vectorize/embeddings_company.index    # Faiss Company 인덱스
data/vectorize/embeddings_metadata.pkl     # ID 매핑 및 메타데이터
```

## ⚙️ 성능 최적화

### GPU 메모리 최적화

```bash
# 배치 크기를 줄여서 GPU 메모리 사용량 감소
python scripts/vectorize.py \
    --checkpoint ... \
    --output ... \
    --batch_size 512  # 기본값 1024에서 감소
```

### CPU 사용 (GPU 없을 때)

```bash
python scripts/vectorize.py \
    --checkpoint ... \
    --output ... \
    --device cpu \
    --batch_size 256  # CPU는 더 작은 배치 사용
```

### 대용량 데이터 처리

```bash
# 피처 로딩 청크 크기 조정
python scripts/vectorize.py \
    --checkpoint ... \
    --output ... \
    --feature_chunksize 10000  # 기본값 5000에서 증가
```

## 🧪 테스트

```bash
# 벡터 DB 조회 (실제 데이터)
python scripts/query_vector_db.py --db output/vector/embeddings_v1 --stats

# 임베딩 분석 (품질 검증)
python test/analyze_embeddings.py

# 벡터 DB 메타데이터 검사
python test/inspect_vector_db.py
```

## 📖 참고

- Faiss 문서: https://github.com/facebookresearch/faiss
- Two-Tower 모델 아키텍처: `src/towers/two_tower_model.py`
- 피처 전처리: `src/torchrec_preprocess/`
