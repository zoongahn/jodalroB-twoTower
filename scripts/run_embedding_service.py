#!/usr/bin/env python
"""
Embedding Service CLI - 공고/업체 임베딩 추출 실행 스크립트

사용 예제:
    # 단일 공고 임베딩 추출
    python scripts/run_embedding_service.py --checkpoint output/models/best_model.pt --table notice --id 20230106038-00

    # 단일 업체 임베딩 추출
    python scripts/run_embedding_service.py --checkpoint output/models/best_model.pt --table company --id 1234567890

    # 여러 공고 임베딩 추출 (쉼표로 구분)
    python scripts/run_embedding_service.py --checkpoint output/models/best_model.pt --table notice --ids "20230106038-00,20230106039-00"

    # 파일에서 ID 목록 읽기
    python scripts/run_embedding_service.py --checkpoint output/models/best_model.pt --table notice --id-file ids.txt

    # 결과를 파일로 저장
    python scripts/run_embedding_service.py --checkpoint output/models/best_model.pt --table notice --id 20230106038-00 --output embeddings.npy

    # DB WHERE 조건으로 조회 (대량 데이터)
    python scripts/run_embedding_service.py --checkpoint output/models/best_model.pt --table notice --where "bidntceno LIKE '2023%'" --output /data/embeddings.npz

    # 전체 테이블 조회
    python scripts/run_embedding_service.py --checkpoint output/models/best_model.pt --table company --all --output /data/company_embeddings.npz
"""

import argparse
import sys
import os
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# .env 파일 로드
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from sqlalchemy import create_engine, text
from src.prediction.embedding_service import EmbeddingService
from preprocess.torchrec.feature_store import FeatureStore, load_feature_store_from_parquet


def get_db_url_from_env() -> str:
    """환경변수에서 DB URL 생성"""
    host = os.getenv('POSTGRES_HOST', 'localhost').strip()
    port = os.getenv('POSTGRES_PORT', '5432').strip()
    user = os.getenv('POSTGRES_USER', 'postgres').strip()
    password = os.getenv('POSTGRES_PASSWORD', '').strip()
    db = os.getenv('POSTGRES_DB', 'postgres').strip()
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def parse_notice_id(id_str: str) -> Tuple[str, str]:
    """공고 ID 문자열을 (bidntceno, bidntceord)로 파싱"""
    if '-' in id_str:
        parts = id_str.split('-')
        return parts[0], parts[1]
    else:
        return id_str, "00"


def load_ids_from_file(file_path: str) -> List[str]:
    """파일에서 ID 목록 로드 (한 줄에 하나씩)"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def extract_embeddings_from_parquet(
    service: EmbeddingService,
    parquet_path: str,
    table: str,
    batch_size: int = 100000
) -> Tuple[List, np.ndarray]:
    """
    Parquet 파일에서 직접 데이터를 로드하여 임베딩 추출

    Args:
        service: EmbeddingService 인스턴스
        parquet_path: Parquet 파일 경로
        table: 테이블 타입 ('notice' 또는 'company')
        batch_size: 배치 처리 크기

    Returns:
        (valid_ids, embeddings) 튜플
    """
    import torch
    from src.towers.kjt_utils import create_kjt_from_batch_gpu

    # 스키마 선택
    if table == 'notice':
        schema = service.schema.notice
    else:
        schema = service.schema.company

    # Parquet에서 데이터 로드
    print(f"\nParquet 파일 로드 중...")
    feature_data = load_feature_store_from_parquet(
        parquet_path=parquet_path,
        side_schema=schema,
        show_progress=True,
        use_chunked=True,
        chunksize=batch_size
    )

    valid_ids = feature_data["ids"]
    total_count = len(valid_ids)

    if total_count == 0:
        print("Parquet 파일에 데이터가 없습니다.")
        return [], np.array([])

    print(f"로드된 데이터 수: {total_count:,}")

    # FeatureStore 형태로 구성 (train과 동일한 구조)
    class ParquetFeatureStore:
        """Parquet 데이터를 FeatureStore 인터페이스로 제공"""
        def __init__(self, ids, numeric, categorical, text):
            self._key_to_row = {key: idx for idx, key in enumerate(ids)}
            self._num_mat = numeric
            self._cat_mat = categorical
            self._txt_mat = text

    full_store = ParquetFeatureStore(
        ids=valid_ids,
        numeric=feature_data["numeric"],
        categorical=feature_data["categorical"],
        text=feature_data["text"]
    )

    # 배치 단위로 처리 (train과 동일한 방식)
    all_embeddings = []
    num_batches = (total_count + batch_size - 1) // batch_size

    print(f"배치 처리 중... (배치 크기: {batch_size:,})")

    service.model.eval()
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_count)

            # 배치 FeatureStore 생성
            batch_ids = valid_ids[start_idx:end_idx]
            batch_store = ParquetFeatureStore(
                ids=batch_ids,
                numeric=full_store._num_mat[start_idx:end_idx] if full_store._num_mat is not None else None,
                categorical=full_store._cat_mat[start_idx:end_idx] if full_store._cat_mat is not None else None,
                text={col: emb[start_idx:end_idx] for col, emb in full_store._txt_mat.items()} if full_store._txt_mat else None
            )

            # train과 동일: _extract_*_features -> _create_*_input -> model
            if table == 'notice':
                data = service._extract_notice_features(batch_store)
                batch_input = service._create_notice_input(data)
                embeddings = service.model.notice_tower(batch_input)
            else:
                data = service._extract_company_features(batch_store)
                batch_input = service._create_company_input(data)
                embeddings = service.model.company_tower(batch_input)

            all_embeddings.append(embeddings.cpu().numpy())

            # 진행 상황 출력
            processed = end_idx
            progress = (processed / total_count) * 100
            print(f"\r  배치 {batch_idx + 1}/{num_batches} ({progress:.1f}%)", end="", flush=True)

    print()  # 줄바꿈

    # 결과 합치기
    embeddings_np = np.vstack(all_embeddings)

    return valid_ids, embeddings_np


def extract_embeddings_with_condition(
    service: EmbeddingService,
    engine,
    table: str,
    where_condition: str = None,
    batch_size: int = 10000
) -> Tuple[List, np.ndarray]:
    """
    WHERE 조건으로 DB에서 직접 조회하여 임베딩 추출

    Args:
        service: EmbeddingService 인스턴스
        engine: SQLAlchemy 엔진
        table: 테이블 타입 ('notice' 또는 'company')
        where_condition: WHERE 조건 (None이면 전체 조회)
        batch_size: 배치 처리 크기

    Returns:
        (valid_ids, embeddings) 튜플
    """
    import torch
    from src.towers.kjt_utils import create_kjt_from_batch_gpu

    # 스키마 선택
    if table == 'notice':
        schema = service.schema.notice
    else:
        schema = service.schema.company

    # FeatureStore로 데이터 로드
    print(f"\n데이터 로드 중...")
    store = FeatureStore(
        engine=engine,
        side_schema=schema,
        chunksize=batch_size,
        limit=None,
        where_condition=where_condition
    )
    store.build(show_progress=True)

    total_count = len(store._key_to_row)
    if total_count == 0:
        print("조건에 맞는 데이터가 없습니다.")
        return [], np.array([])

    print(f"로드된 데이터 수: {total_count:,}")

    # Feature 추출
    if table == 'notice':
        data = service._extract_notice_features(store)
    else:
        data = service._extract_company_features(store)

    # 배치 처리로 임베딩 추출
    all_embeddings = []
    num_batches = (total_count + batch_size - 1) // batch_size

    print(f"임베딩 추출 중... (배치 크기: {batch_size:,})")

    service.model.eval()
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_count)

            # 배치 데이터 추출
            batch_dense = data['dense_projected'][start_idx:end_idx]
            batch_cat = data['categorical'][start_idx:end_idx]

            # 텐서 변환
            dense_tensor = torch.from_numpy(batch_dense).float().to(service.device)
            cat_tensor = torch.from_numpy(batch_cat).long().to(service.device)

            # KJT 생성
            if table == 'notice':
                kjt = create_kjt_from_batch_gpu(cat_tensor, service.schema.notice.categorical, service.device)
                batch_input = {"dense": dense_tensor, "kjt": kjt}
                embeddings = service.model.notice_tower(batch_input)
            else:
                kjt = create_kjt_from_batch_gpu(cat_tensor, service.schema.company.categorical, service.device)
                batch_input = {"dense": dense_tensor, "kjt": kjt}
                embeddings = service.model.company_tower(batch_input)

            all_embeddings.append(embeddings.cpu().numpy())

            # 진행 상황 출력
            processed = end_idx
            progress = (processed / total_count) * 100
            print(f"\r  배치 {batch_idx + 1}/{num_batches} ({progress:.1f}%)", end="", flush=True)

    print()  # 줄바꿈

    # 결과 합치기
    embeddings_np = np.vstack(all_embeddings)

    # ID 추출
    valid_ids = list(store._key_to_row.keys())

    return valid_ids, embeddings_np


def main():
    parser = argparse.ArgumentParser(
        description='Embedding Service CLI - 공고/업체 임베딩 추출',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 필수 인자
    parser.add_argument(
        '--checkpoint', '-c',
        required=True,
        help='모델 체크포인트 경로 (예: output/models/best_model.pt)'
    )
    parser.add_argument(
        '--table', '-t',
        required=True,
        choices=['notice', 'company'],
        help='테이블 선택 (notice 또는 company)'
    )

    # ID 지정 방식 (상호 배타적)
    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument(
        '--id', '-i',
        help='단일 ID (공고: bidntceno-bidntceord, 업체: bizno)'
    )
    id_group.add_argument(
        '--ids',
        help='여러 ID (쉼표로 구분)'
    )
    id_group.add_argument(
        '--id-file', '-f',
        help='ID 목록 파일 경로 (한 줄에 하나씩)'
    )
    id_group.add_argument(
        '--where', '-w',
        help='DB WHERE 조건 (예: "bidntceno LIKE \'2023%%\'")'
    )
    id_group.add_argument(
        '--all', '-a',
        action='store_true',
        help='테이블 전체 데이터 조회'
    )
    id_group.add_argument(
        '--parquet', '-p',
        help='Parquet 파일 경로 (기본: data/parquet/{table}.parquet)'
    )
    id_group.add_argument(
        '--parquet-default',
        action='store_true',
        help='기본 Parquet 경로 사용 (data/parquet/{table}.parquet)'
    )

    # 선택적 인자
    parser.add_argument(
        '--output', '-o',
        help='결과 저장 경로 (.npy, .npz, .csv) (기본: data/embeddings/{table}.npz)'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='CSV 형식으로도 저장 (NPZ와 함께)'
    )
    parser.add_argument(
        '--config',
        help='모델 설정 파일 경로 (기본: 자동 탐색)'
    )
    parser.add_argument(
        '--device', '-d',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='사용할 디바이스 (기본: cuda)'
    )
    parser.add_argument(
        '--db-url',
        default=None,
        help='데이터베이스 URL (기본: .env 환경변수 사용)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 출력'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='배치 처리 크기 (기본: 10000)'
    )

    args = parser.parse_args()

    # 데이터베이스 연결
    db_url = args.db_url if args.db_url else get_db_url_from_env()
    engine = create_engine(db_url)

    # Parquet 기본 경로 설정
    if args.parquet_default:
        args.parquet = f"data/parquet/{args.table}.parquet"

    # WHERE 또는 ALL 또는 Parquet 모드인지 확인
    use_where_mode = args.where is not None or args.all
    use_parquet_mode = args.parquet is not None

    if use_parquet_mode:
        # Parquet 모드
        print(f"Parquet 파일 로드 모드: {args.parquet}")
        id_list = None
        where_condition = None
    elif use_where_mode:
        # WHERE 조건 또는 전체 조회 모드
        where_condition = args.where if args.where else None
        print(f"DB 조건 조회 모드")
        if where_condition:
            print(f"WHERE 조건: {where_condition}")
        else:
            print("전체 테이블 조회")
        id_list = None
    else:
        # ID 목록 준비
        if args.id:
            id_list = [args.id]
        elif args.ids:
            id_list = [id.strip() for id in args.ids.split(',')]
        else:
            id_list = load_ids_from_file(args.id_file)

        if not id_list:
            print("오류: ID가 지정되지 않았습니다.")
            sys.exit(1)

        print(f"처리할 ID 수: {len(id_list)}")
        where_condition = None

    # EmbeddingService 초기화
    service = EmbeddingService(
        checkpoint_path=args.checkpoint,
        db_engine=engine,
        config_path=args.config,
        device=args.device
    )

    # 임베딩 추출
    if use_parquet_mode:
        # Parquet 모드
        valid_ids, embeddings = extract_embeddings_from_parquet(
            service=service,
            parquet_path=args.parquet,
            table=args.table,
            batch_size=args.batch_size
        )
    elif use_where_mode:
        # WHERE 조건 모드 - FeatureStore 직접 사용
        valid_ids, embeddings = extract_embeddings_with_condition(
            service=service,
            engine=engine,
            table=args.table,
            where_condition=where_condition,
            batch_size=args.batch_size
        )
    elif args.table == 'notice':
        if len(id_list) == 1:
            # 단일 공고
            bidntceno, bidntceord = parse_notice_id(id_list[0])
            embedding = service.get_notice_embedding(bidntceno, bidntceord)

            if embedding is not None:
                valid_ids = [(bidntceno, bidntceord)]
                embeddings = embedding.reshape(1, -1)
            else:
                valid_ids = []
                embeddings = np.array([])
        else:
            # 배치 공고
            notice_ids = [parse_notice_id(id_str) for id_str in id_list]
            valid_ids, embeddings = service.get_notice_embeddings_batch(notice_ids)

    else:  # company
        if len(id_list) == 1:
            # 단일 업체
            embedding = service.get_company_embedding(id_list[0])

            if embedding is not None:
                valid_ids = [id_list[0]]
                embeddings = embedding.reshape(1, -1)
            else:
                valid_ids = []
                embeddings = np.array([])
        else:
            # 배치 업체
            valid_ids, embeddings = service.get_company_embeddings_batch(id_list)

    # 결과 출력
    print("\n" + "=" * 80)
    print("결과")
    print("=" * 80)

    if len(valid_ids) == 0:
        print("임베딩을 추출할 수 없습니다. ID를 확인해주세요.")
        sys.exit(1)

    print(f"추출된 임베딩 수: {len(valid_ids)}")
    print(f"임베딩 차원: {embeddings.shape[1]}")

    if args.verbose or len(valid_ids) <= 5:
        print("\n추출된 ID:")
        for i, vid in enumerate(valid_ids):
            if args.table == 'notice':
                print(f"  {i+1}. {vid[0]}-{vid[1]}")
            else:
                print(f"  {i+1}. {vid}")

            if args.verbose:
                emb = embeddings[i]
                print(f"      shape: {emb.shape}, min: {emb.min():.4f}, max: {emb.max():.4f}, mean: {emb.mean():.4f}")

    # 결과 저장
    # 기본 출력 경로 설정
    if not args.output:
        os.makedirs('data/embeddings', exist_ok=True)
        args.output = f"data/embeddings/{args.table}.npz"

    if args.output:
        # ID 배열 생성
        if args.table == 'notice':
            id_array = np.array([f"{v[0]}-{v[1]}" for v in valid_ids])
        else:
            id_array = np.array(valid_ids)

        # NPZ/NPY 저장
        if args.output.endswith('.npz'):
            np.savez(args.output, ids=id_array, embeddings=embeddings)
            print(f"\n✓ NPZ 저장: {args.output}")
        elif args.output.endswith('.npy'):
            np.save(args.output, embeddings)
            print(f"\n✓ NPY 저장: {args.output}")
        elif args.output.endswith('.csv'):
            # CSV 저장
            import pandas as pd
            df = pd.DataFrame(embeddings)
            df.insert(0, 'id', id_array)
            df.to_csv(args.output, index=False)
            print(f"\n✓ CSV 저장: {args.output}")
        else:
            # 확장자 없으면 npz로
            np.savez(args.output, ids=id_array, embeddings=embeddings)
            print(f"\n✓ NPZ 저장: {args.output}")

        # CSV 추가 저장
        if args.csv and not args.output.endswith('.csv'):
            csv_path = args.output.rsplit('.', 1)[0] + '.csv'
            import pandas as pd
            df = pd.DataFrame(embeddings)
            df.insert(0, 'id', id_array)
            df.to_csv(csv_path, index=False)
            print(f"✓ CSV 추가 저장: {csv_path}")

    # 단일 결과인 경우 임베딩 값 일부 출력
    if len(valid_ids) == 1 and not args.output:
        print(f"\n임베딩 벡터 (처음 10개 값):")
        print(embeddings[0][:10])

    print("\n완료!")


if __name__ == '__main__':
    main()
