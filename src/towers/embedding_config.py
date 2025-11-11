"""
TorchRec EmbeddingBagConfig 생성 모듈

metadata.csv 기반으로 EmbeddingBagCollection용 설정 자동 생성
"""

from typing import List, Dict, Optional
from torchrec import EmbeddingBagConfig
from torchrec.modules.embedding_configs import PoolingType
from preprocess.metadata_parser import MetadataParser


def create_embedding_bag_configs(
    table_name: str,
    metadata_path: str = "meta/metadata.csv",
    embedding_dim: int = 32,
    pooling_mode: PoolingType = PoolingType.MEAN,
    add_unk_token: bool = True
) -> List[EmbeddingBagConfig]:
    """
    특정 테이블의 범주형 피처에 대한 EmbeddingBagConfig 리스트 생성

    Args:
        table_name: 테이블명 ("notice" or "company")
        metadata_path: metadata.csv 경로
        embedding_dim: 임베딩 차원
        pooling_mode: Pooling 방식 (PoolingType.MEAN, PoolingType.SUM)
        add_unk_token: UNK 토큰용 +1 추가 여부

    Returns:
        List[EmbeddingBagConfig]
    """
    parser = MetadataParser(metadata_path)

    # metadata에서 범주형 피처 정보 추출
    config_dicts = parser.generate_embedding_config_dict(
        table_name=table_name,
        embedding_dim=embedding_dim,
        add_unk_token=add_unk_token
    )

    # EmbeddingBagConfig 객체 생성
    configs = []
    for cfg in config_dicts:
        ebc = EmbeddingBagConfig(
            name=f"{table_name}_{cfg['name']}",  # 테이블명 prefix 추가
            embedding_dim=cfg['embedding_dim'],
            num_embeddings=cfg['num_embeddings'],
            feature_names=cfg['feature_names'],  # KJT의 keys와 매칭되어야 함
            pooling=pooling_mode
        )
        configs.append(ebc)

    return configs


def create_notice_embedding_configs(
    metadata_path: str = "meta/metadata.csv",
    embedding_dim: int = 32,
    pooling_mode: PoolingType = PoolingType.MEAN,
    add_unk_token: bool = True
) -> List[EmbeddingBagConfig]:
    """
    Notice 타워용 EmbeddingBagConfig 생성

    Args:
        metadata_path: metadata.csv 경로
        embedding_dim: 임베딩 차원
        pooling_mode: Pooling 방식
        add_unk_token: UNK 토큰 추가 여부

    Returns:
        List[EmbeddingBagConfig]
    """
    return create_embedding_bag_configs(
        table_name="notice",
        metadata_path=metadata_path,
        embedding_dim=embedding_dim,
        pooling_mode=pooling_mode,
        add_unk_token=add_unk_token
    )


def create_company_embedding_configs(
    metadata_path: str = "meta/metadata.csv",
    embedding_dim: int = 32,
    pooling_mode: PoolingType = PoolingType.MEAN,
    add_unk_token: bool = True
) -> List[EmbeddingBagConfig]:
    """
    Company 타워용 EmbeddingBagConfig 생성

    Args:
        metadata_path: metadata.csv 경로
        embedding_dim: 임베딩 차원
        pooling_mode: Pooling 방식
        add_unk_token: UNK 토큰 추가 여부

    Returns:
        List[EmbeddingBagConfig]
    """
    return create_embedding_bag_configs(
        table_name="company",
        metadata_path=metadata_path,
        embedding_dim=embedding_dim,
        pooling_mode=pooling_mode,
        add_unk_token=add_unk_token
    )


def get_embedding_configs_summary(configs: List[EmbeddingBagConfig]) -> Dict:
    """
    EmbeddingBagConfig 리스트의 요약 정보 반환

    Args:
        configs: EmbeddingBagConfig 리스트

    Returns:
        {
            "num_features": int,
            "total_embeddings": int,
            "total_params": int,
            "memory_mb": float,
            "features": [{"name": str, "vocab_size": int, "dim": int}, ...]
        }
    """
    num_features = len(configs)
    total_embeddings = sum(cfg.num_embeddings for cfg in configs)

    # 파라미터 수 계산 (vocab_size * embedding_dim)
    total_params = sum(cfg.num_embeddings * cfg.embedding_dim for cfg in configs)

    # 메모리 추정 (float32 = 4 bytes)
    memory_mb = total_params * 4 / 1024 / 1024

    features = [
        {
            "name": cfg.name,
            "vocab_size": cfg.num_embeddings,
            "dim": cfg.embedding_dim,
            "pooling": cfg.pooling
        }
        for cfg in configs
    ]

    return {
        "num_features": num_features,
        "total_embeddings": total_embeddings,
        "total_params": total_params,
        "memory_mb": memory_mb,
        "features": features
    }


def print_embedding_configs_summary(
    notice_configs: List[EmbeddingBagConfig],
    company_configs: List[EmbeddingBagConfig]
):
    """
    Notice/Company EmbeddingBagConfig 요약 출력

    Args:
        notice_configs: Notice용 configs
        company_configs: Company용 configs
    """
    notice_summary = get_embedding_configs_summary(notice_configs)
    company_summary = get_embedding_configs_summary(company_configs)

    print("=" * 80)
    print("EmbeddingBagCollection 설정 요약")
    print("=" * 80)

    print(f"\n📊 Notice Tower:")
    print(f"  - 범주형 피처 수: {notice_summary['num_features']}")
    print(f"  - 총 임베딩 개수: {notice_summary['total_embeddings']:,}")
    print(f"  - 총 파라미터 수: {notice_summary['total_params']:,}")
    print(f"  - 메모리 사용량:  {notice_summary['memory_mb']:.2f} MB")

    print(f"\n📊 Company Tower:")
    print(f"  - 범주형 피처 수: {company_summary['num_features']}")
    print(f"  - 총 임베딩 개수: {company_summary['total_embeddings']:,}")
    print(f"  - 총 파라미터 수: {company_summary['total_params']:,}")
    print(f"  - 메모리 사용량:  {company_summary['memory_mb']:.2f} MB")

    print(f"\n💾 전체:")
    total_params = notice_summary['total_params'] + company_summary['total_params']
    total_memory = notice_summary['memory_mb'] + company_summary['memory_mb']
    print(f"  - 총 파라미터 수: {total_params:,}")
    print(f"  - 총 메모리:      {total_memory:.2f} MB")

    # 상위 5개 대규모 피처
    print(f"\n🔝 상위 5개 대규모 피처:")
    all_features = notice_summary['features'] + company_summary['features']
    top5 = sorted(all_features, key=lambda x: x['vocab_size'], reverse=True)[:5]

    for i, feat in enumerate(top5, 1):
        params = feat['vocab_size'] * feat['dim']
        mem_mb = params * 4 / 1024 / 1024
        print(f"  {i}. {feat['name']:<40} vocab={feat['vocab_size']:>6,}  "
              f"dim={feat['dim']:>3}  params={params:>9,}  mem={mem_mb:>6.2f}MB")

    print("\n" + "=" * 80)


# ============================================================================
# 전역 상수 및 헬퍼 함수 (train.py에서 사용)
# ============================================================================

# 캐싱 (중복 생성 방지)
_NOTICE_CONFIGS_CACHE = None
_COMPANY_CONFIGS_CACHE = None


def get_notice_ebc_configs_cached(
    metadata_path: str = "meta/metadata.csv",
    embedding_dim: int = 32,
    pooling_mode: PoolingType = PoolingType.MEAN,
    add_unk_token: bool = True
) -> List[EmbeddingBagConfig]:
    """Notice EBC configs 캐싱 버전"""
    global _NOTICE_CONFIGS_CACHE
    if _NOTICE_CONFIGS_CACHE is None:
        _NOTICE_CONFIGS_CACHE = create_notice_embedding_configs(
            metadata_path, embedding_dim, pooling_mode, add_unk_token
        )
    return _NOTICE_CONFIGS_CACHE


def get_company_ebc_configs_cached(
    metadata_path: str = "meta/metadata.csv",
    embedding_dim: int = 32,
    pooling_mode: PoolingType = PoolingType.MEAN,
    add_unk_token: bool = True
) -> List[EmbeddingBagConfig]:
    """Company EBC configs 캐싱 버전"""
    global _COMPANY_CONFIGS_CACHE
    if _COMPANY_CONFIGS_CACHE is None:
        _COMPANY_CONFIGS_CACHE = create_company_embedding_configs(
            metadata_path, embedding_dim, pooling_mode, add_unk_token
        )
    return _COMPANY_CONFIGS_CACHE


def get_notice_feature_metadata(
    metadata_path: str = "meta/metadata.csv",
    embedding_dim: int = 32,
    add_unk_token: bool = True
) -> Dict:
    """
    Notice 피처 메타데이터 반환 (train.py용)

    Returns:
        {
            "feature_names": List[str],        # KJT keys
            "num_embeddings": List[int],       # 각 키의 vocab size
            "embedding_dim": int,
        }
    """
    configs = get_notice_ebc_configs_cached(metadata_path, embedding_dim, add_unk_token=add_unk_token)
    return {
        "feature_names": [cfg.feature_names[0] for cfg in configs],
        "num_embeddings": [cfg.num_embeddings for cfg in configs],
        "embedding_dim": embedding_dim,
    }


def get_company_feature_metadata(
    metadata_path: str = "meta/metadata.csv",
    embedding_dim: int = 32,
    add_unk_token: bool = True
) -> Dict:
    """
    Company 피처 메타데이터 반환 (train.py용)

    Returns:
        {
            "feature_names": List[str],        # KJT keys
            "num_embeddings": List[int],       # 각 키의 vocab size
            "embedding_dim": int,
        }
    """
    configs = get_company_ebc_configs_cached(metadata_path, embedding_dim, add_unk_token=add_unk_token)
    return {
        "feature_names": [cfg.feature_names[0] for cfg in configs],
        "num_embeddings": [cfg.num_embeddings for cfg in configs],
        "embedding_dim": embedding_dim,
    }


if __name__ == "__main__":
    # 테스트 실행
    print("EmbeddingBagConfig 생성 테스트\n")

    # Notice configs 생성
    notice_configs = create_notice_embedding_configs(
        embedding_dim=32,
        pooling_mode=PoolingType.MEAN,
        add_unk_token=True
    )

    # Company configs 생성
    company_configs = create_company_embedding_configs(
        embedding_dim=32,
        pooling_mode=PoolingType.MEAN,
        add_unk_token=True
    )

    # 요약 출력
    print_embedding_configs_summary(notice_configs, company_configs)

    # 상세 정보 샘플 출력
    print("\n📋 Notice 필드 상세 (처음 5개):")
    for i, cfg in enumerate(notice_configs[:5], 1):
        print(f"  {i}. {cfg.name}")
        print(f"     - num_embeddings: {cfg.num_embeddings}")
        print(f"     - embedding_dim: {cfg.embedding_dim}")
        print(f"     - pooling: {cfg.pooling}")
        print(f"     - feature_names: {cfg.feature_names}")

    print("\n📋 Company 필드 상세 (전체):")
    for i, cfg in enumerate(company_configs, 1):
        print(f"  {i}. {cfg.name}")
        print(f"     - num_embeddings: {cfg.num_embeddings}")
        print(f"     - embedding_dim: {cfg.embedding_dim}")
        print(f"     - pooling: {cfg.pooling}")
        print(f"     - feature_names: {cfg.feature_names}")
