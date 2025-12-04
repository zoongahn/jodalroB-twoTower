"""
metadata.csv 파서 - 범주형 피처의 vocab_size 추출

EmbeddingBagCollection 구성에 필요한 정보:
- 사용 여부 = Y
- 범주형 여부 = Y
- 범주 갯수 (num_embeddings)
"""

import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


class MetadataParser:
    """metadata.csv에서 범주형 피처 정보 추출"""

    def __init__(self, metadata_path: str = "meta/metadata.csv"):
        """
        Args:
            metadata_path: metadata.csv 파일 경로
        """
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

    def get_categorical_features(
        self,
        table_name: str,
        use_only: bool = True
    ) -> Dict[str, int]:
        """
        특정 테이블의 범주형 피처와 vocab_size 추출

        Args:
            table_name: 테이블명 (예: "notice", "company")
            use_only: True면 "사용 여부 = Y"인 것만 반환

        Returns:
            {feature_name: vocab_size} 딕셔너리
        """
        # 필터링 조건
        table_df = self.df[self.df['테이블명'] == table_name].copy()

        # 범주형 여부 = Y
        cat_df = table_df[table_df['범주형 여부'] == 'Y'].copy()

        # 사용 여부 = Y (선택적)
        if use_only:
            cat_df = cat_df[cat_df['사용 여부'] == 'Y'].copy()

        # vocab_size 추출
        result = {}
        for _, row in cat_df.iterrows():
            col_name = row['컬럼명']
            vocab_size = row['범주 갯수']

            # vocab_size가 비어있거나 0이면 기본값 설정
            if pd.isna(vocab_size) or vocab_size == 0:
                print(f"⚠️  {table_name}.{col_name}: vocab_size가 없음 (기본값 100 사용)")
                vocab_size = 100
            else:
                vocab_size = int(vocab_size)

            result[col_name] = vocab_size

        return result

    def get_notice_categorical_config(self) -> Dict[str, int]:
        """Notice 테이블의 범주형 피처 설정"""
        return self.get_categorical_features("notice", use_only=True)

    def get_company_categorical_config(self) -> Dict[str, int]:
        """Company 테이블의 범주형 피처 설정"""
        return self.get_categorical_features("company", use_only=True)

    def get_all_categorical_config(self) -> Dict[str, Dict[str, int]]:
        """모든 테이블의 범주형 피처 설정"""
        return {
            "notice": self.get_notice_categorical_config(),
            "company": self.get_company_categorical_config()
        }

    def print_summary(self):
        """범주형 피처 요약 출력"""
        notice_config = self.get_notice_categorical_config()
        company_config = self.get_company_categorical_config()

        print("=" * 80)
        print("범주형 피처 Vocabulary 요약 (사용 여부 = Y만)")
        print("=" * 80)

        print(f"\n📊 Notice 테이블 ({len(notice_config)}개 범주형 피처):")
        print("-" * 80)
        print(f"{'컬럼명':<35} {'Vocab Size':>15} {'권장 num_embeddings':>25}")
        print("-" * 80)

        notice_total = 0
        for col_name, vocab_size in sorted(notice_config.items(), key=lambda x: -x[1]):
            recommended = self._recommend_num_embeddings(vocab_size)
            notice_total += vocab_size
            print(f"{col_name:<35} {vocab_size:>15,} {recommended:>25,}")

        print("-" * 80)
        print(f"{'총합':<35} {notice_total:>15,}")

        print(f"\n📊 Company 테이블 ({len(company_config)}개 범주형 피처):")
        print("-" * 80)
        print(f"{'컬럼명':<35} {'Vocab Size':>15} {'권장 num_embeddings':>25}")
        print("-" * 80)

        company_total = 0
        for col_name, vocab_size in sorted(company_config.items(), key=lambda x: -x[1]):
            recommended = self._recommend_num_embeddings(vocab_size)
            company_total += vocab_size
            print(f"{col_name:<35} {vocab_size:>15,} {recommended:>25,}")

        print("-" * 80)
        print(f"{'총합':<35} {company_total:>15,}")

        # 메모리 추정
        print("\n" + "=" * 80)
        print("💾 메모리 사용량 추정 (embedding_dim=32, float32 기준)")
        print("=" * 80)

        def estimate_memory(vocab_sizes, embedding_dim=32):
            total_embeddings = sum(self._recommend_num_embeddings(v) for v in vocab_sizes.values())
            return total_embeddings * embedding_dim * 4 / 1024 / 1024  # MB

        notice_mem = estimate_memory(notice_config)
        company_mem = estimate_memory(company_config)

        print(f"Notice Embedding 테이블:  {notice_mem:>10.2f} MB")
        print(f"Company Embedding 테이블: {company_mem:>10.2f} MB")
        print(f"총 메모리:                {notice_mem + company_mem:>10.2f} MB")

        print("\n" + "=" * 80)

    def _recommend_num_embeddings(self, vocab_size: int) -> int:
        """
        현재 vocab_size에서 여유를 둔 num_embeddings 권장값

        Args:
            vocab_size: 현재 고유값 개수

        Returns:
            권장 num_embeddings (여유 포함)
        """
        if vocab_size < 10:
            return 20  # 100% 여유
        elif vocab_size < 100:
            return int(vocab_size * 2)  # 100% 여유
        elif vocab_size < 1000:
            return int(vocab_size * 1.5)  # 50% 여유
        elif vocab_size < 10000:
            return int(vocab_size * 1.3)  # 30% 여유
        else:
            return int(vocab_size * 1.2)  # 20% 여유 (대규모)

    def generate_embedding_config_dict(
        self,
        table_name: str,
        embedding_dim: int = 32,
        add_unk_token: bool = True
    ) -> List[Dict]:
        """
        EmbeddingBagConfig 생성에 필요한 정보 딕셔너리 리스트 반환

        Args:
            table_name: 테이블명
            embedding_dim: 임베딩 차원
            add_unk_token: UNK 토큰용 +1 추가 여부

        Returns:
            [{"name": str, "num_embeddings": int, "embedding_dim": int}, ...]
        """
        cat_config = self.get_categorical_features(table_name, use_only=True)

        result = []
        for col_name, vocab_size in cat_config.items():
            recommended = self._recommend_num_embeddings(vocab_size)

            # UNK 토큰용 +1
            if add_unk_token:
                recommended += 1

            result.append({
                "name": col_name,
                "num_embeddings": recommended,
                "embedding_dim": embedding_dim,
                "feature_names": [col_name]
            })

        return result


# 편의 함수
def load_categorical_vocab_sizes(
    metadata_path: str = "meta/metadata.csv"
) -> Dict[str, Dict[str, int]]:
    """
    metadata.csv에서 범주형 vocab_size 로딩

    Returns:
        {
            "notice": {col_name: vocab_size, ...},
            "company": {col_name: vocab_size, ...}
        }
    """
    parser = MetadataParser(metadata_path)
    return parser.get_all_categorical_config()


if __name__ == "__main__":
    # 테스트 실행
    parser = MetadataParser()

    # 요약 출력
    parser.print_summary()

    # EmbeddingBagConfig용 딕셔너리 생성 예시
    print("\n" + "=" * 80)
    print("EmbeddingBagConfig 생성 예시")
    print("=" * 80)

    notice_configs = parser.generate_embedding_config_dict("notice", embedding_dim=32)
    print(f"\nNotice 필드 개수: {len(notice_configs)}")
    print("\n처음 3개 예시:")
    for cfg in notice_configs[:3]:
        print(f"  {cfg}")

    company_configs = parser.generate_embedding_config_dict("company", embedding_dim=32)
    print(f"\nCompany 필드 개수: {len(company_configs)}")
    print("\n처음 3개 예시:")
    for cfg in company_configs[:3]:
        print(f"  {cfg}")
