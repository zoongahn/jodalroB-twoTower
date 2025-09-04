from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Dict, Tuple
from pathlib import Path


from data.column_classifier import classify_columns

# -----------------------------
# 스키마 정의 (컬럼 명세)
# -----------------------------
# SideSchema를 PK 리스트 기반으로 정의(단일/복합 PK 모두 지원)
@dataclass
class SideSchema:
    table: str
    pk_cols: List[str]                         # ex) ["bidntceno", "bidntceord"] 또는 ["company_id"]
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)          # 원문 텍스트 컬럼(있으면)
    # (선택) 임베딩이 컬럼으로 펼쳐져 있다면 prefix/dims로 별도 지정 가능
    text_embed_prefix: Optional[str] = None
    text_embed_dims: Optional[int] = 768

@dataclass
class PairSchema:
    table: str = "bid_two_tower"
    notice_id_cols: List[str] = field(default_factory=list)    # 페어 테이블의 공고 FK 컬럼들
    company_id_cols: List[str] = field(default_factory=list)   # 페어 테이블의 업체 FK 컬럼들

@dataclass
class TorchRecSchema:
    notice: SideSchema
    company: SideSchema
    pair: PairSchema

def build_side_schema_from_meta(table_name: str, metadata_path: str | Path = "meta/metadata.csv") -> SideSchema:
    """
    meta/metadata.csv에서 table_name 행만 읽어 스키마 구성.
    classify_columns(table) 결과 예:
      { "total": 68, "pk": [...], "numeric": [...], "categorical": [...], "text": [...] }
    """
    md = classify_columns(table_name, metadata_path)
    pk_cols = md.get("pk", []) or []
    numeric = md.get("numeric", []) or []
    categorical = md.get("categorical", []) or []
    text = md.get("text", []) or []

    # PK는 피처에서 제외
    pk_set = set(pk_cols)
    numeric = [c for c in numeric if c not in pk_set]
    categorical = [c for c in categorical if c not in pk_set]
    text = [c for c in text if c not in pk_set]

    return SideSchema(
        table=table_name,
        pk_cols=pk_cols,
        numeric=numeric,
        categorical=categorical,
        text=text,
        # 필요 시 아래 두 필드로 임베딩(예: title_emb_0..767) 연결 가능
        text_embed_prefix=None,
        text_embed_dims=None,
    )

def build_torchrec_schema_from_meta(
    *,
    notice_table: str,
    company_table: str,
    pair_table: str,
    # 페어 테이블의 FK 컬럼명들(복합키면 리스트로 모두 지정)
    pair_notice_id_cols: List[str],     # ex) ["bidntceno", "bidntceord"]
    pair_company_id_cols: List[str],    # ex) ["company_id"]
    metadata_path: str | Path = "meta/metadata.csv",
) -> TorchRecSchema:
    notice = build_side_schema_from_meta(notice_table, metadata_path)
    company = build_side_schema_from_meta(company_table, metadata_path)

    # FK 컬럼이 메타에서 자동 추론되지 않으므로, 외부에서 넘기지 않으면 공고/업체의 PK를 기본으로 사용
    notice_fk = pair_notice_id_cols if pair_notice_id_cols is not None else notice.pk_cols
    company_fk = pair_company_id_cols if pair_company_id_cols is not None else company.pk_cols

    pair = PairSchema(
        table=pair_table,
        notice_id_cols=notice_fk,
        company_id_cols=company_fk,
    )
    return TorchRecSchema(notice=notice, company=company, pair=pair)


    
if __name__ == "__main__":
    from pprint import pprint
    from pathlib import Path

    # 1) 스키마 생성
    schema = build_torchrec_schema_from_meta(
        notice_table="notice",
        company_table="company",
        pair_table="bid_two_tower",
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv",
    )

    # 2) 파일 존재 확인
    assert Path("meta/metadata.csv").exists(), "meta/metadata.csv 가 존재하지 않습니다."

    # 3) 간단 검증
    def _check(side, name):
        # PK가 비었는지
        assert side.pk_cols, f"{name} pk_cols 비어있음"
        # PK가 피처에 섞이지 않았는지
        overlap = set(side.pk_cols) & (set(side.numeric) | set(side.categorical) | set(side.text))
        assert not overlap, f"{name} PK가 피처에 섞여 있음: {overlap}"
        # 중복 제거 확인
        assert len(side.numeric) == len(set(side.numeric)), f"{name} numeric 중복 존재"
        assert len(side.categorical) == len(set(side.categorical)), f"{name} categorical 중복 존재"

    _check(schema.notice, "notice")
    _check(schema.company, "company")
    assert schema.pair.notice_id_cols, "pair.notice_id_cols 비어있음"
    assert schema.pair.company_id_cols, "pair.company_id_cols 비어있음"

    # 4) 요약 출력
    print("\n==== TorchRec Schema (from meta) ====")
    print("[notice]")
    print(" PK      :", schema.notice.pk_cols)
    print(" numeric :", len(schema.notice.numeric))
    print(" categorical :", len(schema.notice.categorical))
    if schema.notice.text:
        print(" text    :", schema.notice.text)
    print(" total   :", len(schema.notice.pk_cols) + len(schema.notice.numeric) + len(schema.notice.categorical) + len(schema.notice.text))

    print("\n[company]")
    print(" PK      :", schema.company.pk_cols)
    print(" numeric :", len(schema.company.numeric))
    print(" categorical :", len(schema.company.categorical))
    if schema.company.text:
        print(" text    :", schema.company.text)
    print(" total   :", len(schema.company.pk_cols) + len(schema.company.numeric) + len(schema.company.categorical) + len(schema.company.text))

    print("\n[pair]")
    print(" table   :", schema.pair.table)
    print(" notice_id_cols :", schema.pair.notice_id_cols)
    print(" company_id_cols:", schema.pair.company_id_cols)
    print(" total   :", len(schema.pair.notice_id_cols) + len(schema.pair.company_id_cols))

    # 5) (선택) 예시 SELECT 미리보기
    try:
        n_keys = ", ".join(schema.notice.pk_cols[:2])  # 너무 길면 일부만
        c_keys = ", ".join(schema.company.pk_cols[:2])
        print("\n[preview] sample join keys:")
        print(f" notice keys : {n_keys}")
        print(f" company keys: {c_keys}")
    except Exception as e:
        print("preview build skipped:", e)

    print("\n✅ schema build OK")