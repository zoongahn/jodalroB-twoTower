# projectRoot/data/classify_columns.py
from __future__ import annotations
import re
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

# 메타데이터 CSV의 기본 경로 (프로젝트 루트 기준)
DEFAULT_METADATA_PATH = Path("meta/metadata.csv")

# 컬럼 헤더 이름(한글 헤더 가정). 필요시 아래 매핑에 다른 별칭을 추가하세요.
COL_MAP = {
    "table": ["테이블명", "table", "TABLE"],
    "column": ["컬럼명", "컬럼", "column", "COLUMN", "필드명"],
    "use": ["사용 여부", "사용여부", "use", "USE"],
    "pk": ["PK", "pk", "Pk"],
    "dtype": ["타입", "데이터타입", "type", "TYPE", "data_type"],
    "is_categorical": ["범주형 여부", "범주형여부", "categorical", "IS_CATEGORICAL"],
}

NUMERIC_TYPES = {
    "bigint",
    "double precision",
    "numeric",
    "integer",
}

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # 소문자/공백 제거 등 느슨한 매칭
    norm = {re.sub(r"\s+", "", k).lower(): k for k in df.columns}
    for c in candidates:
        key = re.sub(r"\s+", "", c).lower()
        if key in norm:
            return norm[key]
    raise KeyError(f"메타데이터 CSV에 컬럼이 없습니다: {candidates}")

def _normalize_yes(value) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"y", "yes", "true", "1", "t"}

def _is_char_len_1(dtype_str: str) -> bool:
    """
    'character(1)', 'char(1)', 'character varying(1)' 등 길이 1 텍스트인지 확인
    """
    s = dtype_str.strip().lower()
    # exact 'character(1)' or 'char(1)'
    if re.fullmatch(r"(character|char)\s*\(\s*1\s*\)", s):
        return True
    # character varying(1)도 허용
    if re.fullmatch(r"(character varying|varchar)\s*\(\s*1\s*\)", s):
        return True
    return False

def _is_text(dtype_str: str) -> bool:
    s = dtype_str.strip().lower()
    return s == "text" or s.startswith("text") or s == "varchar" or s.startswith("character varying")

def _is_numeric(dtype_str: str) -> bool:
    s = dtype_str.strip().lower()
    return s in NUMERIC_TYPES

def classify_columns(
    table_name: str,
    metadata_path: Optional[str | Path] = None,
) -> Dict[str, List[str]]:
    """
    /meta/metadata.csv를 읽어 테이블별 피처 분류:
    1) 테이블명 행만 선택
    2) '사용 여부'가 Y인 컬럼만 남김
    3) PK=Y는 분류에서 제외(내부 저장만)
    4) 타입이 bigint/double precision/numeric/integer → numeric
    5) 타입이 text 또는 character(1) → 범주형 여부가 Y면 categorical, N이면 텍스트(이번 출력에선 무시)

    Returns:
        {"numeric": [...], "categorical": [...], "text": [...]}
    """
    mpath = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    # UTF-8 BOM 대응
    df = pd.read_csv(mpath, encoding="utf-8-sig")

    # 헤더 결정
    col_table = _find_col(df, COL_MAP["table"])
    col_column = _find_col(df, COL_MAP["column"])
    col_use = _find_col(df, COL_MAP["use"])
    col_pk = _find_col(df, COL_MAP["pk"])
    col_dtype = _find_col(df, COL_MAP["dtype"])
    col_is_cat = _find_col(df, COL_MAP["is_categorical"])

    # 1) 테이블 필터
    df_tbl = df[df[col_table].astype(str).str.strip() == str(table_name).strip()].copy()

    # 2) 사용 여부=Y
    use_mask = df_tbl[col_use].apply(_normalize_yes)
    df_tbl = df_tbl[use_mask].copy()

    # 3) PK 수집 및 제외
    pk_mask = df_tbl[col_pk].apply(_normalize_yes)
    pk_cols = df_tbl.loc[pk_mask, col_column].astype(str).str.strip().tolist()
    df_feat = df_tbl.loc[~pk_mask].copy()

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    text_cols: List[str] = []

    for _, row in df_feat.iterrows():
        colname = str(row[col_column]).strip()
        dtype = str(row[col_dtype]).strip().lower()

        if _is_numeric(dtype):
            numeric_cols.append(colname)
            continue

        # text/character(1) 처리
        if _is_text(dtype) or _is_char_len_1(dtype):
            is_cat = _normalize_yes(row[col_is_cat])
            if is_cat:
                categorical_cols.append(colname)
            else:
                text_cols.append(colname)
            continue

        # 기타 타입은 이번 분류에서 제외 (필요시 여기에 로직 확장)
        # 예: date/timestamp/boolean 등

    return {"total": len(df_tbl), "pk": pk_cols, "numeric": numeric_cols, "categorical": categorical_cols, "text": text_cols}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="메타데이터에서 조사할 테이블명")
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH), help="메타데이터 CSV 경로 (기본: meta/metadata.csv)")
    args = ap.parse_args()

    result = classify_columns(args.table, args.metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))