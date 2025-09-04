#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_multiple_notices_preprocessing_v2.py
- 기존 데이터 조회 로직(get_multiple_notices_data) + pipeline_v2(preprocess_and_upload) 결합
- CSV 중간 저장 없이 메모리 내 처리 → 최종 결과만 저장 (+ DB 업로드는 preprocess_and_upload 내부에서 수행)
"""

import os
from typing import List, Tuple, Dict

import pandas as pd

# 사용자의 데이터 로더 / 커넥터
from data.data_loader import get_multiple_notices_data
from data.database_connector import DatabaseConnector

from preprocess.pipeline import preprocess_and_upload
from preprocess.pipeline import load_preprocess_configs

OUTPUT_DIR = "output/preprocessed"


# ===== 메인 ==================================================================
def test_multiple_notices_preprocessing_v2():
    """여러 공고 데이터 조회 → preprocess_and_upload로 전처리+DB업로드 → CSV 저장"""
    # 1) 테스트용 공고 ID 목록 (사용자 코드 그대로)
    notice_ids: List[Tuple[str, str]] = [
        ("20240406546", "000"),
        ("20240406548", "000"),
        ("20240406553", "000"),
        ("20240406556", "000"),
        ("20240406557", "000"),
        ("20140228597", "000"),
        ("20240406558", "000"),
        ("20140232077", "000"),
        ("20240406559", "000"),
        ("20240406563", "000"),
    ]

    # 2) 데이터 조회 (기존 로직 재사용)
    print("🚀 다중 공고 데이터 조회 시작")
    print("=" * 80)
    db = DatabaseConnector()
    try:
        fetched = get_multiple_notices_data(notice_ids, save_to_csv=False)
    finally:
        try:
            db.close()
        except Exception:
            pass

    notice_df = fetched.get("notice", pd.DataFrame())
    company_df = fetched.get("company", pd.DataFrame())

    # 3) preprocess_and_upload로 전처리(+DB 업로드) → 반환 df는 그대로 CSV 저장
    print("\n🚀 전체 데이터 전처리 시작 (preprocess_and_upload)")
    print("=" * 80)

    preprocessed_notice = pd.DataFrame()
    preprocessed_company = pd.DataFrame()

    # --- NOTICE ---
    NUM_CONFIG, CAT_CONFIG, TXT_CONFIG = load_preprocess_configs("notice")
    if not notice_df.empty:
        preprocessed_notice = preprocess_and_upload(
            df=notice_df,
            table_name="notice",
            num_config=NUM_CONFIG,
            cat_config=CAT_CONFIG,
            txt_config=TXT_CONFIG,
        )
        notice_out = os.path.join(OUTPUT_DIR, "final_preprocessed_notice.csv")
        preprocessed_notice.to_csv(notice_out, index=False, encoding="utf-8-sig")
        print(f"✅ 공고 전처리 저장: {notice_out} (rows={len(preprocessed_notice)}, cols={len(preprocessed_notice.columns)})")
    else:
        print("⚠️ 공고 데이터가 비어 있어 전처리를 건너뜁니다.")

    # --- COMPANY ---
    NUM_CONFIG, CAT_CONFIG, TXT_CONFIG = load_preprocess_configs("company")
    if not company_df.empty:
        preprocessed_company = preprocess_and_upload(
            df=company_df,
            table_name="company",
            num_config=NUM_CONFIG,
            cat_config=CAT_CONFIG,
            txt_config=TXT_CONFIG,
        )
        company_out = os.path.join(OUTPUT_DIR, "final_preprocessed_company.csv")
        preprocessed_company.to_csv(company_out, index=False, encoding="utf-8-sig")
        print(f"✅ 업체 전처리 저장: {company_out} (rows={len(preprocessed_company)}, cols={len(preprocessed_company.columns)})")
    else:
        print("⚠️ 업체 데이터가 비어 있어 전처리를 건너뜁니다.")

    return preprocessed_notice, preprocessed_company


if __name__ == "__main__":
    test_multiple_notices_preprocessing_v2()