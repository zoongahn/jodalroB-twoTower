# data_loader.py
from typing import List, Tuple, Dict, Optional
import pandas as pd
from .database_connector import DatabaseConnector
from .query_helper import QueryHelper
import os

def get_multiple_notices_data(
    notice_ids: List[Tuple[str, str]],
    *,
    save_to_csv: bool = False,
    output_dir: str = "output/multiple"
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    여러 공고에 대한 notice, bid, company 데이터 조회 및 통합.
    Returns: {'notice': df, 'bid': df, 'company': df}
    """
    db = DatabaseConnector()
    helper = QueryHelper(db)
    try:
        all_notice, all_bid, all_company_ids = [], [], set()

        for i, nid in enumerate(notice_ids, 1):
            notice_df = helper.get_notice_by_ids([nid])
            bid_df = helper.get_bids_by_notice_ids([nid])

            if not notice_df.empty:
                all_notice.append(notice_df)
            if not bid_df.empty:
                all_bid.append(bid_df)
                if 'bidprccorpbizrno' in bid_df.columns:
                    all_company_ids.update(bid_df['bidprccorpbizrno'].dropna().unique().tolist())

        notice = pd.concat(all_notice, ignore_index=True) if all_notice else pd.DataFrame()
        bid = pd.concat(all_bid, ignore_index=True) if all_bid else pd.DataFrame()
        company = helper.get_companies_by_ids(list(all_company_ids)) if all_company_ids else pd.DataFrame()

        if save_to_csv:
            os.makedirs(output_dir, exist_ok=True)
            if not notice.empty:
                notice.to_csv(os.path.join(output_dir, "multiple_notices.csv"), index=False, encoding="utf-8-sig")
            if not bid.empty:
                bid.to_csv(os.path.join(output_dir, "multiple_bids.csv"), index=False, encoding="utf-8-sig")
            if not company.empty:
                company.to_csv(os.path.join(output_dir, "multiple_companies.csv"), index=False, encoding="utf-8-sig")

        return {"notice": notice, "bid": bid, "company": company}
    finally:
        db.close()