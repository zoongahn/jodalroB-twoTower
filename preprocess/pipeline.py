# pipeline.py
from typing import Tuple, List

from data.database_connector import DatabaseConnector
from data.query_helper import QueryHelper
import pandas as pd

from preprocess.numeric_preprocess import preprocess_numeric_data
from preprocess.categorical_preprocess import preprocess_categorical_data
# from text_preprocess import preprocess_text_data
import os
import json

# 전처리기에서 사용하는 PK맵을 파이프라인에서도 정의
TABLE_PK_MAP = {
    'notice': ['bidntceno', 'bidntceord'],
    'company': ['bizno']
}

class DataPreprocessor:
    """데이터 전처리 파이프라인 클래스"""

    def __init__(self):
        self.numeric_preprocessor = preprocess_numeric_data
        self.categorical_preprocessor = preprocess_categorical_data
        # self.text_preprocessor = preprocess_text_data

        self._load_metadata()

    def _load_metadata(self):
        """메타데이터 로드 및 컬럼 분류"""
        import os
        import pandas as pd
        from dotenv import load_dotenv

        load_dotenv()
        metadata_file = os.getenv("METADATA_FILE_PATH")
        use_keyword = os.getenv("METADATA_USE_KEYWORD")

        try:
            meta_df = pd.read_csv(metadata_file, dtype=str)
            # 사용 여부가 Y인 데이터만 필터링
            use_df = meta_df[meta_df[use_keyword].str.upper() == 'Y']

            # 테이블별 PK 정의 (query_helper와 동일)
            table_pk_map = {
                'notice': ['bidntceno', 'bidntceord'],
                'company': ['bizno']
            }

            # 테이블별 컬럼 분류
            self.column_types = {}

            for table_name in ['notice', 'company']:
                table_meta = use_df[use_df['테이블명'] == table_name]

                self.column_types[table_name] = {
                    'pk': table_pk_map.get(table_name, []),  # PK 컬럼 추가
                    'numeric': [],
                    'categorical': [],
                    'text': []
                }

                # PK 컬럼 목록
                pk_columns = set(table_pk_map.get(table_name, []))

                for _, row in table_meta.iterrows():
                    col_name = row['컬럼명']

                    # PK 컬럼은 건너뛰기
                    if col_name in pk_columns:
                        continue

                    data_type = row['타입'].lower()
                    categorical_flag = row.get('범주형 여부')

                    # 수치형 분류
                    if any(numeric_type in data_type for numeric_type in
                           ['integer', 'bigint', 'numeric', 'double precision', 'int', 'float']):
                        self.column_types[table_name]['numeric'].append(col_name)

                    # 텍스트/범주형 분류
                    elif any(text_type in data_type for text_type in
                             ['text', 'character', 'varchar', 'char']):
                        if categorical_flag == 'Y':
                            self.column_types[table_name]['categorical'].append(col_name)
                        else:
                            self.column_types[table_name]['text'].append(col_name)

            print("✅ 메타데이터 기반 컬럼 분류 완료:")
            for table in self.column_types:
                print(f"  📋 {table}: PK {len(self.column_types[table]['pk'])}개, "
                      f"수치형 {len(self.column_types[table]['numeric'])}개, "
                      f"범주형 {len(self.column_types[table]['categorical'])}개, "
                      f"텍스트 {len(self.column_types[table]['text'])}개")

        except Exception as e:
            print(f"⚠️ 메타데이터 로드 실패: {e}")
            # 기본값 설정
            self.column_types = {
                'notice': {'pk': ['bidntceno', 'bidntceord'], 'numeric': [], 'categorical': [], 'text': []},
                'company': {'pk': ['bizno'], 'numeric': [], 'categorical': [], 'text': []}
            }


    def preprocess_dataframe(self, df, table_type):
        """
        데이터프레임 전처리 (notice, company 공통)

        Args:
            df: 전처리할 데이터프레임
            table_type: 'notice' 또는 'company'

        Returns:
            preprocessed_df: 전처리된 데이터프레임
        """
        if df is None or df.empty:
            return df

        print(f"🔄 {table_type} 데이터 전처리 시작...")

        # 해당 테이블의 컬럼 타입 정보 가져오기
        table_columns = self.column_types.get(table_type, {})

        # 1. 수치형 데이터 전처리
        numeric_cols = [col for col in table_columns.get('numeric', []) if col in df.columns]
        if numeric_cols:
            pk_cols = self.column_types.get(table_type, {}).get('pk', [])
            numeric_work_df = df[pk_cols + numeric_cols].copy()
            processed_numeric = self.numeric_preprocessor(numeric_work_df, table_type=table_type)
            df = df.drop(columns=numeric_cols).merge(processed_numeric, on=pk_cols, how='left')


        # 2. 범주형 데이터 전처리
        categorical_cols = [col for col in table_columns.get('categorical', []) if col in df.columns]
        if categorical_cols:
            pk_cols = self.column_types.get(table_type, {}).get('pk', [])
            categorical_work_df = df[pk_cols + categorical_cols].copy()
            processed_categorical = self.categorical_preprocessor(categorical_work_df, table_type=table_type)
            df = df.drop(columns=categorical_cols).merge(processed_categorical, on=pk_cols, how='left')


        # 3. 텍스트 데이터 전처리
        text_cols = [col for col in table_columns.get('text', []) if col in df.columns]
        if text_cols:
            text_df = df[text_cols].copy()
            processed_text = self.text_preprocessor(text_df, table_type=table_type)
            df[text_cols] = processed_text[text_cols]

        print(f"✅ {table_type} 데이터 전처리 완료!")
        return df



def get_notice_full_data(notice_id_tuple: List[Tuple[str, str]]) -> dict:
    """특정 공고에 대한 notice, bid, company 데이터 조회"""
    db_connector = DatabaseConnector()
    helper = QueryHelper(db_connector)

    try:
        notice_data = helper.get_notice_by_ids([notice_id_tuple])
        bid_data = helper.get_bids_by_notice_ids([notice_id_tuple])

        company_data = None
        if not bid_data.empty:
            company_ids = bid_data['bidprccorpbizrno'].unique().tolist()
            company_data = helper.get_companies_by_ids(company_ids)

        return {
            'notice': notice_data,
            'bid': bid_data,
            'company': company_data
        }
    finally:
        db_connector.close()


def preprocess_pipeline(notice_id_tuple):
    """
    전체 데이터 전처리 파이프라인

    Args:
        notice_id_tuple: ('20240406546', '000') 형태의 공고 ID

    Returns:
        dict: {'notice': preprocessed_df, 'bid': df, 'company': preprocessed_df}
    """
    print(f"🚀 데이터 전처리 파이프라인 시작: {notice_id_tuple}")

    # 1. 원본 데이터 조회
    raw_data = get_notice_full_data(notice_id_tuple)

    # 2. 전처리기 생성
    preprocessor = DataPreprocessor()

    # 3. notice, company 데이터 전처리
    preprocessed_notice = preprocessor.preprocess_dataframe(raw_data['notice'], 'notice')
    preprocessed_company = preprocessor.preprocess_dataframe(raw_data['company'], 'company')


    print("✅ 전체 전처리 파이프라인 완료!")

    return {
        'notice': preprocessed_notice,
        'bid': raw_data['bid'],
        'company': preprocessed_company
    }


def get_multiple_notices_data(notice_ids: List[Tuple[str, str]]) -> dict:
    """
    여러 공고에 대한 notice, bid, company 데이터 조회 및 통합

    Args:
        notice_ids: [(bidntceno, bidntceord), ...] 형태의 공고 ID 리스트

    Returns:
        dict: {'notice': combined_notice_df, 'bid': combined_bid_df, 'company': combined_company_df}
    """
    db_connector = DatabaseConnector()
    helper = QueryHelper(db_connector)

    import os

    try:
        all_notice_data = []
        all_bid_data = []
        all_company_ids = set()

        print(f"🔄 {len(notice_ids)}개 공고 데이터 조회 중...")

        for i, notice_id in enumerate(notice_ids, 1):
            print(f"  📋 {i}/{len(notice_ids)}: {notice_id}")

            # 각 공고별 데이터 조회
            notice_data = helper.get_notice_by_ids([notice_id])
            bid_data = helper.get_bids_by_notice_ids([notice_id])

            if not notice_data.empty:
                all_notice_data.append(notice_data)

            if not bid_data.empty:
                all_bid_data.append(bid_data)
                # 업체 ID 수집
                company_ids = bid_data['bidprccorpbizrno'].unique()
                all_company_ids.update(company_ids)

        # 데이터 통합
        combined_notice = pd.concat(all_notice_data, ignore_index=True) if all_notice_data else pd.DataFrame()
        combined_bid = pd.concat(all_bid_data, ignore_index=True) if all_bid_data else pd.DataFrame()

        # 업체 데이터 조회 (중복 제거된 업체 ID로)
        combined_company = None
        if all_company_ids:
            company_ids_list = list(all_company_ids)
            combined_company = helper.get_companies_by_ids(company_ids_list)

        print(f"✅ 데이터 통합 완료:")
        print(f"  📊 공고: {len(combined_notice)}행")
        print(f"  💰 투찰: {len(combined_bid)}행")
        print(f"  🏢 업체: {len(combined_company) if combined_company is not None else 0}행")

        # CSV 파일로 저장 - 절대 경로 사용
        output_dir = os.path.abspath("output/multiple")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 출력 디렉터리 생성: {output_dir}")

        if not combined_notice.empty:
            notice_path = os.path.join(output_dir, "multiple_notices.csv")
            combined_notice.to_csv(notice_path, index=False, encoding='utf-8-sig')
            print(f"📁 공고 데이터 저장: {notice_path}")

        if not combined_bid.empty:
            bid_path = os.path.join(output_dir, "multiple_bids.csv")
            combined_bid.to_csv(bid_path, index=False, encoding='utf-8-sig')
            print(f"📁 투찰 데이터 저장: {bid_path}")

        if combined_company is not None and not combined_company.empty:
            company_path = os.path.join(output_dir, "multiple_companies.csv")
            combined_company.to_csv(company_path, index=False, encoding='utf-8-sig')
            print(f"📁 업체 데이터 저장: {company_path}")

        return {
            'notice': combined_notice,
            'bid': combined_bid,
            'company': combined_company
        }

    finally:
        db_connector.close()


def test_multiple_notices_preprocessing():
    """10개 공고 데이터로 전처리 테스트"""

    # 테스트용 공고 ID 목록
    notice_ids = [
        ("20240406546", "000"),
        ("20240406548", "000"),
        ("20240406553", "000"),
        ("20240406556", "000"),
        ("20240406557", "000"),
        ("20140228597", "000"),
        ("20240406558", "000"),
        ("20140232077", "000"),
        ("20240406559", "000"),
        ("20240406563", "000")
    ]

    # 1. 여러 공고 데이터 조회
    print("🚀 다중 공고 데이터 조회 시작")
    print("=" * 80)
    data = get_multiple_notices_data(notice_ids)
    
    # 2. 전처리기 생성 및 실행
    print("\n🚀 전체 데이터 전처리 시작")
    print("=" * 80)
    preprocessor = DataPreprocessor()

    preprocessed_notice = preprocessor.preprocess_dataframe(data['notice'].copy(), 'notice')
    preprocessed_company = preprocessor.preprocess_dataframe(data['company'].copy(), 'company')
    
    # 3. 결과 저장
    output_dir = "output/preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    
    notice_path = os.path.join(output_dir, "final_preprocessed_notice.csv")
    company_path = os.path.join(output_dir, "final_preprocessed_company.csv")
    
    preprocessed_notice.to_csv(notice_path, index=False, encoding='utf-8-sig')
    preprocessed_company.to_csv(company_path, index=False, encoding='utf-8-sig')
    
    print("\n✅ 최종 전처리 결과 저장 완료:")
    print(f"  - 공고: {notice_path}")
    print(f"  - 업체: {company_path}")

    print("\n📋 최종 공고 데이터 샘플:")
    print(preprocessed_notice.head())


if __name__ == "__main__":
    # 다중 공고 데이터로 테스트
    test_multiple_notices_preprocessing()