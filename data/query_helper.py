# query_helper.py
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from data.database_connector import DatabaseConnector
import traceback
from sqlalchemy import inspect, text


class QueryHelper:
    """자주 사용하는 DB 쿼리 함수들"""

    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector

        # .env에서 스키마 및 테이블 설정 로드
        import os
        from dotenv import load_dotenv
        load_dotenv()

        db_schema = os.getenv("DB_SCHEMA", "public")
        notice_table = os.getenv("NOTICE_TABLE", "notice")
        company_table = os.getenv("COMPANY_TABLE", "company")
        bid_table = os.getenv("BID_TWO_TOWER_TABLE", "bid_two_tower")

        # 테이블명 정의 (스키마.테이블 형식)
        self.table_names = {
            'notice': f'{db_schema}.{notice_table}',     # 공고 테이블명
            'bid': f'{db_schema}.{bid_table}',           # 투찰 테이블명
            'company': f'{db_schema}.{company_table}'    # 업체 테이블명
        }

        # 테이블별 PK 정의 (복합키 지원)
        self.table_pk_map = {
            'notice': ['bidntceno', 'bidntceord'],  # 공고 테이블 PK
            'bid': ['bidntceno', 'bidntceord', 'bidprccorpbizrno'],  # 투찰 테이블 PK
            'company': ['bizno']  # 업체 테이블 PK
        }

        self._load_use_columns()

        self._build_select_statements()

    def _load_use_columns(self):
        """메타데이터에서 사용 컬럼 로드"""
        import os
        import pandas as pd
        from dotenv import load_dotenv

        load_dotenv()
        metadata_file = os.getenv("METADATA_FILE_PATH")
        use_keyword = os.getenv("METADATA_USE_KEYWORD")

        if not metadata_file or not use_keyword:
            # 기본값 설정
            self.notice_use_columns = ['*']
            self.company_use_columns = ['*']
            return

        try:
            meta_df = pd.read_csv(metadata_file, dtype=str)

            # 사용 여부가 Y인 데이터만 필터링
            use_df = meta_df[meta_df[use_keyword].str.upper() == 'Y']

            # 테이블별로 컬럼 분류
            notice_columns = use_df[use_df['테이블명'] == 'notice']['컬럼명'].tolist()
            company_columns = use_df[use_df['테이블명'] == 'company']['컬럼명'].tolist()

            self.notice_use_columns = notice_columns if notice_columns else ['*']
            self.company_use_columns = company_columns if company_columns else ['*']

            print(f"✅ notice 사용 컬럼: {len(self.notice_use_columns)}개")
            print(f"✅ company 사용 컬럼: {len(self.company_use_columns)}개")

        except Exception as e:
            print(f"⚠️ 메타데이터 로드 실패: {e}")
            self.notice_use_columns = ['*']
            self.company_use_columns = ['*']

    def _build_select_statements(self):
        """SELECT 문 미리 생성"""
        # notice SELECT 문
        if self.notice_use_columns == ['*']:
            self.notice_select = "*"
        else:
            self.notice_select = ", ".join(self.notice_use_columns)

        # company SELECT 문
        if self.company_use_columns == ['*']:
            self.company_select = "*"
            self.company_select_prefixed = "c.*"  # JOIN용
        else:
            self.company_select = ", ".join(self.company_use_columns)
            self.company_select_prefixed = ", ".join([f"c.{col}" for col in self.company_use_columns])


    def get_table_pk_columns(self, table_name: str) -> List[str]:
        """테이블의 PK 컬럼 목록 반환"""
        return self.table_pk_map.get(table_name, [])

    def get_rows_by_pk(self,
                       table_name: str,
                       pk_values: List[Tuple],
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        PK 값들로 특정 row들 조회

        Args:
            table_name: 테이블명
            pk_values: PK 값들의 리스트 [("20240406546", "000"), ("20240406548", "000"), ...]
            columns: 조회할 컬럼들 (None이면 모든 컬럼)

        Returns:
            조회된 데이터프레임
        """
        if table_name not in self.table_pk_map:
            print(f"❌ 지원하지 않는 테이블: {table_name}")
            return pd.DataFrame()

        pk_columns = self.table_pk_map[table_name]

        if not pk_values:
            print("❌ PK 값이 비어있습니다.")
            return pd.DataFrame()

        # 첫 번째 PK 값으로 컬럼 수 검증
        expected_pk_count = len(pk_columns)
        if len(pk_values[0]) != expected_pk_count:
            print(f"❌ PK 컬럼 수 불일치: 예상 {expected_pk_count}개, 입력 {len(pk_values[0])}개")
            return pd.DataFrame()

        # SELECT 컬럼 결정 - 멤버변수 사용
        if columns is None:
            if table_name == 'notice':
                select_columns = self.notice_select
            elif table_name == 'company':
                select_columns = self.company_select
            else:  # bid는 항상 *
                select_columns = "*"
        else:
            select_columns = ", ".join(columns)

        # WHERE 조건 생성
        where_conditions = []
        for pk_tuple in pk_values:
            pk_conditions = []
            for i, pk_col in enumerate(pk_columns):
                pk_conditions.append(f"{pk_col} = '{pk_tuple[i]}'")
            where_conditions.append(f"({' AND '.join(pk_conditions)})")

        # 최종 쿼리 생성
        where_clause = " OR ".join(where_conditions)
        table_name_actual = self.table_names[table_name]
        query = f"SELECT {select_columns} FROM {table_name_actual} WHERE {where_clause}"

        print(f"🔄 {table_name} 테이블에서 {len(pk_values)}개 row 조회 중...")

        try:
            df = self.db_connector.execute_query(query)
            print(f"✅ 조회 완료: {len(df)}행")
            return df
        except Exception as e:
            print(f"❌ 쿼리 실행 실패: {e}")
            return pd.DataFrame()

    def get_notice_by_ids(self, notice_ids: List[Tuple[str, str]],
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        공고 ID들로 공고 정보 조회

        Args:
            notice_ids: [(bidntceno, bidntceord), ...] 형태의 공고 ID 리스트
            columns: 조회할 컬럼들
        """
        if columns is None:
            columns = self.notice_use_columns

        if columns == ['*']:
            return self.get_rows_by_pk('notice', notice_ids, None)
        else:
            return self.get_rows_by_pk('notice', notice_ids, columns)


    def get_bids_by_notice_ids(self, notice_ids: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        공고 ID들로 해당 공고의 모든 투찰 정보 조회

        Args:
            notice_ids: [(bidntceno, bidntceord), ...] 형태의 공고 ID 리스트
        """
        if not notice_ids:
            return pd.DataFrame()

        # WHERE 조건 생성 (공고 ID만으로 검색) - PK 매핑 사용
        where_conditions = []
        for bidntceno, bidntceord in notice_ids:
            where_conditions.append(f"({self.table_pk_map['bid'][0]} = '{bidntceno}' AND {self.table_pk_map['bid'][1]} = '{bidntceord}')")

        where_clause = " OR ".join(where_conditions)
        query = f"SELECT * FROM {self.table_names['bid']} WHERE {where_clause}"

        print(f"🔄 {len(notice_ids)}개 공고의 투찰 정보 조회 중...")

        try:
            df = self.db_connector.execute_query(query)
            print(f"✅ 투찰 조회 완료: {len(df)}행")
            return df
        except Exception as e:
            print(f"❌ 투찰 조회 실패: {e}")
            return pd.DataFrame()

    def get_companies_by_ids(self, company_ids: List[str],
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        업체 ID들로 업체 정보 조회

        Args:
            company_ids: [bizno1, bizno2, ...] 형태의 업체 ID 리스트
            columns: 조회할 컬럼들
        """
        if columns is None:
            columns = self.company_use_columns

        pk_tuples = [(company_id,) for company_id in company_ids]

        return self.get_rows_by_pk('company', pk_tuples, columns)


    def get_bid_participants(self, notice_id: Tuple[str, str]) -> pd.DataFrame:
        """
        특정 공고의 참여 업체 목록 조회

        Args:
            notice_id: (bidntceno, bidntceord) 형태의 공고 ID
        """
        bidntceno, bidntceord = notice_id

        if self.company_use_columns == ['*']:
            company_select = "c.*"
        else:
            company_cols = [f"c.{col}" for col in self.company_use_columns]
            company_select = ", ".join(company_cols)


        query = f"""
        SELECT {company_select}
        FROM {self.table_names['bid']} b
        LEFT JOIN {self.table_names['company']} c ON b.{self.table_pk_map["bid"][2]} = c.{self.table_pk_map["company"][0]}
        WHERE b.{self.table_pk_map['bid'][0]} = '{bidntceno}' AND b.{self.table_pk_map['bid'][1]} = '{bidntceord}'
        """

        print(f"🔄 공고 {notice_id}의 참여업체 조회 중...")

        try:
            df = self.db_connector.execute_query(query)
            print(f"✅ 참여업체 조회 완료: {len(df)}개 업체")
            return df
        except Exception as e:
            print(f"❌ 참여업체 조회 실패: {e}")
            return pd.DataFrame()

    def get_company_bid_history(self, company_id: str, limit: int = 100) -> pd.DataFrame:
        """
        특정 업체의 투찰 이력 조회

        Args:
            company_id: 업체 ID (bizno)
            limit: 조회할 최대 개수
        """
        query = f"""
        SELECT b.{self.table_pk_map['bid'][0]}, b.{self.table_pk_map['bid'][1]}, n.bidnm, b.bidprc, b.sucsfbsnslttnrt, n.rgstdt
        FROM {self.table_names['bid']} b
        LEFT JOIN {self.table_names['notice']} n ON b.{self.table_pk_map['bid'][0]} = n.{self.table_pk_map['notice'][0]} AND b.{self.table_pk_map['bid'][1]} = n.{self.table_pk_map['notice'][1]}
        WHERE b.{self.table_pk_map['bid'][2]} = '{company_id}'
        ORDER BY n.rgstdt DESC
        LIMIT {limit}
        """

        print(f"🔄 업체 {company_id}의 투찰 이력 조회 중...")

        try:
            df = self.db_connector.execute_query(query)
            print(f"✅ 투찰 이력 조회 완료: {len(df)}건")
            return df
        except Exception as e:
            print(f"❌ 투찰 이력 조회 실패: {e}")
            return pd.DataFrame()

    def get_recent_notices(self, limit: int = 100) -> pd.DataFrame:
        """
        최근 공고 목록 조회 (등록일 기준 최신순)

        Args:
            limit: 최대 조회 개수 (기본 100개)
        """
        query = f"""
        SELECT *
        FROM {self.table_names['notice']}
        ORDER BY rgstdt DESC
        LIMIT {limit}
        """

        print(f"🔄 최근 공고 {limit}개 조회 중...")

        try:
            df = self.db_connector.execute_query(query)
            print(f"✅ 최근 공고 조회 완료: {len(df)}건")
            return df
        except Exception as e:
            print(f"❌ 최근 공고 조회 실패: {e}")
            return pd.DataFrame()

    def count_table_rows(self, table_name: str) -> int:
        """테이블 총 행 수 조회"""
        try:
            table_name_actual = self.table_names[table_name]
            query = f"SELECT COUNT(*) as count FROM {table_name_actual}"
            result = self.db_connector.execute_query(query)
            count = result.iloc[0]['count']
            print(f"📊 {table_name_actual} 테이블: {count:,}행")
            return count
        except Exception as e:
            print(f"❌ {table_name} 행 수 조회 실패: {e}")
            return 0

    def get_all_with_use_columns(
        self,
        table_name: str,
        where_clause: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        try:
            # 1) 실제 테이블명/유즈컬럼 테이블명
            if table_name not in self.table_names:
                print(f"❌ 지원하지 않는 테이블 키: {table_name}")
                return pd.DataFrame()

            # 1) SELECT 절 결정 (이미 빌드된 멤버 사용)
            if table_name == 'notice':
                select_expr = self.notice_select  # 예: "*", "col1, col2, ..."
            elif table_name == 'company':
                select_expr = self.company_select
            else:  # 'bid' 등은 항상 전체 컬럼
                select_expr = "*"

            data_table = self.table_names[table_name]

             # 2) 본문 쿼리 조립
            query = f"SELECT {select_expr} FROM {data_table}"
            if where_clause and where_clause.strip():
                query += f" WHERE {where_clause.strip()}"
            if order_by and order_by.strip():
                query += f" ORDER BY {order_by.strip()}"
            if isinstance(limit, int) and limit > 0:
                query += f" LIMIT {limit}"

            print(f"🔄 {data_table} 테이블 조회 (use_column SELECT 적용) ...")
            df = self.db_connector.execute_query(query)
            print(f"✅ 조회 완료: {len(df)}행")
            return df

        except Exception as e:
            print(f"❌ get_all_with_use_columns 실패: {e}")
            return pd.DataFrame()
        
    # ✔ use columns SELECT 생성 (텍스트 컬럼 제외 가능)
    def get_use_columns_select(self, table_name: str, exclude_text_cols: list[str] | None = None) -> str:
        exclude_text_cols = set(exclude_text_cols or [])
        # 너의 기존 "use 컬럼" 리스트를 가져오는 로직이 있다면 그걸 재사용하세요.
        # 없으면 전체 컬럼에서 제외만 반영:
        import os
        db_schema = os.getenv("DB_SCHEMA", "public")
        insp = inspect(self.db_connector.engine)
        cols = [c["name"] for c in insp.get_columns(table_name, schema=db_schema)]
        use_cols = [c for c in cols if c not in exclude_text_cols]
        col_list = ", ".join(f'"{c}"' for c in use_cols)
        return f'SELECT {col_list} FROM "{db_schema}"."{table_name}"'

    # ✔ COUNT (SELECT와 동일 조건/조인 반영해줘야 정확)
    def get_use_columns_count(self, table_name: str, exclude_text_cols: list[str] | None = None) -> str:
        # 필터/조인이 있다면 동일하게 반영해야 함 (여기선 테이블 전체 기준)
        import os
        db_schema = os.getenv("DB_SCHEMA", "public")
        return f'SELECT COUNT(*) FROM "{db_schema}"."{table_name}"'

    # ✔ PK+TEXT 최소 SELECT
    def select_pk_and_text(self, table_name: str, pk_cols: list[str], text_col: str) -> str:
        import os
        db_schema = os.getenv("DB_SCHEMA", "public")
        cols = ", ".join(f'"{c}"' for c in [*pk_cols, text_col])
        return f'SELECT {cols} FROM "{db_schema}"."{table_name}"'
    
    
    
# 사용 예시
def test_query_helper():
    """QueryHelper 테스트"""
    try:
        from data.database_connector import DatabaseConnector

        # 데이터베이스 연결
        db_connector = DatabaseConnector()
        helper = QueryHelper(db_connector)

        print("🚀 QueryHelper 테스트 시작")
        print("=" * 50)

        # 1. 테이블 행 수 확인
        helper.count_table_rows('notice')
        # helper.count_table_rows('bid')
        helper.count_table_rows('company')

        # 2. 샘플 데이터 조회
        notice_sample = helper.get_table_sample('notice', 5)
        print(f"\n📋 공고 샘플:\n{notice_sample}")

        # 3. 특정 공고 조회 (예시) - PK 매핑 사용
        if not notice_sample.empty:
            # 첫 번째 공고의 PK 값 추출
            first_notice = notice_sample.iloc[0]
            notice_id = (first_notice[helper.table_pk_map['notice'][0]], first_notice[helper.table_pk_map['notice'][1]])

            print(f"\n🔍 공고 {notice_id} 상세 조회:")
            notice_detail = helper.get_notice_by_ids([notice_id])
            print(notice_detail)

            # 해당 공고의 참여업체 조회
            participants = helper.get_bid_participants(notice_id)
            print(f"\n👥 참여업체 목록:\n{participants}")

        # 4. 최근 공고 조회
        limit = 100
        recent_notices = helper.get_recent_notices(limit)
        print(f"\n📅 최근 {limit}개 공고:\n{recent_notices}")

        return helper

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        traceback.print_exc()
        return None

    finally:
        try:
            db_connector.close()
        except:
            pass


if __name__ == "__main__":
    test_query_helper()