# ml_data_loader.py
from typing import Dict, List, Optional
import pandas as pd

from data.database_connector import DatabaseConnector
import os



class DataLoader:
    """ML 사용가능 컬럼만을 이용한 테이블별 데이터 로더"""

    def __init__(self, db_connector: DatabaseConnector, metadata_csv_path: str):
        """
        Args:
            db_connector: 데이터베이스 연결 객체
            metadata_csv_path: metadata_ML.csv 파일 경로
        """
        self.use_keyword = os.getenv("METADATA_USE_KEYWORD")

        self.db_connector = db_connector
        self.metadata_csv_path = metadata_csv_path
        self.table_columns_map = {}
        self._load_metadata()


    def _load_metadata(self):
        """메타데이터 파일 로드 및 테이블별 ML 컬럼 매핑 구축"""
        try:
            # 메타데이터 파일 로드
            meta_df = pd.read_csv(self.metadata_csv_path, dtype=str)
            print(f"📄 메타데이터 파일 로드: {len(meta_df)}개 행")

            # 사용 여부가 'Y'인 컬럼들만 필터링
            if self.use_keyword in meta_df.columns:
                use_df = meta_df[meta_df[self.use_keyword].str.upper() == 'Y'].copy()
                print(f"📊 \"{self.use_keyword}\" 컬럼: {len(use_df)}개")
            else:
                print(f"⚠️ \"{self.use_keyword}\" 컬럼이 없습니다. 모든 컬럼을 사용합니다.")
                use_df = meta_df.copy()

            # 테이블명과 컬럼명 컬럼 찾기
            table_col = self._find_column(use_df, ['테이블명', 'table_name', '테이블'])
            column_col = self._find_column(use_df, ['컬럼명', 'column_name', '컬럼'])

            if not table_col or not column_col:
                raise ValueError("테이블명 또는 컬럼명 컬럼을 찾을 수 없습니다.")

            # 테이블별 컬럼 그룹화
            for _, row in use_df.iterrows():
                table_name = row[table_col]
                column_name = row[column_col]

                if pd.isna(table_name) or pd.isna(column_name):
                    continue

                if table_name not in self.table_columns_map:
                    self.table_columns_map[table_name] = []

                self.table_columns_map[table_name].append(column_name)

            print("✅ 테이블별 \"{self.use_keyword}\" 매핑 완료!")
            for table_name, columns in self.table_columns_map.items():
                print(f"  📋 {table_name}: {len(columns)}개 컬럼")

        except Exception as e:
            print(f"❌ 메타데이터 로드 실패: {e}")
            raise

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """가능한 컬럼명 후보에서 실제 존재하는 컬럼 찾기"""
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    def get_available_tables(self) -> List[str]:
        """사용 가능한 테이블 목록 반환"""
        return list(self.table_columns_map.keys())

    def get_table_ml_columns(self, table_name: str) -> List[str]:
        """특정 테이블의 ML 사용가능 컬럼 목록 반환"""
        return self.table_columns_map.get(table_name, [])

    def load_table_data(self,
                        table_name: str,
                        limit: Optional[int] = None) -> pd.DataFrame:
        """
        특정 테이블의 ML 컬럼 데이터만 로드

        Args:
            table_name: 로드할 테이블명
            limit: 로드할 행 수 제한
        """
        if table_name not in self.table_columns_map:
            print(f"❌ 테이블 '{table_name}'에 대한 ML 컬럼 정보가 없습니다.")
            return pd.DataFrame()

        ml_columns = self.table_columns_map[table_name]

        if not ml_columns:
            print(f"❌ 테이블 '{table_name}'에 \"{self.use_keyword}\" 컬럼이 없습니다.")
            return pd.DataFrame()

        # SQL 쿼리 구성
        columns_str = ", ".join(ml_columns)
        query = f"SELECT {columns_str} FROM {table_name}"

        if limit:
            query += f" LIMIT {limit}"

        print(f"🔄 테이블 '{table_name}' ML 데이터 로딩 중...")
        print(f"📄 로드할 컬럼 수: {len(ml_columns)}개")

        try:
            df = self.db_connector.execute_query(query)
            print(f"✅ 로딩 완료! {len(df):,}행 × {len(df.columns)}열")
            return df

        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            return pd.DataFrame()

    def load_all_tables_data(self,
                             limit_per_table: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        모든 테이블의 ML 데이터 로드

        Args:
            limit_per_table: 테이블당 로드할 행 수 제한
        """
        print("🚀 모든 테이블 ML 데이터 로딩 시작")
        print("=" * 50)

        results = {}

        for table_name in self.table_columns_map.keys():
            print(f"\n📋 처리 중: {table_name}")
            df = self.load_table_data(table_name, limit=limit_per_table)

            if not df.empty:
                results[table_name] = df
            else:
                print(f"⚠️ '{table_name}' 테이블 데이터 로딩 실패")

        print(f"\n✅ 전체 로딩 완료! 성공한 테이블: {len(results)}개")
        return results


# 사용 예시
def main():
    """사용 예시"""
    try:
        # 1. 데이터베이스 연결
        db_connector = DatabaseConnector()

        # 2. ML 데이터 로더 초기화
        loader = DataLoader(
            db_connector=db_connector,
            metadata_csv_path=os.getenv("METADATA_FILE_PATH")
        )

        # 3. 사용 가능한 테이블 확인
        tables = loader.get_available_tables()
        print(f"\n📋 사용 가능한 테이블: {tables}")

        # 4. 각 테이블별 데이터 로드 (예시)
        for table_name in tables:
            # 각 테이블에서 1000행씩 로드
            df = loader.load_table_data(table_name, limit=1000)
            if not df.empty:
                print(f"✅ {table_name}: {len(df)}행 로드 완료")

        # 5. 모든 테이블 데이터 한번에 로드
        all_data = loader.load_all_tables_data(limit_per_table=500)

        return all_data

    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        return None

    finally:
        try:
            db_connector.close()
        except:
            pass


if __name__ == "__main__":
    main()