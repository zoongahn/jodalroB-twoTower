"""
Company 테이블 전처리 모듈

public.company → step1.company_preprocessed
"""

import pandas as pd
from typing import List
from sqlalchemy import text
from database.database_connector import DatabaseConnector
from preprocess.db.numeric import preprocess_numeric_data
from preprocess.db.categorical import preprocess_categorical_data
from preprocess.db.text import preprocess_text_data
import os


class CompanyPreprocessor:
    """Company 테이블 전처리 클래스"""

    def __init__(self, column_types: dict = None):
        """
        Args:
            column_types: 컬럼 타입 정보 딕셔너리
                         {'pk': [...], 'numeric': [...], 'categorical': [...], 'text': [...]}
        """
        self.column_types = column_types or {}
        self.table_name = 'company'

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Company 데이터프레임 전처리

        Args:
            df: 전처리할 데이터프레임

        Returns:
            preprocessed_df: 전처리된 데이터프레임 (PK + 전처리된 피처)
        """
        if df is None or df.empty:
            return df

        print(f"🔄 {self.table_name} 데이터 전처리 시작...")
        print(f"  입력 데이터: {len(df)}행")

        # PK 컬럼
        pk_cols = self.column_types.get('pk', ['bizno'])

        # 1. __DEFAULT__ 행 제거
        original_count = len(df)
        df = df[df['bizno'] != '__DEFAULT__'].copy()
        removed_count = original_count - len(df)

        if removed_count > 0:
            print(f"  🗑️  bizno='__DEFAULT__' 행 제거: {removed_count}행")

        # 결과 데이터프레임 초기화 (PK만)
        result_df = df[pk_cols].copy()

        # 1. 수치형 데이터 전처리
        numeric_cols = [col for col in self.column_types.get('numeric', []) if col in df.columns]
        if numeric_cols:
            print(f"  📊 수치형 컬럼 {len(numeric_cols)}개 전처리 중...")
            numeric_work_df = df[pk_cols + numeric_cols].copy()
            processed_numeric = preprocess_numeric_data(numeric_work_df, table_type=self.table_name)
            processed_numeric_features = processed_numeric.drop(columns=pk_cols)
            result_df = pd.concat([result_df, processed_numeric_features], axis=1)

        # 2. 범주형 데이터 전처리
        categorical_cols = [col for col in self.column_types.get('categorical', []) if col in df.columns]
        if categorical_cols:
            print(f"  🏷️  범주형 컬럼 {len(categorical_cols)}개 전처리 중...")
            categorical_work_df = df[pk_cols + categorical_cols].copy()
            processed_categorical = preprocess_categorical_data(categorical_work_df, table_type=self.table_name)
            processed_categorical_features = processed_categorical.drop(columns=pk_cols)
            result_df = pd.concat([result_df, processed_categorical_features], axis=1)

        # 3. 텍스트 데이터 전처리 (company는 텍스트가 없을 수 있음)
        text_cols = [col for col in self.column_types.get('text', []) if col in df.columns]
        if text_cols:
            print(f"  📝 텍스트 컬럼 {len(text_cols)}개 전처리 중...")
            text_work_df = df[pk_cols + text_cols].copy()
            config_path = f"preprocess/db/config/{self.table_name}_text_config.json"

            if os.path.exists(config_path):
                processed_text = preprocess_text_data(text_work_df, table_name=self.table_name, json_config_path=config_path)
                processed_text_features = processed_text.drop(columns=pk_cols)
                result_df = pd.concat([result_df, processed_text_features], axis=1)
            else:
                print(f"    ⚠️ {config_path} 없음, 텍스트 전처리 건너뜀")

        print(f"✅ {self.table_name} 데이터 전처리 완료! (총 {len(result_df.columns)}개 컬럼)")
        return result_df

    def save_to_db(self, df: pd.DataFrame, target_schema: str, db_connector: DatabaseConnector):
        """
        전처리된 데이터를 DB에 저장

        Args:
            df: 저장할 데이터프레임
            target_schema: 타겟 스키마 (예: 'step1')
            db_connector: DB 연결 객체
        """
        target_table = f"{target_schema}.{self.table_name}_preprocessed"

        print(f"💾 DB 저장 중: {target_table} ({len(df)}행)...")

        df_to_save = df.copy()

        try:
            with db_connector.engine.begin() as conn:
                # 기존 데이터 확인
                result = conn.execute(text(f"SELECT COUNT(*) FROM {target_table}"))
                existing_count = result.scalar()
                print(f"  기존 데이터: {existing_count}행")

                # PK 기반 UPSERT
                pk_cols = self.column_types.get('pk', ['bizno'])

                # INSERT SQL 생성 (UPSERT)
                col_names = ', '.join([f'"{c}"' for c in df_to_save.columns])
                placeholders = ', '.join([f':{c}' for c in df_to_save.columns])

                # UPDATE 절 생성 (PK 제외)
                update_cols = [c for c in df_to_save.columns if c not in pk_cols]
                update_set = ', '.join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])

                conflict_target = ', '.join([f'"{c}"' for c in pk_cols])

                insert_sql = f"""
                INSERT INTO {target_table} ({col_names})
                VALUES ({placeholders})
                ON CONFLICT ({conflict_target})
                DO UPDATE SET {update_set}
                """

                print(f"  🔄 UPSERT 모드: 중복 시 업데이트")

                # 각 행을 개별적으로 삽입/업데이트
                for idx, row in df_to_save.iterrows():
                    row_dict = row.to_dict()
                    conn.execute(text(insert_sql), row_dict)

                print(f"✅ DB 저장 완료: {len(df_to_save)}행 추가/업데이트됨")

        except Exception as e:
            print(f"❌ DB 저장 실패: {e}")
            print(f"   (CSV 파일로는 저장되었습니다)")
            import traceback
            traceback.print_exc()
