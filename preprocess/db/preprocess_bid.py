"""
Bid 테이블 전처리 모듈

public.bid → step1.bid_two_tower

주요 기능:
1. bidprccorpbizrno = '__DEFAULT__' 행 제거
2. 공고-업체 페어 데이터 생성 (positive pairs only)
3. 중복 제거

참고:
- step1.bid_two_tower는 positive pairs만 저장 (실제 입찰 기록)
- label 컬럼 없음 (모든 페어는 암묵적으로 positive)
- Negative sampling은 학습 시 동적으로 생성
"""

import pandas as pd
from typing import List, Tuple
from sqlalchemy import text
from database.database_connector import DatabaseConnector


class BidPreprocessor:
    """Bid 테이블 전처리 클래스 (Two-Tower 모델용 Positive Pair 생성)"""

    def __init__(self):
        """
        Bid 테이블은 공고-업체 positive pair를 생성하는 용도
        (실제로 입찰한 조합만 저장)
        """
        self.table_name = 'bid'

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bid 데이터프레임 전처리 → bid_two_tower 페어 테이블 생성

        Args:
            df: 원본 bid 데이터프레임

        Returns:
            pair_df: 페어 테이블 (bidntceno, bidntceord, bizno)
        """
        if df is None or df.empty:
            print(f"⚠️ {self.table_name} 데이터가 비어있습니다.")
            return pd.DataFrame()

        print(f"🔄 {self.table_name} 데이터 전처리 시작 (Positive Pairs 생성)...")
        print(f"  입력 데이터: {len(df)}행")

        # 1. __DEFAULT__ 행 제거
        original_count = len(df)
        df_filtered = df[df['bidprccorpbizrno'] != '__DEFAULT__'].copy()
        removed_count = original_count - len(df_filtered)

        if removed_count > 0:
            print(f"  🗑️  bidprccorpbizrno='__DEFAULT__' 행 제거: {removed_count}행")

        # 2. 필요한 컬럼만 선택 (공고 ID + 업체 ID)
        required_cols = ['bidntceno', 'bidntceord', 'bidprccorpbizrno']
        missing_cols = [col for col in required_cols if col not in df_filtered.columns]

        if missing_cols:
            print(f"  ❌ 필수 컬럼 누락: {missing_cols}")
            return pd.DataFrame()

        # 3. 페어 테이블 생성
        pair_df = df_filtered[required_cols].copy()

        # 컬럼명 변경 (bizno로 통일)
        pair_df = pair_df.rename(columns={
            'bidprccorpbizrno': 'bizno'
        })

        # 4. 중복 제거 (같은 공고-업체 페어가 여러 번 있을 수 있음)
        before_dedup = len(pair_df)
        pair_df = pair_df.drop_duplicates(subset=['bidntceno', 'bidntceord', 'bizno'], keep='first')
        after_dedup = len(pair_df)

        if before_dedup != after_dedup:
            print(f"  🗑️  중복 페어 제거: {before_dedup - after_dedup}행")

        print(f"✅ {self.table_name} 데이터 전처리 완료!")
        print(f"  최종 Positive Pair 수: {len(pair_df):,}행")
        print(f"  (모든 페어는 실제 입찰 기록 = Positive samples)")

        return pair_df

    def save_to_db(self, df: pd.DataFrame, target_schema: str, db_connector: DatabaseConnector):
        """
        전처리된 페어 데이터를 DB에 저장

        Args:
            df: 저장할 페어 데이터프레임 (bidntceno, bidntceord, bizno)
            target_schema: 타겟 스키마 (예: 'step1')
            db_connector: DB 연결 객체
        """
        target_table = f"{target_schema}.bid_two_tower"

        print(f"💾 DB 저장 중: {target_table} ({len(df):,}행)...")

        df_to_save = df.copy()

        try:
            with db_connector.engine.begin() as conn:
                # 기존 데이터 확인
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {target_table}"))
                    existing_count = result.scalar()
                    print(f"  기존 데이터: {existing_count:,}행")
                except Exception:
                    print(f"  ⚠️ 테이블이 존재하지 않을 수 있습니다. 새로 생성됩니다.")

                # 컬럼 확인 (bidntceno, bidntceord, bizno만)
                expected_cols = ['bidntceno', 'bidntceord', 'bizno']
                actual_cols = list(df_to_save.columns)

                if set(actual_cols) != set(expected_cols):
                    print(f"  ⚠️ 예상 컬럼: {expected_cols}, 실제 컬럼: {actual_cols}")
                    df_to_save = df_to_save[expected_cols]

                # INSERT SQL 생성 (IGNORE 모드 - 중복 무시)
                col_names = ', '.join([f'"{c}"' for c in df_to_save.columns])
                placeholders = ', '.join([f':{c}' for c in df_to_save.columns])
                conflict_target = ', '.join([f'"{c}"' for c in df_to_save.columns])

                insert_sql = f"""
                INSERT INTO {target_table} ({col_names})
                VALUES ({placeholders})
                ON CONFLICT ({conflict_target})
                DO NOTHING
                """

                print(f"  🔄 INSERT 모드: 중복 시 무시 (IGNORE)")

                # 배치 삽입 (더 빠름)
                batch_size = 1000
                inserted = 0

                for i in range(0, len(df_to_save), batch_size):
                    batch = df_to_save.iloc[i:i+batch_size]
                    for _, row in batch.iterrows():
                        row_dict = row.to_dict()
                        try:
                            conn.execute(text(insert_sql), row_dict)
                            inserted += 1
                        except Exception:
                            pass  # 중복이면 무시

                    if (i + batch_size) % 10000 == 0:
                        print(f"    진행: {i + batch_size:,}/{len(df_to_save):,} 행 처리 중...")

                print(f"✅ DB 저장 완료: {inserted:,}행 추가됨 (중복 제외)")

        except Exception as e:
            print(f"❌ DB 저장 실패: {e}")
            print(f"   (CSV 파일로는 저장되었습니다)")
            import traceback
            traceback.print_exc()
