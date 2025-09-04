# database_connector.py
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, Optional
import os
from dotenv import load_dotenv
import contextlib
import io


class DatabaseConnector:
    """PostgreSQL 데이터베이스 연결 관리"""

    def __init__(self, db_config: Optional[Dict] = None):
        if db_config is None:
            load_dotenv()
            self.db_config = {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", 5432)),
                "user": os.getenv("POSTGRES_USER"),
                "password": os.getenv("POSTGRES_PASSWORD"),
                "database": os.getenv("POSTGRES_DB"),
            }
        else:
            self.db_config = db_config

        self.engine = None
        self._connect()


    def _connect(self):
        try:
            conn_str = (
                f"postgresql+psycopg://{self.db_config['user']}:"
                f"{self.db_config['password']}@"
                f"{self.db_config['host']}:{self.db_config['port']}/"
                f"{self.db_config['database']}?connect_timeout=10"
            )
            self.engine = create_engine(
                conn_str,
                pool_pre_ping=True,
                pool_recycle=1800,
                future=True,
            )
            with self.engine.connect() as cx:
                cx.execute(text("SELECT 1"))
            print("✅ PostgreSQL 연결 성공!")
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """SQL 쿼리 실행"""
        return pd.read_sql(query, self.engine)

    def close(self):
        """연결 종료"""
        if self.engine:
            self.engine.dispose()
            
    def count_rows(conn, sql: str) -> int:
        """주어진 SELECT COUNT(*) 쿼리 실행 후 row 수 반환"""
        with conn.connect() as cur:
            result = cur.execute(sql).scalar()
        return result
            
    @contextlib.contextmanager
    def raw_connection(self):
        conn = self.engine.raw_connection()  # psycopg2 connection
        try:
            yield conn
        finally:
            conn.close()
            
        # ✔ 진행률용 카운트
    def count_rows(self, count_sql: str) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(count_sql).scalar())

    # ✔ 청크 제너레이터 (SQLAlchemy 엔진으로 pandas chunksize)
    def iter_sql_chunks(self, select_sql: str, chunksize: int = 50_000):
        return pd.read_sql_query(select_sql, self.engine, chunksize=chunksize)

    # ✔ pgvector 확장/컬럼 보장 (SQLAlchemy로 실행)
    def ensure_pgvector_and_column(self, schema: str, table: str, vec_col: str, dims: int):
        with self.engine.begin() as conn:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.exec_driver_sql(
                f'ALTER TABLE "{schema}"."{table}" '
                f'ADD COLUMN IF NOT EXISTS "{vec_col}" vector({dims});'
            )

    # ✔ 임시테이블 COPY + UPDATE JOIN (COPY는 raw_connection으로 psycopg2 사용)
    def copy_temp_and_update_vector(
        self,
        schema: str,
        table: str,
        pk_cols: list[str],
        vec_col: str,
        df_vec: pd.DataFrame,   # PK들 + vec_col('[...]' 문자열)
        dims: int,
    ):
        raw = self.engine.raw_connection()  # psycopg2 connection
        try:
            cur = raw.cursor()
            # 1) TEMP TABLE
            pk_defs = ", ".join(f'"{c}" text' for c in pk_cols)
            cur.execute(
                f'CREATE TEMP TABLE tmp_vec ({pk_defs}, "{vec_col}" vector({dims}));'
            )

            # 2) COPY
            buf = io.StringIO()
            df_vec.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            cols_csv = ", ".join([f'"{c}"' for c in pk_cols + [vec_col]])
            cur.copy_expert(
                f'COPY tmp_vec ({cols_csv}) FROM STDIN WITH (FORMAT csv, DELIMITER \',\', NULL \'\\N\', QUOTE \'"\', ESCAPE \'"\')',
                buf,
            )

            # 3) UPDATE JOIN
            on_clause = " AND ".join([f't."{c}" = s."{c}"' for c in pk_cols])
            cur.execute(
                f'UPDATE "{schema}"."{table}" AS t '
                f'SET "{vec_col}" = s."{vec_col}" '
                f'FROM tmp_vec AS s WHERE {on_clause};'
            )
            raw.commit()
        finally:
            raw.close()