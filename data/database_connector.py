# database_connector.py
import pandas as pd
from sqlalchemy import create_engine
from typing import Dict, Optional
import os
from dotenv import load_dotenv


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
        """데이터베이스 연결"""
        try:
            conn_str = (
                f"postgresql://{self.db_config['user']}:"
                f"{self.db_config['password']}@"
                f"{self.db_config['host']}:"
                f"{self.db_config['port']}/"
                f"{self.db_config['database']}"
            )
            self.engine = create_engine(conn_str)
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