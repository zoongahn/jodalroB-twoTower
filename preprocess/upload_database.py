from __future__ import annotations

import io
import re
from typing import Dict, List, Tuple, Optional
import os

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql


EMB_RE = re.compile(r"^(?P<base>.+)_emb(?P<idx>\d{3})$")


class DataUploader:
    # 🔹 클래스 공유 커넥션( close_shared 용 )
    conn: Optional["psycopg.extensions.connection"] = None

    def __init__(
        self,
        create_schema: bool = True,
        if_exists: str = "replace",  # fail | replace | append
    ):
        self.create_schema = create_schema
        self.if_exists = if_exists
        self.schema = "public"
        self.conn = self._get_conn()
    
    def set_if_exists(self, mode:str):
        assert mode in {"fail", "replace", "append"}
        self.if_exists = mode

    def _build_dsn(self) -> str:
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db   = os.getenv("POSTGRES_DB")
        user = os.getenv("POSTGRES_USER")
        pwd  = os.getenv("POSTGRES_PASSWORD")
        if not all([db, user, pwd]):
            raise ValueError("POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD 환경변수가 필요합니다.")
        return f"host={host} port={port} dbname={db} user={user} password={pwd}"

    def _get_conn(self):
        # 🔹 없으면 만들고, 클래스 변수에 캐시
        if DataUploader.conn is not None:
            return DataUploader.conn
        dsn = self._build_dsn()
        DataUploader.conn = psycopg.connect(dsn)
        DataUploader.conn.autocommit = False
        return DataUploader.conn

    @classmethod
    def close_shared(cls):
        """공유 커넥션 닫기"""
        if cls.conn is not None:
            try:
                cls.conn.close()
            finally:
                cls.conn = None

    # ------------------------- public API -------------------------
    def upload_df(self, df: pd.DataFrame, base_table_name: str, pk_cols: List[str]) -> None:
        table_pre = f"{base_table_name}_preprocessed"

        # Defensive copy and PK -> string to preserve leading zeros
        df = df.copy()
        for pk in pk_cols:
            if pk in df.columns:
                df[pk] = df[pk].astype("string").fillna("")

        # 🔹 기존 conn_dsn 기반 신규 연결(X) → __init__에서 만든 self.conn 재사용(O)
        conn = self.conn
        try:
            conn.autocommit = False
            # 🔹 create_schema_flag → create_schema
            if self.create_schema:
                self._ensure_schema(conn)
            self._ensure_pgvector(conn)

            exists = self._table_exists(conn, self.schema, table_pre)
            if exists:
                if self.if_exists == "fail":
                    raise RuntimeError(f"Table {self.schema}.{table_pre} already exists (if_exists=fail)")
                elif self.if_exists == "replace":
                    self._drop_table(conn, self.schema, table_pre)
                # append -> keep existing

            if (not exists) or (self.if_exists == "replace"):
                self._create_table_from_df(conn, df, self.schema, table_pre, pk_cols)

            # Prepare rows (collapse embeddings to vector columns)
            df2, vector_meta = self._collapse_embeddings(df, pk_cols)

            # COPY
            self._copy_dataframe(conn, df2, self.schema, table_pre, vector_meta)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------- helpers: DDL -------------------------
    def _ensure_schema(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(self.schema)))
        conn.commit()

    def _ensure_pgvector(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

    def _table_exists(self, conn, schema: str, table: str) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", [f"{schema}.{table}"])
            return cur.fetchone()[0] is not None

    def _drop_table(self, conn, schema: str, table: str) -> None:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE;").format(
                sql.Identifier(schema), sql.Identifier(table)
            ))
        conn.commit()

    def _map_dtype_to_pg(self, s: pd.Series) -> str:
        dt = str(s.dtype)
        # pandas nullable 정수/실수 대응
        if dt.lower().startswith("int"):
            return "bigint"
        if dt.lower().startswith("float"):
            return "double precision"
        if dt == "bool":
            return "boolean"
        return "text"  # object, string, etc.

    def _group_embeddings(self, columns: List[str]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for c in columns:
            m = EMB_RE.match(str(c))
            if not m:
                continue
            base = m.group("base")
            groups.setdefault(base, []).append(c)
        # sort by numeric idx
        def _order_key(label: str) -> int:
            return int(EMB_RE.match(str(label)).group("idx"))
        for base, cols in groups.items():
            groups[base] = sorted(cols, key=_order_key)
        return groups

    def _create_table_from_df(self, conn, df: pd.DataFrame, schema: str, table: str, pk_cols: List[str]) -> None:
        emb_groups = self._group_embeddings(df.columns.tolist())
        vector_cols = set(emb_groups.keys())

        col_defs = []
        used = set()

        # 1) PK first (TEXT NOT NULL)
        for pk in pk_cols:
            col_defs.append(sql.SQL("{} {} NOT NULL").format(sql.Identifier(pk), sql.SQL("text")))
            used.add(pk)

        # 2) Regular columns (skip raw emb###, skip vector base names)
        emb_members = set(c for cols in emb_groups.values() for c in cols)
        for col in df.columns:
            if col in used:
                continue
            if col in emb_members:
                continue
            if col in vector_cols:
                continue
            pg_type = self._map_dtype_to_pg(df[col])
            col_defs.append(sql.SQL("{} {}").format(sql.Identifier(str(col)), sql.SQL(pg_type)))
            used.add(col)

        # 3) Vector columns
        for base, cols in emb_groups.items():
            dims = len(cols)
            # 🔹 sql.Literal(dims) → 따옴표가 붙어서 invalid. 문자열로 직접 삽입.
            col_defs.append(sql.SQL("{} vector({})").format(sql.Identifier(base), sql.SQL(str(dims))))
            used.add(base)

        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE TABLE {}.{} ({});").format(
                sql.Identifier(schema), sql.Identifier(table), sql.SQL(", ").join(col_defs)
            ))
            if pk_cols:
                pk_name = f"pk_{table}"
                cur.execute(sql.SQL("ALTER TABLE {}.{} ADD CONSTRAINT {} PRIMARY KEY ({});").format(
                    sql.Identifier(schema), sql.Identifier(table), sql.Identifier(pk_name),
                    sql.SQL(", ").join([sql.Identifier(c) for c in pk_cols])
                ))
        conn.commit()

    # ------------------------- helpers: data shaping -------------------------
    def _collapse_embeddings(self, df: pd.DataFrame, pk_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Return (df2, vector_meta). df2 has pgvector base columns and no *_emb### columns.
        vector_meta maps {base: dims}.
        Any non-finite vector row -> NULL for that vector cell.
        """
        emb_groups = self._group_embeddings(df.columns.tolist())
        if not emb_groups:
            # ensure PK string cast (again) and return as-is
            df2 = df.copy()
            for pk in pk_cols:
                if pk in df2.columns:
                    df2[pk] = df2[pk].astype("string").fillna("")
            return df2, {}

        df2 = df.copy()

        # Create vector columns as text literals like "[1,2,3]"
        for base, cols in emb_groups.items():
            # select and cast to float
            sub = df2[cols].astype(float).to_numpy()
            # build per row
            vec_strings: List[Optional[str]] = []
            for row in sub:
                if not np.all(np.isfinite(row)):
                    vec_strings.append(None)  # NULL
                else:
                    vec_strings.append("[" + ",".join(f"{x:.6g}" for x in row) + "]")
            df2[base] = pd.Series(vec_strings, index=df2.index, dtype="string")

        # drop raw emb### columns
        all_emb_members = [c for cols in emb_groups.values() for c in cols]
        df2.drop(columns=all_emb_members, inplace=True)

        vector_meta = {base: len(cols) for base, cols in emb_groups.items()}
        return df2, vector_meta

    # ------------------------- COPY -------------------------
    def _copy_dataframe(self, conn, df: pd.DataFrame, schema: str, table: str, vector_meta: Dict[str, int]) -> None:
        r"""COPY FROM STDIN with CSV. We explicitly list the columns.
        We represent NULL as \N to let COPY parse nulls correctly.
        """
        cols = [str(c) for c in df.columns]
        copy_sql = sql.SQL(
            "COPY {}.{} ({}) FROM STDIN WITH (FORMAT csv, DELIMITER ',', NULL '\\N', QUOTE '\"', ESCAPE '\"')"
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(', ').join(sql.Identifier(c) for c in cols)
        )

        # Build an in-memory CSV WITHOUT headers; use na_rep='\\N' so NULLs go unquoted
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep='\\N')
        buf.seek(0)

        with conn.cursor() as cur:
            # psycopg3에서는 copy_expert 대신 copy 사용
            try:
                # psycopg3 방식
                with cur.copy(copy_sql.as_string()) as copy:
                    while True:
                        data = buf.read(8192)  # 8KB 청크로 읽기
                        if not data:
                            break
                        copy.write(data)
            except AttributeError:
                # psycopg2 호환성을 위한 폴백
                cur.copy_expert(copy_sql.as_string(conn), file=buf)
        # No commit here; caller commits