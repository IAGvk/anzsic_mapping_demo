"""
adapters/postgres_db.py
──────────────────────────────────────────────────────────────────────────────
Implements DatabasePort using psycopg2 + pgvector.

Database layout (from ingest.py):
  Table : anzsic_codes
  Cols  : anzsic_code (PK), anzsic_desc, class_code, class_desc,
          group_code, group_desc, subdivision_desc, division_desc,
          class_exclusions, enriched_text, embedding vector(768), fts_vector
  Index : HNSW cosine (embedding), GIN (fts_vector)

Three atomic methods match DatabasePort:
  vector_search  → ANN search via pgvector <=> operator
  fts_search     → FTS via tsquery
  fetch_by_codes → bulk SELECT by primary key list

Connection management:
  - A single connection is opened lazily and reused.
  - On OperationalError the connection is reset and one retry is attempted.
  - For Streamlit (single-process, single-thread) this is sufficient.
  - For FastAPI: replace with a psycopg2 connection pool (e.g. psycopg2.pool.
    ThreadedConnectionPool) — change only this file.

To swap the database engine (e.g. to Weaviate or Pinecone):
  1. Write WeaviateDatabaseAdapter implementing DatabasePort
  2. Change ONE import in services/container.py
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from prod.config.settings import Settings
from prod.domain.exceptions import DatabaseError

logger = logging.getLogger(__name__)

# Columns returned for every ANZSIC record (must match domain/models.py Candidate)
_RECORD_COLS = (
    "anzsic_code",
    "anzsic_desc",
    "class_code",
    "class_desc",
    "group_code",
    "group_desc",
    "subdivision_desc",
    "division_desc",
    "class_exclusions",
    "enriched_text",
)
_SELECT_COLS = ", ".join(_RECORD_COLS)


class PostgresDatabaseAdapter:
    """psycopg2 + pgvector implementation of DatabasePort.

    Injected into HybridRetriever via services/container.py.
    """

    def __init__(self, settings: Settings) -> None:
        self._dsn = settings.db_dsn
        self._conn: Any = None
        logger.debug("PostgresDatabaseAdapter ready | dsn=%s", self._dsn)

    # ── DatabasePort implementation ────────────────────────────────────────

    def vector_search(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[tuple[str, int]]:
        """Approximate nearest-neighbour search via pgvector HNSW index.

        Returns list of (anzsic_code, rank) tuples, rank starting at 1.
        """
        sql = """
            SELECT anzsic_code,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
            FROM   anzsic_codes
            WHERE  embedding IS NOT NULL
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s
        """
        try:
            rows = self._execute(sql, (embedding, embedding, limit))
            return [(row["anzsic_code"], row["rank"]) for row in rows]
        except Exception as exc:
            raise DatabaseError(f"vector_search failed: {exc}") from exc

    def fts_search(
        self,
        query_text: str,
        limit: int,
    ) -> list[tuple[str, int]]:
        """Full-text search using the GIN-indexed tsvector column.

        Falls back to an empty list rather than raising if no FTS results
        (colloquial queries often produce zero FTS hits — vector covers it).
        """
        sql = """
            SELECT anzsic_code,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(fts_vector, query) DESC
                   ) AS rank
            FROM   anzsic_codes,
                   plainto_tsquery('english', %s) AS query
            WHERE  fts_vector @@ query
            ORDER  BY ts_rank_cd(fts_vector, query) DESC
            LIMIT  %s
        """
        try:
            rows = self._execute(sql, (query_text, limit))
            return [(row["anzsic_code"], row["rank"]) for row in rows]
        except Exception as exc:
            logger.warning("fts_search error (returning empty): %s", exc)
            return []

    def fetch_by_codes(self, codes: list[str]) -> dict[str, dict]:
        """Fetch full records for a list of ANZSIC codes.

        Returns a dict keyed by anzsic_code.  Missing codes are absent.
        """
        if not codes:
            return {}
        sql = f"""
            SELECT {_SELECT_COLS}
            FROM   anzsic_codes
            WHERE  anzsic_code = ANY(%s)
        """
        try:
            rows = self._execute(sql, (codes,))
            return {row["anzsic_code"]: dict(row) for row in rows}
        except Exception as exc:
            raise DatabaseError(f"fetch_by_codes failed: {exc}") from exc

    # ── Connection helpers ─────────────────────────────────────────────────

    def _get_conn(self) -> Any:
        """Return an open connection, creating or reusing one."""
        if self._conn is None or self._conn.closed:
            self._conn = self._new_conn()
        return self._conn

    def _new_conn(self) -> Any:
        """Open a fresh psycopg2 connection with pgvector registered."""
        try:
            conn = psycopg2.connect(self._dsn)
            register_vector(conn)
            conn.autocommit = True
            logger.debug("PostgresDatabaseAdapter: new connection opened")
            return conn
        except psycopg2.Error as exc:
            raise DatabaseError(f"Cannot connect to database: {exc}") from exc

    def _execute(self, sql: str, params: tuple) -> list[dict]:
        """Execute a query and return rows as dicts, with one auto-reconnect."""
        for attempt in (1, 2):
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, params)
                    return list(cur.fetchall())
            except psycopg2.OperationalError as exc:
                if attempt == 1:
                    logger.warning("DB OperationalError — reconnecting: %s", exc)
                    self._conn = None
                else:
                    raise DatabaseError(f"DB query failed after reconnect: {exc}") from exc
        return []  # unreachable

    def close(self) -> None:
        """Explicitly close the connection (optional — GC handles it otherwise)."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.debug("PostgresDatabaseAdapter: connection closed")
