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
import psycopg2.pool
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


# ── Connection pool settings ──────────────────────────────────────────────
# minconn=2   — keep 2 warm connections at all times (avoids cold-start latency)
# maxconn=20  — cap at 20 to stay within PostgreSQL's max_connections limit
#               (raise if you add more Uvicorn workers)
_POOL_MINCONN = 2
_POOL_MAXCONN = 20


class PostgresDatabaseAdapter:
    """psycopg2 + pgvector implementation of DatabasePort.

    Uses a ThreadedConnectionPool so concurrent threads (FastAPI + Uvicorn
    thread pool) each borrow their own connection.  The pool is created once
    per adapter instance and shared across all threads in a process.

    Injected into HybridRetriever via services/container.py.
    """

    def __init__(self, settings: Settings) -> None:
        self._dsn = settings.db_dsn
        self._pool: Any = None
        # Legacy single-connection attribute kept for close() backward compat
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

        Uses OR between stemmed query tokens so that descriptive free-text
        queries like "fixes pipes in industries for AC" match records containing
        ANY of the meaningful terms (pipe, fix, industri, etc.) rather than
        requiring ALL terms to be present in the same record (AND semantics of
        plainto_tsquery would return zero hits for most natural-language inputs).

        Falls back to an empty list rather than raising if no FTS results
        (colloquial queries often produce zero FTS hits — vector covers it).
        """
        sql = """
            SELECT anzsic_code,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(fts_vector, query) DESC
                   ) AS rank
            FROM   anzsic_codes,
                   (SELECT to_tsquery(string_agg(lexeme, ' | '))
                    FROM   unnest(to_tsvector('english', %s))
                   ) AS t(query)
            WHERE  query IS NOT NULL
              AND  fts_vector @@ query
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

    # ── Connection pool helpers ────────────────────────────────────────────

    def _get_pool(self) -> Any:
        """Return (or lazily create) the ThreadedConnectionPool."""
        if self._pool is None:
            try:
                pool = psycopg2.pool.ThreadedConnectionPool(
                    _POOL_MINCONN,
                    _POOL_MAXCONN,
                    self._dsn,
                )
                # Register pgvector on every connection in the pool
                for _ in range(_POOL_MINCONN):
                    conn = pool.getconn()
                    conn.autocommit = True
                    register_vector(conn)
                    pool.putconn(conn)
                logger.info(
                    "PostgresDatabaseAdapter: pool created min=%d max=%d",
                    _POOL_MINCONN,
                    _POOL_MAXCONN,
                )
                self._pool = pool
            except psycopg2.Error as exc:
                raise DatabaseError(f"Cannot create connection pool: {exc}") from exc
        return self._pool

    def _new_conn(self) -> Any:
        """Open a single connection with pgvector registered (pool bootstrap)."""
        try:
            conn = psycopg2.connect(self._dsn)
            conn.autocommit = True
            register_vector(conn)
            logger.debug("PostgresDatabaseAdapter: new connection opened")
            return conn
        except psycopg2.Error as exc:
            raise DatabaseError(f"Cannot connect to database: {exc}") from exc

    def _execute(self, sql: str, params: tuple) -> list[dict]:
        """Execute a query borrowing a connection from the pool.

        The connection is returned to the pool after use, whether the query
        succeeds or fails — so the pool is never exhausted by exceptions.
        """
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            if conn.autocommit is False:
                conn.autocommit = True
                register_vector(conn)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = list(cur.fetchall())
            pool.putconn(conn)
            return rows
        except psycopg2.OperationalError as exc:
            # Connection may have gone stale — discard it and open a fresh one
            logger.warning("DB OperationalError — replacing stale connection: %s", exc)
            pool.putconn(conn, close=True)
            raise DatabaseError(f"DB query failed (stale connection discarded): {exc}") from exc
        except Exception:
            pool.putconn(conn)
            raise

    def close(self) -> None:
        """Close all connections in the pool (called on process shutdown)."""
        if self._pool:
            self._pool.closeall()
            logger.debug("PostgresDatabaseAdapter: pool closed")
