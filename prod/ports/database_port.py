"""
ports/database_port.py
──────────────────────────────────────────────────────────────────────────────
Abstract interface for the vector + FTS database.

The port deliberately separates the three database concerns:
  1. vector_search   — ANN similarity search
  2. fts_search      — keyword / full-text search
  3. fetch_by_codes  — bulk record retrieval by primary key

This separation means:
  • RRF fusion is done in pure Python (services/retriever.py), making it
    trivially unit-testable with no database dependency.
  • Each search method can be mocked or replaced independently.

Current implementation: PostgresDatabaseAdapter (psycopg2 + pgvector)
To swap: write a new adapter (e.g. WeaviateDatabaseAdapter) implementing
this Protocol and change ONE line in services/container.py.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DatabasePort(Protocol):
    """Contract for the hybrid search database backend."""

    def vector_search(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[tuple[str, int]]:
        """Run approximate nearest-neighbour search over stored embeddings.

        Args:
            embedding: Query vector (must match the stored dimension).
            limit:     Maximum number of results.

        Returns:
            List of (anzsic_code, rank) tuples ordered by similarity
            (rank 1 = most similar).

        Raises:
            DatabaseError: On connection or query failure.
        """
        ...

    def fts_search(
        self,
        query_text: str,
        limit: int,
    ) -> list[tuple[str, int]]:
        """Run full-text search using the stored tsvector index.

        Args:
            query_text: Natural-language search string.
            limit:      Maximum number of results.

        Returns:
            List of (anzsic_code, rank) tuples ordered by FTS score
            (rank 1 = highest relevance).

        Raises:
            DatabaseError: On connection or query failure.
        """
        ...

    def fetch_by_codes(self, codes: list[str]) -> dict[str, dict]:
        """Fetch full ANZSIC records for a list of codes.

        Args:
            codes: List of anzsic_code primary keys to retrieve.

        Returns:
            Dict mapping anzsic_code → record dict.
            Codes not found in the database are silently omitted.

        Raises:
            DatabaseError: On connection or query failure.
        """
        ...
