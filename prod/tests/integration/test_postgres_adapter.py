"""
tests/integration/test_postgres_adapter.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for PostgresDatabaseAdapter.

Requires a running PostgreSQL instance with the anzsic_db database populated.
These tests are marked @pytest.mark.integration and are SKIPPED in the
standard test run.

Run with:
  pytest -m integration prod/tests/integration/test_postgres_adapter.py -v

Environment:
  DB_DSN defaults to "dbname=anzsic_db"
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def db_adapter():
    """Create a real PostgresDatabaseAdapter for integration testing."""
    from prod.adapters.postgres_db import PostgresDatabaseAdapter
    from prod.config.settings import get_settings
    adapter = PostgresDatabaseAdapter(get_settings())
    yield adapter
    adapter.close()


class TestVectorSearch:
    def test_returns_tuples_with_rank(self, db_adapter):
        """vector_search should return (code, rank) tuples."""
        fake_vec = [0.01] * 768  # Not meaningful but won't crash
        results = db_adapter.vector_search(fake_vec, limit=5)
        assert len(results) <= 5
        for code, rank in results:
            assert isinstance(code, str)
            assert isinstance(rank, int)
            assert rank >= 1

    def test_rank_starts_at_1(self, db_adapter):
        fake_vec = [0.01] * 768
        results = db_adapter.vector_search(fake_vec, limit=10)
        if results:
            ranks = [r for _, r in results]
            assert min(ranks) == 1

    def test_respects_limit(self, db_adapter):
        fake_vec = [0.01] * 768
        results = db_adapter.vector_search(fake_vec, limit=3)
        assert len(results) <= 3

    def test_limit_larger_than_rows_returns_all(self, db_adapter):
        """Asking for more than exist should not raise."""
        fake_vec = [0.01] * 768
        results = db_adapter.vector_search(fake_vec, limit=10_000)
        assert len(results) > 0


class TestFTSSearch:
    def test_known_term_returns_results(self, db_adapter):
        """'mechanic' should match automotive-related codes."""
        results = db_adapter.fts_search("mechanic", limit=10)
        assert len(results) > 0

    def test_returns_rank_1_best(self, db_adapter):
        results = db_adapter.fts_search("plumber", limit=5)
        if results:
            ranks = [r for _, r in results]
            assert min(ranks) == 1

    def test_nonsense_query_returns_empty(self, db_adapter):
        """A query with no matching tokens should return []."""
        results = db_adapter.fts_search("xyzzy_no_such_word_4567", limit=5)
        assert results == []

    def test_respects_limit(self, db_adapter):
        results = db_adapter.fts_search("retail", limit=2)
        assert len(results) <= 2


class TestFetchByCodes:
    def test_known_code_returned(self, db_adapter):
        """A known ANZSIC code should be returned with all expected fields."""
        results = db_adapter.fetch_by_codes(["S9419_03"])
        assert "S9419_03" in results
        rec = results["S9419_03"]
        assert "anzsic_code" in rec
        assert "anzsic_desc" in rec
        assert "division_desc" in rec

    def test_missing_code_silently_omitted(self, db_adapter):
        """Unknown codes are omitted, not raised."""
        results = db_adapter.fetch_by_codes(["DOES_NOT_EXIST_9999"])
        assert "DOES_NOT_EXIST_9999" not in results

    def test_empty_list_returns_empty_dict(self, db_adapter):
        results = db_adapter.fetch_by_codes([])
        assert results == {}

    def test_multiple_codes(self, db_adapter):
        codes = ["S9419_03", "S9411_01"]
        results = db_adapter.fetch_by_codes(codes)
        assert len(results) >= 1  # At least one must exist
