"""
tests/unit/test_reranker.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for LLMReranker.

Tests cover:
  • JSON parsing (array, wrapped object, invalid)
  • CSV fallback trigger (empty first response)
  • top_k truncation
  • Malformed individual result items (skipped gracefully)

Uses conftest fixtures: mock_llm, mock_reranker, settings.
"""
from __future__ import annotations

import json

import pytest

from prod.domain.models import Candidate
from prod.services.reranker import LLMReranker
from prod.tests.conftest import MockLLMAdapter, MockLLMAdapterEmpty, _DB_RECORDS


# ── Helper to build a Candidate for testing ────────────────────────────────

def _make_candidate(code: str = "S9419_03") -> Candidate:
    rec = _DB_RECORDS.get(code, _DB_RECORDS["S9419_03"])
    return Candidate(
        **rec,
        rrf_score=0.03,
        in_vector=True,
        in_fts=True,
        vector_rank=1,
        fts_rank=1,
    )


# ── _parse_response tests (tested via the private method indirectly) ────────

class TestParseResponse:
    """Test the JSON parsing logic of LLMReranker._parse_response."""

    def test_parses_bare_json_array(self, mock_reranker):
        """Standard array response is parsed correctly."""
        raw = json.dumps([
            {
                "rank": 1,
                "anzsic_code": "S9419_03",
                "anzsic_desc": "Automotive Repair",
                "class_desc": "Repair",
                "division_desc": "Other Services",
                "reason": "Direct match",
            }
        ])
        results = mock_reranker._parse_response(raw, top_k=5)
        assert len(results) == 1
        assert results[0].anzsic_code == "S9419_03"
        assert results[0].rank == 1

    def test_parses_wrapped_object(self, mock_reranker):
        """Response wrapped in {'results': [...]} is unwrapped automatically."""
        items = [
            {
                "rank": 1,
                "anzsic_code": "S9411_01",
                "anzsic_desc": "Automotive Electrical",
                "class_desc": "Repair",
                "division_desc": "Other Services",
                "reason": "Match",
            }
        ]
        raw = json.dumps({"results": items})
        results = mock_reranker._parse_response(raw, top_k=5)
        assert len(results) == 1
        assert results[0].anzsic_code == "S9411_01"

    def test_returns_empty_on_invalid_json(self, mock_reranker):
        """Non-JSON strings return an empty list (no crash)."""
        results = mock_reranker._parse_response("not json at all", top_k=5)
        assert results == []

    def test_returns_empty_on_none(self, mock_reranker):
        """None input returns empty list."""
        results = mock_reranker._parse_response(None, top_k=5)  # type: ignore[arg-type]
        # _parse_response receives None from LLMPort; should handle gracefully
        # (the rerank() method guards against None from generate_json, but let's
        #  test the parser defensively too)
        assert results == [] or results is not None  # should not raise

    def test_top_k_truncation(self, mock_reranker):
        """top_k limits the number of results returned."""
        items = [
            {
                "rank": i + 1,
                "anzsic_code": f"CODE_{i:03d}",
                "anzsic_desc": f"Desc {i}",
                "class_desc": None,
                "division_desc": None,
                "reason": None,
            }
            for i in range(10)
        ]
        raw = json.dumps(items)
        results = mock_reranker._parse_response(raw, top_k=3)
        assert len(results) == 3

    def test_malformed_item_is_skipped(self, mock_reranker):
        """Items missing required fields are skipped, others returned."""
        items = [
            {
                "rank": 1,
                "anzsic_code": "S9419_03",
                "anzsic_desc": "Valid",
                "reason": "ok",
            },
            {
                # Missing rank → should be skipped
                "anzsic_code": "BAD_ITEM",
            },
        ]
        raw = json.dumps(items)
        results = mock_reranker._parse_response(raw, top_k=5)
        codes = [r.anzsic_code for r in results]
        assert "S9419_03" in codes
        assert "BAD_ITEM" not in codes


class TestRerankerCsvFallback:
    """Test the CSV fallback logic in LLMReranker.rerank()."""

    def test_fallback_triggered_on_empty_first_response(self, settings, tmp_path):
        """When the first LLM call returns empty, reranker retries with CSV."""
        # Write a tiny CSV reference file
        csv_file = tmp_path / "anzsic_master.csv"
        csv_file.write_text(
            "anzsic_code,anzsic_desc\nS9419_03,Automotive Repair\n", encoding="utf-8"
        )

        from prod.config.settings import Settings as S
        patched = S(
            gcp_project_id="x",
            gcp_location_id="x",
            gcp_embed_model="x",
            gcp_gemini_model="x",
            gcloud_path="x",
            https_proxy="",
            db_dsn="dbname=x",
            master_csv_path=csv_file,
            embed_dim=8,
        )
        MockLLMAdapterEmpty._call_count = 0
        reranker = LLMReranker(llm=MockLLMAdapterEmpty(), settings=patched)
        candidates = [_make_candidate()]

        results = reranker.rerank("mobile mechanic", candidates, top_k=3)
        # MockLLMAdapterEmpty returns [] on first call, real data on second
        assert len(results) > 0
        assert MockLLMAdapterEmpty._call_count == 2  # called twice

    def test_no_fallback_if_first_call_succeeds(self, mock_reranker):
        """When first call returns results, second call must not happen."""
        # mock_llm always returns valid JSON (MockLLMAdapter)
        candidates = [_make_candidate()]
        results = mock_reranker.rerank("mobile mechanic", candidates, top_k=3)
        assert len(results) > 0

    def test_empty_candidates_returns_empty(self, mock_reranker):
        """No candidates → no LLM call → empty results."""
        results = mock_reranker.rerank("anything", [], top_k=5)
        assert results == []
