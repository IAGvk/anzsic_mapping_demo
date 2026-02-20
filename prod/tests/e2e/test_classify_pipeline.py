"""
tests/e2e/test_classify_pipeline.py
──────────────────────────────────────────────────────────────────────────────
End-to-end pipeline tests using mock adapters.

These run WITHOUT real GCP or DB connections (mock adapters from conftest.py).
They test the full pipeline wiring: SearchRequest → ClassifyResponse.

For real end-to-end tests that hit live services, see the integration/ folder
and run with: pytest -m integration
"""
from __future__ import annotations

import json

import pytest

from prod.domain.models import ClassifyResponse, SearchMode, SearchRequest


class TestFullPipelineHighFidelity:
    def test_basic_classify_returns_response(self, pipeline):
        req = SearchRequest(
            query="mobile mechanic",
            mode=SearchMode.HIGH_FIDELITY,
            top_k=3,
        )
        resp = pipeline.classify(req)
        assert isinstance(resp, ClassifyResponse)
        assert len(resp.results) > 0

    def test_response_has_correct_query(self, pipeline):
        q = "chartered accountant"
        resp = pipeline.classify(SearchRequest(query=q))
        assert resp.query == q

    def test_results_are_ranked(self, pipeline):
        resp = pipeline.classify(SearchRequest(query="nurse"))
        ranks = [r.rank for r in resp.results]
        assert ranks == sorted(ranks)

    def test_result_fields_populated(self, pipeline):
        resp = pipeline.classify(SearchRequest(query="plumber"))
        for r in resp.results:
            assert r.anzsic_code
            assert r.anzsic_desc
            assert r.rank >= 1

    def test_to_dict_is_json_serialisable(self, pipeline):
        resp = pipeline.classify(SearchRequest(query="electrician"))
        serialised = json.dumps(resp.to_dict())
        assert "results" in serialised


class TestFullPipelineFastMode:
    def test_fast_mode_returns_results(self, pipeline):
        req = SearchRequest(query="barista", mode=SearchMode.FAST)
        resp = pipeline.classify(req)
        assert len(resp.results) > 0
        assert resp.mode == "fast"
        assert resp.llm_model == ""

    def test_fast_mode_reason_contains_rrf_score(self, pipeline):
        req = SearchRequest(query="barista", mode=SearchMode.FAST, top_k=1)
        resp = pipeline.classify(req)
        assert resp.results
        assert "RRF" in resp.results[0].reason


class TestBatchClassify:
    """Simulate batch processing of multiple queries."""

    QUERIES = [
        "mobile mechanic",
        "cafe owner",
        "registered nurse",
        "software engineer",
        "primary school teacher",
    ]

    def test_all_queries_return_results(self, pipeline):
        for query in self.QUERIES:
            req = SearchRequest(query=query, mode=SearchMode.FAST, top_k=3)
            resp = pipeline.classify(req)
            assert len(resp.results) > 0, f"No results for: {query}"

    def test_no_query_raises_unhandled_exception(self, pipeline):
        """Ensure classification errors don't propagate without context."""
        # A very short query is valid; only an empty string should fail at the model level
        req = SearchRequest(query="a", mode=SearchMode.FAST)
        resp = pipeline.classify(req)
        # Should not raise — may return empty results
        assert resp is not None


class TestEdgeCases:
    def test_very_long_query(self, pipeline):
        long_q = "person who fixes cars and vans and trucks at the customer's home " * 5
        req = SearchRequest(query=long_q.strip(), mode=SearchMode.FAST)
        resp = pipeline.classify(req)
        assert resp is not None

    def test_unicode_query(self, pipeline):
        req = SearchRequest(query="médecin généraliste", mode=SearchMode.FAST)
        resp = pipeline.classify(req)
        assert resp.query == "médecin généraliste"

    def test_top_k_1_returns_single_result(self, pipeline):
        req = SearchRequest(query="nurse", mode=SearchMode.HIGH_FIDELITY, top_k=1)
        resp = pipeline.classify(req)
        assert len(resp.results) <= 1
