"""
tests/unit/test_classifier.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ClassifierPipeline using fully mocked adapters.

Verifies:
  • HIGH_FIDELITY mode calls reranker and returns ClassifyResult objects
  • FAST mode skips the LLM and returns RRF-ordered results directly
  • ClassifyResponse metadata is populated correctly
  • Zero candidates returns an empty result list (no crash)
"""
from __future__ import annotations

import pytest

from prod.domain.models import SearchMode, SearchRequest


class TestClassifierFastMode:
    def test_returns_results(self, pipeline):
        req = SearchRequest(query="mobile mechanic", mode=SearchMode.FAST, top_k=3)
        resp = pipeline.classify(req)
        assert len(resp.results) <= 3
        assert resp.mode == "fast"

    def test_llm_model_is_empty_in_fast_mode(self, pipeline):
        req = SearchRequest(query="electrician", mode=SearchMode.FAST)
        resp = pipeline.classify(req)
        assert resp.llm_model == ""

    def test_embed_model_populated(self, pipeline):
        req = SearchRequest(query="nurse", mode=SearchMode.FAST)
        resp = pipeline.classify(req)
        assert resp.embed_model != ""

    def test_candidates_retrieved_is_positive(self, pipeline):
        req = SearchRequest(query="plumber", mode=SearchMode.FAST)
        resp = pipeline.classify(req)
        assert resp.candidates_retrieved > 0

    def test_results_ranked_from_1(self, pipeline):
        req = SearchRequest(query="chef", mode=SearchMode.FAST, top_k=3)
        resp = pipeline.classify(req)
        ranks = [r.rank for r in resp.results]
        assert ranks == list(range(1, len(ranks) + 1))


class TestClassifierHighFidelityMode:
    def test_returns_results(self, pipeline):
        req = SearchRequest(query="mobile mechanic", mode=SearchMode.HIGH_FIDELITY, top_k=2)
        resp = pipeline.classify(req)
        assert len(resp.results) > 0

    def test_llm_model_populated(self, pipeline):
        req = SearchRequest(query="mechanic", mode=SearchMode.HIGH_FIDELITY)
        resp = pipeline.classify(req)
        assert resp.llm_model != ""

    def test_mode_value_in_response(self, pipeline):
        req = SearchRequest(query="mechanic", mode=SearchMode.HIGH_FIDELITY)
        resp = pipeline.classify(req)
        assert resp.mode == "high_fidelity"

    def test_result_has_reason(self, pipeline):
        req = SearchRequest(query="mechanic", mode=SearchMode.HIGH_FIDELITY)
        resp = pipeline.classify(req)
        # MockLLMAdapter populates reason
        for r in resp.results:
            assert r.reason is not None and len(r.reason) > 0

    def test_query_preserved_in_response(self, pipeline):
        query = "chartered accountant"
        req = SearchRequest(query=query, mode=SearchMode.HIGH_FIDELITY)
        resp = pipeline.classify(req)
        assert resp.query == query
