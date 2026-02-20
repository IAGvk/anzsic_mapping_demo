"""
tests/unit/test_models.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for domain model validation (Pydantic).

Tests cover:
  • Required fields enforcement
  • Default values
  • SearchMode enum values
  • ClassifyResponse.to_dict() serialisation
  • SearchRequest validator constraints
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from prod.domain.models import (
    Candidate,
    ClassifyResponse,
    ClassifyResult,
    SearchMode,
    SearchRequest,
)


class TestSearchMode:
    def test_fast_value(self):
        assert SearchMode.FAST.value == "fast"

    def test_high_fidelity_value(self):
        assert SearchMode.HIGH_FIDELITY.value == "high_fidelity"

    def test_from_string(self):
        assert SearchMode("fast") == SearchMode.FAST


class TestSearchRequest:
    def test_minimal_valid(self):
        r = SearchRequest(query="plumber")
        assert r.query == "plumber"
        assert r.top_k == 5
        assert r.retrieval_n == 20
        assert r.mode == SearchMode.HIGH_FIDELITY

    def test_mode_enum_accepted(self):
        r = SearchRequest(query="x", mode=SearchMode.FAST)
        assert r.mode == SearchMode.FAST

    def test_mode_string_accepted(self):
        r = SearchRequest(query="x", mode="fast")
        assert r.mode == SearchMode.FAST

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest()  # type: ignore[call-arg]

    def test_empty_query_raises(self):
        """Empty string should be rejected."""
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_top_k_minimum(self):
        """top_k must be at least 1."""
        with pytest.raises(ValidationError):
            SearchRequest(query="x", top_k=0)

    def test_retrieval_n_minimum(self):
        """retrieval_n must be at least 1."""
        with pytest.raises(ValidationError):
            SearchRequest(query="x", retrieval_n=0)


class TestCandidate:
    def test_required_fields(self):
        """Only anzsic_code and anzsic_desc are required."""
        c = Candidate(anzsic_code="S9419_03", anzsic_desc="Automotive Repair")
        assert c.anzsic_code == "S9419_03"
        assert c.rrf_score == 0.0
        assert c.in_vector is False
        assert c.in_fts is False

    def test_optional_fields_default_none(self):
        c = Candidate(anzsic_code="X", anzsic_desc="Y")
        assert c.class_code is None
        assert c.division_desc is None

    def test_source_label_both(self):
        c = Candidate(anzsic_code="X", anzsic_desc="Y", in_vector=True, in_fts=True)
        assert c.source_label == "BOTH"

    def test_source_label_vector_only(self):
        c = Candidate(anzsic_code="X", anzsic_desc="Y", in_vector=True, in_fts=False)
        assert c.source_label == "VEC"

    def test_source_label_fts_only(self):
        c = Candidate(anzsic_code="X", anzsic_desc="Y", in_vector=False, in_fts=True)
        assert c.source_label == "FTS"

    def test_source_label_neither(self):
        c = Candidate(anzsic_code="X", anzsic_desc="Y", in_vector=False, in_fts=False)
        assert c.source_label == "\u2014"  # em-dash


class TestClassifyResult:
    def test_minimal_valid(self):
        r = ClassifyResult(rank=1, anzsic_code="S9419_03", anzsic_desc="Repair")
        assert r.rank == 1

    def test_rank_is_integer(self):
        """rank is stored as int; negative values are technically accepted by the model
        but the pipeline always produces rank >= 1."""
        r = ClassifyResult(rank=1, anzsic_code="X", anzsic_desc="Y")
        assert isinstance(r.rank, int)

    def test_optional_fields(self):
        r = ClassifyResult(rank=1, anzsic_code="X", anzsic_desc="Y")
        assert r.reason is None
        assert r.class_desc is None


class TestClassifyResponse:
    def _make_response(self):
        results = [
            ClassifyResult(
                rank=1,
                anzsic_code="S9419_03",
                anzsic_desc="Automotive Repair",
                class_desc="Other Repair",
                division_desc="Other Services",
                reason="Best match",
            )
        ]
        return ClassifyResponse(
            query="mobile mechanic",
            mode="high_fidelity",
            results=results,
            candidates_retrieved=20,
            embed_model="text-embedding-005",
            llm_model="gemini-2.5-flash",
        )

    def test_to_dict_keys(self):
        d = self._make_response().to_dict()
        assert "query" in d
        assert "mode" in d
        assert "results" in d
        assert "candidates_retrieved" in d
        assert "generated_at" in d

    def test_to_dict_results_is_list(self):
        d = self._make_response().to_dict()
        assert isinstance(d["results"], list)
        assert len(d["results"]) == 1

    def test_to_dict_is_json_serialisable(self):
        import json
        d = self._make_response().to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert "S9419_03" in serialised
