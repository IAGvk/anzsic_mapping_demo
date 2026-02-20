"""
tests/unit/test_rrf_fusion.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the compute_rrf() pure function.

No mocks, no I/O, no fixtures needed — just plain arithmetic assertions.
These tests document and enforce the RRF behaviour contract.
"""
from __future__ import annotations

import pytest

from prod.services.retriever import compute_rrf


class TestComputeRRF:
    """Tests for Reciprocal Rank Fusion scoring."""

    def test_code_in_both_systems_has_highest_score(self):
        """A code appearing in both vector and FTS should outscore single-system codes."""
        vec_hits = [("A", 1), ("B", 2), ("C", 3)]
        fts_hits = [("A", 2), ("D", 1)]

        results = {r.anzsic_code: r for r in compute_rrf(vec_hits, fts_hits)}

        # A appears in both → combined score
        assert results["A"].rrf_score > results["B"].rrf_score  # B only in vec
        assert results["A"].rrf_score > results["D"].rrf_score  # D only in fts

    def test_rrf_scores_are_positive(self):
        """All RRF scores must be strictly positive."""
        hits = [("X", 1), ("Y", 2)]
        results = compute_rrf(hits, hits)
        for r in results:
            assert r.rrf_score > 0

    def test_higher_rank_gives_lower_score(self):
        """Within a single system, rank 1 should produce a higher score than rank 2."""
        vec_hits = [("A", 1), ("B", 2)]
        results = {r.anzsic_code: r for r in compute_rrf(vec_hits, [])}
        assert results["A"].rrf_score > results["B"].rrf_score

    def test_empty_fts_returns_vec_only_results(self):
        """When FTS returns nothing, only vector hits should be in output."""
        vec_hits = [("A", 1), ("B", 2)]
        results = compute_rrf(vec_hits, [])
        codes = {r.anzsic_code for r in results}
        assert codes == {"A", "B"}

    def test_empty_vec_returns_fts_only_results(self):
        """When vector returns nothing, only FTS hits should be in output."""
        fts_hits = [("C", 1), ("D", 2)]
        results = compute_rrf([], fts_hits)
        codes = {r.anzsic_code for r in results}
        assert codes == {"C", "D"}

    def test_both_empty_returns_empty(self):
        """No inputs → no output."""
        assert compute_rrf([], []) == []

    def test_provenance_flags(self):
        """in_vector and in_fts flags must reflect which systems returned the code."""
        vec_hits = [("A", 1)]
        fts_hits = [("B", 1), ("A", 2)]
        results = {r.anzsic_code: r for r in compute_rrf(vec_hits, fts_hits)}

        assert results["A"].in_vector is True
        assert results["A"].in_fts is True
        assert results["B"].in_vector is False
        assert results["B"].in_fts is True

    def test_rank_values_preserved(self):
        """vector_rank and fts_rank should carry the original ranks."""
        vec_hits = [("A", 3)]
        fts_hits = [("A", 7)]
        results = {r.anzsic_code: r for r in compute_rrf(vec_hits, fts_hits)}
        assert results["A"].vector_rank == 3
        assert results["A"].fts_rank == 7

    def test_k_constant_affects_score_magnitude(self):
        """Larger k produces smaller individual scores (flatter distribution)."""
        hits = [("A", 1)]
        score_k10 = compute_rrf(hits, [], k=10)[0].rrf_score
        score_k60 = compute_rrf(hits, [], k=60)[0].rrf_score
        score_k200 = compute_rrf(hits, [], k=200)[0].rrf_score
        assert score_k10 > score_k60 > score_k200

    def test_score_formula_correctness(self):
        """Verify the exact RRF formula: score = 1/(k + rank)."""
        vec_hits = [("A", 1)]
        fts_hits = [("A", 2)]
        k = 60
        results = {r.anzsic_code: r for r in compute_rrf(vec_hits, fts_hits, k=k)}
        expected = 1 / (k + 1) + 1 / (k + 2)
        assert abs(results["A"].rrf_score - expected) < 1e-10

    def test_duplicate_codes_are_merged(self):
        """The same code appearing twice in one list should not double-count."""
        # Normally shouldn't happen but protect against it
        vec_hits = [("A", 1), ("A", 2)]  # malformed input
        fts_hits = []
        results = compute_rrf(vec_hits, fts_hits)
        # dict() keeps last value, so "A" should appear exactly once
        codes = [r.anzsic_code for r in results]
        assert len(codes) == len(set(codes))

    def test_large_input_does_not_raise(self):
        """Should handle large ranked lists without error."""
        vec_hits = [(f"CODE_{i:04d}", i + 1) for i in range(1000)]
        fts_hits = [(f"CODE_{i:04d}", i + 1) for i in range(500, 1500)]
        results = compute_rrf(vec_hits, fts_hits)
        assert len(results) > 0
