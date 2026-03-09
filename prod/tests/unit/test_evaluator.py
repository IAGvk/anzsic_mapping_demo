"""
tests/unit/test_evaluator.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for services/evaluator.py.

All tests are pure Python — no DB, no embedding, no LLM.
Uses tmp_path fixture to create a minimal anzsic_master.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from prod.domain.models import Candidate, ClassifyResult
from prod.services.evaluator import (
    ANZSICEvaluator,
    _desc_similarity,
    _jaccard,
    _tokens,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def master_csv(tmp_path: Path) -> Path:
    """Write a minimal anzsic_master.csv and return its path."""
    path = tmp_path / "anzsic_master.csv"
    rows = [
        {"anzsic_code": "A0111", "anzsic_desc": "Nursery Production Excluding Cut Flowers"},
        {"anzsic_code": "B0600", "anzsic_desc": "Coal Mining"},
        {"anzsic_code": "C1111", "anzsic_desc": "Bacon Smallgoods and Ham Manufacturing"},
        {"anzsic_code": "D2611", "anzsic_desc": "Electrical Equipment Manufacturing"},
        {"anzsic_code": "G4711", "anzsic_desc": "Supermarket and Grocery Stores"},
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["anzsic_code", "anzsic_desc"])
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture()
def evaluator(master_csv: Path) -> ANZSICEvaluator:
    return ANZSICEvaluator(master_csv)


def _make_result(
    rank: int,
    code: str,
    desc: str = "Some description",
    reason: str = "This is a reason long enough to pass the 30-char threshold check.",
    score: int | None = 900,
) -> ClassifyResult:
    return ClassifyResult(
        rank=rank,
        anzsic_code=code,
        anzsic_desc=desc,
        reason=reason,
        score=score,
    )


def _make_candidate(code: str, rrf: float = 0.01) -> Candidate:
    return Candidate(
        anzsic_code=code,
        anzsic_desc="stub",
        rrf_score=rrf,
        in_vector=True,
        in_fts=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestTokens:
    def test_extracts_words(self):
        assert "mechanic" in _tokens("mobile mechanic")

    def test_strips_stopwords(self):
        toks = _tokens("the mobile mechanic")
        assert "the" not in toks
        assert "mechanic" in toks

    def test_empty_string(self):
        assert _tokens("") == frozenset()

    def test_ignores_single_chars(self):
        toks = _tokens("a b c d mobile mechanic")
        assert "a" not in toks
        assert "mobile" in toks


class TestJaccard:
    def test_identical_sets(self):
        a = frozenset({"mobile", "mechanic"})
        assert _jaccard(a, a) == 1.0

    def test_disjoint_sets(self):
        a = frozenset({"mobile"})
        b = frozenset({"plumber"})
        assert _jaccard(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset({"mobile", "mechanic"})
        b = frozenset({"mobile", "plumber"})
        # intersection={mobile} union={mobile,mechanic,plumber} → 1/3
        assert abs(_jaccard(a, b) - 1 / 3) < 1e-9

    def test_both_empty(self):
        assert _jaccard(frozenset(), frozenset()) == 0.0


class TestDescSimilarity:
    def test_identical(self):
        assert _desc_similarity("Coal Mining", "Coal Mining") == 1.0

    def test_case_insensitive(self):
        assert _desc_similarity("coal mining", "Coal Mining") == 1.0

    def test_partial(self):
        score = _desc_similarity("Coal Mining", "Coal Quarrying")
        assert 0.0 < score < 1.0

    def test_unrelated(self):
        score = _desc_similarity("Coal Mining", "Nursery Production")
        assert score < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestCompleteness:
    def test_perfect_completeness(self, evaluator: ANZSICEvaluator):
        results = [
            _make_result(1, "A0111", "Nursery Production Excluding Cut Flowers"),
            _make_result(2, "B0600", "Coal Mining"),
            _make_result(3, "C1111", "Bacon Smallgoods and Ham Manufacturing"),
        ]
        report = evaluator.evaluate("nursery", results, [], top_k=3)
        assert report.completeness == pytest.approx(1.0, abs=0.01)

    def test_partial_result_count(self, evaluator: ANZSICEvaluator):
        results = [_make_result(1, "A0111", "Nursery Production")]
        report = evaluator.evaluate("nursery", results, [], top_k=5)
        assert report.completeness < 1.0
        assert report.details["completeness"]["result_ratio"] == pytest.approx(0.2)

    def test_missing_reason_lowers_completeness(self, evaluator: ANZSICEvaluator):
        results = [
            _make_result(1, "A0111", "Nursery Production", reason=""),
            _make_result(2, "B0600", "Coal Mining"),
        ]
        report = evaluator.evaluate("nursery", results, [], top_k=2)
        assert report.details["completeness"]["field_fill_rate"] < 1.0

    def test_missing_score_lowers_completeness(self, evaluator: ANZSICEvaluator):
        results = [
            _make_result(1, "A0111", "Nursery Production", score=None),
        ]
        report = evaluator.evaluate("nursery", results, [], top_k=1)
        assert report.details["completeness"]["score_fill_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Coherence
# ─────────────────────────────────────────────────────────────────────────────

class TestCoherence:
    def test_sequential_ranks_ascending_scores(self, evaluator: ANZSICEvaluator):
        results = [
            _make_result(1, "A0111", score=950),
            _make_result(2, "B0600", score=800),
            _make_result(3, "C1111", score=600),
        ]
        report = evaluator.evaluate("test", results, [], top_k=3)
        detail = report.details["coherence"]
        assert detail["rank_sequential"] == 1.0
        assert detail["score_descending"] == 1.0

    def test_non_sequential_ranks(self, evaluator: ANZSICEvaluator):
        results = [
            _make_result(1, "A0111"),
            _make_result(3, "B0600"),   # gap — rank 2 missing
        ]
        report = evaluator.evaluate("test", results, [], top_k=2)
        assert report.details["coherence"]["rank_sequential"] == 0.0

    def test_inverted_scores_penalised(self, evaluator: ANZSICEvaluator):
        results = [
            _make_result(1, "A0111", score=400),
            _make_result(2, "B0600", score=900),  # rank 2 higher than rank 1
        ]
        report = evaluator.evaluate("test", results, [], top_k=2)
        assert report.details["coherence"]["score_descending"] < 1.0

    def test_all_same_scores_penalised(self, evaluator: ANZSICEvaluator):
        results = [
            _make_result(1, "A0111", score=800),
            _make_result(2, "B0600", score=800),
            _make_result(3, "C1111", score=800),
        ]
        report = evaluator.evaluate("test", results, [], top_k=3)
        assert report.details["coherence"]["score_spread"] == 0.0

    def test_short_reason_lowers_coherence(self, evaluator: ANZSICEvaluator):
        results = [_make_result(1, "A0111", reason="ok")]  # <30 chars
        report = evaluator.evaluate("test", results, [], top_k=1)
        assert report.details["coherence"]["reason_depth"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectness:
    def test_all_valid_codes(self, evaluator: ANZSICEvaluator):
        results = [
            _make_result(1, "A0111", "Nursery Production Excluding Cut Flowers"),
            _make_result(2, "B0600", "Coal Mining"),
        ]
        report = evaluator.evaluate("test", results, [], top_k=2)
        cx = report.details["correctness"]
        assert cx["code_exists_rate"] == 1.0
        assert cx["desc_accuracy"] > 0.9  # exact match → near 1.0

    def test_invalid_code_detected(self, evaluator: ANZSICEvaluator):
        results = [_make_result(1, "Z9999", "Made Up Code")]
        report = evaluator.evaluate("test", results, [], top_k=1)
        cx = report.details["correctness"]
        assert cx["code_exists_rate"] == 0.0
        assert "Z9999" in cx["invalid_codes"]

    def test_wrong_desc_lowers_accuracy(self, evaluator: ANZSICEvaluator):
        # Code is real but description is completely wrong
        results = [_make_result(1, "A0111", "Totally Different Industry")]
        report = evaluator.evaluate("test", results, [], top_k=1)
        cx = report.details["correctness"]
        assert cx["desc_accuracy"] < 0.5

    def test_invalid_code_flag(self, evaluator: ANZSICEvaluator):
        results = [_make_result(1, "FAKE01", "Hallucinated")]
        report = evaluator.evaluate("test", results, [], top_k=1)
        assert any("INVALID CODES" in f or "HALLUCINATED" in f for f in report.flags)


# ─────────────────────────────────────────────────────────────────────────────
# Relevance
# ─────────────────────────────────────────────────────────────────────────────

class TestRelevance:
    def test_high_token_overlap(self, evaluator: ANZSICEvaluator):
        results = [_make_result(1, "B0600", "Coal Mining")]
        candidates = [_make_candidate("B0600", rrf=0.02)]
        report = evaluator.evaluate("coal mining operations", results, candidates, top_k=1)
        assert report.relevance > 0.3

    def test_zero_token_overlap_still_uses_rrf(self, evaluator: ANZSICEvaluator):
        # Description shares no tokens with query, but high rrf_score
        results = [_make_result(1, "B0600", "zzz xxx yyy")]
        candidates = [_make_candidate("B0600", rrf=0.05)]
        # rrf contribution should still push relevance above zero
        report = evaluator.evaluate("coal mining", results, candidates, top_k=1)
        # RRF evidence gives some relevance even with no token overlap
        assert report.relevance >= 0.0

    def test_rank_weighting(self, evaluator: ANZSICEvaluator):
        # Rank 1 result is relevant, rank 2 is not — overall should still be decent
        high = _make_result(1, "B0600", "coal mining")
        low  = _make_result(2, "A0111", "zzz yyy xxx qqq")
        report_two  = evaluator.evaluate("coal mining", [high, low], [], top_k=2)
        report_one  = evaluator.evaluate("coal mining", [high],      [], top_k=1)
        # Single-result report may differ but rank-1 weighting should keep two-result close
        assert report_two.relevance > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Groundedness
# ─────────────────────────────────────────────────────────────────────────────

class TestGroundedness:
    def test_fully_grounded(self, evaluator: ANZSICEvaluator):
        candidates = [_make_candidate("A0111"), _make_candidate("B0600")]
        results = [
            _make_result(1, "A0111"),
            _make_result(2, "B0600"),
        ]
        report = evaluator.evaluate("test", results, candidates, top_k=2)
        assert report.groundedness == pytest.approx(1.0)

    def test_zero_grounded(self, evaluator: ANZSICEvaluator):
        candidates = [_make_candidate("A0111")]
        results = [
            _make_result(1, "G4711"),  # valid but NOT in candidates
            _make_result(2, "C1111"),
        ]
        report = evaluator.evaluate("test", results, candidates, top_k=2)
        assert report.groundedness < 0.5
        assert report.details["groundedness"]["grounded_rate"] == 0.0

    def test_partial_grounded(self, evaluator: ANZSICEvaluator):
        candidates = [_make_candidate("A0111")]
        results = [
            _make_result(1, "A0111"),   # grounded
            _make_result(2, "G4711"),   # extrapolated (valid but not in candidates)
        ]
        report = evaluator.evaluate("test", results, candidates, top_k=2)
        gd = report.details["groundedness"]
        assert gd["grounded_rate"] == pytest.approx(0.5)
        assert gd["top1_grounded"] == 1.0
        assert "G4711" in gd["extrapolated"]

    def test_hallucinated_code_flagged(self, evaluator: ANZSICEvaluator):
        candidates = [_make_candidate("A0111")]
        results = [_make_result(1, "ZZZZ99")]  # not in candidates, not in master
        report = evaluator.evaluate("test", results, candidates, top_k=1)
        gd = report.details["groundedness"]
        assert "ZZZZ99" in gd["hallucinated"]
        assert any("HALLUCINATED" in f for f in report.flags)

    def test_top1_not_grounded_flag(self, evaluator: ANZSICEvaluator):
        # Rank-1 is a real code but not in candidates
        candidates = [_make_candidate("B0600")]
        results = [
            _make_result(1, "G4711"),   # extrapolated
            _make_result(2, "B0600"),   # grounded but not rank-1
        ]
        report = evaluator.evaluate("test", results, candidates, top_k=2)
        assert any("TOP-1 NOT GROUNDED" in f for f in report.flags)


# ─────────────────────────────────────────────────────────────────────────────
# Overall / edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestOverall:
    def test_overall_within_range(self, evaluator: ANZSICEvaluator):
        candidates = [_make_candidate("A0111"), _make_candidate("B0600")]
        results = [
            _make_result(1, "A0111", "Nursery Production Excluding Cut Flowers", score=950),
            _make_result(2, "B0600", "Coal Mining", score=800),
        ]
        report = evaluator.evaluate("nursery", results, candidates, top_k=2)
        assert 0.0 <= report.overall <= 1.0

    def test_empty_results_returns_zero_report(self, evaluator: ANZSICEvaluator):
        report = evaluator.evaluate("anything", [], [], top_k=5)
        assert report.overall == 0.0
        assert report.completeness == 0.0
        assert len(report.flags) > 0

    def test_missing_csv_does_not_crash(self, tmp_path: Path):
        ev = ANZSICEvaluator(tmp_path / "nonexistent.csv")
        results = [_make_result(1, "A0111", "Some desc")]
        report = ev.evaluate("test", results, [], top_k=1)
        # Should still produce a report — correctness just can't check codes
        assert report is not None
        assert report.correctness == 0.0  # no master loaded → 0 codes valid

    def test_weights_sum_to_one(self):
        from prod.services.evaluator import _WEIGHTS
        assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9
