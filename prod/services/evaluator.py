"""
services/evaluator.py
──────────────────────────────────────────────────────────────────────────────
Programmatic quality evaluation for ANZSIC classify() responses.

Five dimensions — no external LLM, no network calls, pure CPU:

  completeness  — structural completeness of the LLM response
  coherence     — internal ordering / score logic consistency
  correctness   — codes and descriptions match master CSV ground truth
  relevance     — top results are close to the query (lexical + RRF evidence)
  groundedness  — LLM codes were drawn from the Stage-1 candidate pool

How each dimension is measured
──────────────────────────────
COMPLETENESS
  result_ratio      : len(results) / top_k  (capped at 1.0)
  field_fill_rate   : fraction of results where anzsic_code, anzsic_desc,
                      reason are all non-empty strings
  score_fill_rate   : fraction of results that carry an integer LLM score
  → weighted mean of the three signals

COHERENCE
  rank_sequential   : 1.0 if ranks are exactly [1, 2, 3, ...]
  score_descending  : fraction of consecutive rank pairs where score[i] ≥ score[i+1]
  score_spread      : 1.0 when stdev(scores) > 0, else 0.0  (penalises all-same)
  reason_depth      : fraction of results whose reason is ≥ 30 chars
  → mean of applicable checks

CORRECTNESS
  code_exists_rate  : fraction of returned codes present in anzsic_master.csv
  desc_accuracy     : SequenceMatcher ratio between returned anzsic_desc and
                      master anzsic_desc, averaged over existing codes
  → 0.5 × code_exists_rate + 0.5 × desc_accuracy

RELEVANCE
  token_overlap     : Jaccard similarity (query tokens ∩ result tokens) /
                      (query tokens ∪ result tokens), rank-weighted
                      (tokens = lowercase word stems, stopwords stripped)
  rrf_evidence      : normalised RRF score from Stage-1 for results that were
                      in the candidate pool (0 for extrapolated/hallucinated)
  → 0.6 × token_overlap + 0.4 × rrf_evidence

GROUNDEDNESS
  grounded_rate     : fraction of results whose code appeared in Stage-1 candidates
  top1_grounded     : 1.0 if rank-1 result was in candidates, 0.0 otherwise
  → 0.7 × grounded_rate + 0.3 × top1_grounded

OVERALL = weighted mean
  groundedness  0.30  (hallucination is the most critical failure)
  correctness   0.25
  relevance     0.20
  completeness  0.15
  coherence     0.10
"""
from __future__ import annotations

import csv
import logging
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from prod.domain.models import Candidate, ClassifyResult, EvaluationReport

logger = logging.getLogger(__name__)

# ── Stopwords (stripped before token overlap) ──────────────────────────────
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "of", "to", "for",
    "with", "on", "at", "by", "from", "as", "is", "are", "was",
    "be", "been", "not", "no", "nor",
})

# ── Dimension weights for overall score ────────────────────────────────────
_WEIGHTS: dict[str, float] = {
    "groundedness":  0.30,
    "correctness":   0.25,
    "relevance":     0.20,
    "completeness":  0.15,
    "coherence":     0.10,
}

# ── Flag thresholds ────────────────────────────────────────────────────────
_FLAG_COMPLETENESS_LOW  = 0.5
_FLAG_RELEVANCE_LOW     = 0.10
_FLAG_REASON_MIN_CHARS  = 30


# ── Pure helper functions (importable for unit testing) ────────────────────

def _tokens(text: str) -> frozenset[str]:
    """Lowercase word tokens from *text*, stopwords removed."""
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity between two token sets.  Returns 0.0 when both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _desc_similarity(returned: str, master: str) -> float:
    """SequenceMatcher ratio between two description strings (case-insensitive)."""
    return SequenceMatcher(None, returned.lower().strip(), master.lower().strip()).ratio()


def _rank_weight(rank: int) -> float:
    """Inverse-rank weight: rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33, …"""
    return 1.0 / rank


# ── Main evaluator class ────────────────────────────────────────────────────

class ANZSICEvaluator:
    """Computes programmatic quality scores for a classify() response.

    One instance per pipeline — the master CSV is loaded once at construction.

    Args:
        master_csv_path: Path to anzsic_master.csv (used for correctness checks).
    """

    def __init__(self, master_csv_path: Path) -> None:
        self._master: dict[str, str] = {}   # code → anzsic_desc (lowercase)
        self._load_master(master_csv_path)
        logger.debug(
            "ANZSICEvaluator: loaded %d codes from %s",
            len(self._master),
            master_csv_path,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        results: list[ClassifyResult],
        candidates: list[Candidate],
        top_k: int,
    ) -> EvaluationReport:
        """Compute all five evaluation dimensions for a single classify() call.

        Args:
            query:      Original search query.
            results:    Stage-2 LLM outputs (or Stage-1 in FAST mode).
            candidates: Stage-1 retrieval candidates (before LLM ranking).
            top_k:      Requested number of results.

        Returns:
            EvaluationReport with per-dimension scores, details, and flags.
        """
        if not results:
            return self._empty_report("LLM returned zero results")

        completeness,  c_detail  = self._completeness(results, top_k)
        coherence,     co_detail = self._coherence(results)
        correctness,   cx_detail = self._correctness(results)
        relevance,     r_detail  = self._relevance(query, results, candidates)
        groundedness,  g_detail  = self._groundedness(results, candidates)

        overall = sum(
            _WEIGHTS[dim] * score
            for dim, score in {
                "completeness": completeness,
                "coherence":    coherence,
                "correctness":  correctness,
                "relevance":    relevance,
                "groundedness": groundedness,
            }.items()
        )

        flags = self._flags(
            completeness=completeness,
            relevance=relevance,
            results=results,
            candidates=candidates,
            c_detail=c_detail,
            cx_detail=cx_detail,
            co_detail=co_detail,
            g_detail=g_detail,
        )

        return EvaluationReport(
            completeness=round(completeness, 4),
            coherence=round(coherence, 4),
            correctness=round(correctness, 4),
            relevance=round(relevance, 4),
            groundedness=round(groundedness, 4),
            overall=round(overall, 4),
            details={
                "completeness":  c_detail,
                "coherence":     co_detail,
                "correctness":   cx_detail,
                "relevance":     r_detail,
                "groundedness":  g_detail,
            },
            flags=flags,
        )

    # ── Completeness ───────────────────────────────────────────────────────

    def _completeness(
        self, results: list[ClassifyResult], top_k: int
    ) -> tuple[float, dict]:
        result_ratio = min(len(results) / max(top_k, 1), 1.0)

        filled, scored = 0, 0
        for r in results:
            has_code  = bool(r.anzsic_code and r.anzsic_code.strip())
            has_desc  = bool(r.anzsic_desc and r.anzsic_desc.strip())
            has_reason = bool(r.reason and len(r.reason.strip()) >= 1)
            if has_code and has_desc and has_reason:
                filled += 1
            if r.score is not None:
                scored += 1

        field_fill_rate = filled / len(results)
        score_fill_rate = scored / len(results)

        score = 0.40 * result_ratio + 0.40 * field_fill_rate + 0.20 * score_fill_rate
        return score, {
            "result_ratio":     round(result_ratio, 3),
            "field_fill_rate":  round(field_fill_rate, 3),
            "score_fill_rate":  round(score_fill_rate, 3),
            "n_results":        len(results),
            "top_k_requested":  top_k,
        }

    # ── Coherence ──────────────────────────────────────────────────────────

    def _coherence(self, results: list[ClassifyResult]) -> tuple[float, dict]:
        # 1. Ranks are sequential 1, 2, 3, …
        expected_ranks = list(range(1, len(results) + 1))
        actual_ranks   = [r.rank for r in results]
        rank_sequential = 1.0 if actual_ranks == expected_ranks else 0.0

        # 2. Confidence scores are non-increasing (rank 1 ≥ rank 2 ≥ …)
        scores_raw = [r.score for r in results if r.score is not None]
        if len(scores_raw) >= 2:
            pairs_ok = sum(
                1 for a, b in zip(scores_raw, scores_raw[1:]) if a >= b
            )
            score_descending = pairs_ok / (len(scores_raw) - 1)
        else:
            score_descending = 1.0  # can't check — give benefit of the doubt

        # 3. Score spread (penalise all-same confidence)
        if len(scores_raw) >= 2:
            score_spread = 1.0 if statistics.stdev(scores_raw) > 0 else 0.0
        else:
            score_spread = 1.0  # only one result — can't assess spread

        # 4. Reason depth
        reasons = [r.reason or "" for r in results]
        reason_depth = sum(
            1 for reason in reasons if len(reason.strip()) >= _FLAG_REASON_MIN_CHARS
        ) / len(results)

        score = (rank_sequential + score_descending + score_spread + reason_depth) / 4
        return score, {
            "rank_sequential":   rank_sequential,
            "score_descending":  round(score_descending, 3),
            "score_spread":      score_spread,
            "reason_depth":      round(reason_depth, 3),
            "scores":            scores_raw,
        }

    # ── Correctness ────────────────────────────────────────────────────────

    def _correctness(self, results: list[ClassifyResult]) -> tuple[float, dict]:
        valid_codes: list[str] = []
        invalid_codes: list[str] = []
        desc_sims: list[float] = []

        for r in results:
            code = r.anzsic_code
            if code in self._master:
                valid_codes.append(code)
                sim = _desc_similarity(r.anzsic_desc, self._master[code])
                desc_sims.append(sim)
            else:
                invalid_codes.append(code)

        code_exists_rate = len(valid_codes) / len(results)
        desc_accuracy    = (sum(desc_sims) / len(desc_sims)) if desc_sims else 0.0

        score = 0.5 * code_exists_rate + 0.5 * desc_accuracy
        return score, {
            "code_exists_rate": round(code_exists_rate, 3),
            "desc_accuracy":    round(desc_accuracy, 3),
            "valid_codes":      valid_codes,
            "invalid_codes":    invalid_codes,
            "per_code_sim":     {
                r.anzsic_code: round(_desc_similarity(r.anzsic_desc, self._master[r.anzsic_code]), 3)
                for r in results
                if r.anzsic_code in self._master
            },
        }

    # ── Relevance ──────────────────────────────────────────────────────────

    def _relevance(
        self,
        query: str,
        results: list[ClassifyResult],
        candidates: list[Candidate],
    ) -> tuple[float, dict]:
        query_toks = _tokens(query)

        # Build a lookup: code → rrf_score from Stage-1 candidates
        candidate_rrf: dict[str, float] = {
            c.anzsic_code: c.rrf_score for c in candidates
        }
        max_rrf = max(candidate_rrf.values(), default=0.0) or 1.0
        min_rrf = min(candidate_rrf.values(), default=0.0)
        rrf_range = max_rrf - min_rrf or 1.0  # avoid division by zero

        weighted_jaccard_sum = 0.0
        weighted_rrf_sum     = 0.0
        weight_total         = 0.0
        per_result: list[dict] = []

        for r in results:
            w = _rank_weight(r.rank)
            # Token overlap between query and the result's description
            result_toks = _tokens(r.anzsic_desc or "")
            jaccard = _jaccard(query_toks, result_toks)

            # Normalised RRF score (0 if code wasn't in Stage-1 at all)
            raw_rrf = candidate_rrf.get(r.anzsic_code, min_rrf)
            norm_rrf = (raw_rrf - min_rrf) / rrf_range

            weighted_jaccard_sum += w * jaccard
            weighted_rrf_sum     += w * norm_rrf
            weight_total         += w

            per_result.append({
                "rank":    r.rank,
                "code":    r.anzsic_code,
                "jaccard": round(jaccard, 3),
                "rrf_normalised": round(norm_rrf, 3),
                "in_stage1": r.anzsic_code in candidate_rrf,
            })

        if weight_total == 0:
            return 0.0, {"per_result": per_result}

        avg_jaccard  = weighted_jaccard_sum / weight_total
        avg_rrf      = weighted_rrf_sum     / weight_total
        score        = 0.6 * avg_jaccard + 0.4 * avg_rrf

        return score, {
            "weighted_jaccard": round(avg_jaccard, 3),
            "weighted_rrf":     round(avg_rrf, 3),
            "per_result":       per_result,
        }

    # ── Groundedness ───────────────────────────────────────────────────────

    def _groundedness(
        self,
        results: list[ClassifyResult],
        candidates: list[Candidate],
    ) -> tuple[float, dict]:
        candidate_codes = {c.anzsic_code for c in candidates}

        grounded:     list[str] = []
        extrapolated: list[str] = []   # not in candidates but IS in master CSV
        hallucinated: list[str] = []   # not in candidates AND not in master CSV

        for r in results:
            code = r.anzsic_code
            if code in candidate_codes:
                grounded.append(code)
            elif code in self._master:
                extrapolated.append(code)
            else:
                hallucinated.append(code)

        n = len(results)
        grounded_rate = len(grounded) / n
        top1_grounded = 1.0 if (results[0].anzsic_code in candidate_codes) else 0.0

        score = 0.70 * grounded_rate + 0.30 * top1_grounded
        return score, {
            "grounded_rate":  round(grounded_rate, 3),
            "top1_grounded":  top1_grounded,
            "grounded":       grounded,
            "extrapolated":   extrapolated,
            "hallucinated":   hallucinated,
        }

    # ── Flag generation ────────────────────────────────────────────────────

    def _flags(
        self,
        completeness:  float,
        relevance:     float,
        results:       list[ClassifyResult],
        candidates:    list[Candidate],
        c_detail:      dict,
        cx_detail:     dict,
        co_detail:     dict,
        g_detail:      dict,
    ) -> list[str]:
        flags: list[str] = []

        if completeness < _FLAG_COMPLETENESS_LOW:
            flags.append(
                f"LOW COMPLETENESS: {c_detail['n_results']} of "
                f"{c_detail['top_k_requested']} requested results returned."
            )
        if cx_detail["invalid_codes"]:
            flags.append(
                f"INVALID CODES: {cx_detail['invalid_codes']} not found in master CSV "
                f"({'possible hallucination' if cx_detail['invalid_codes'] else ''})."
            )
        if g_detail["hallucinated"]:
            flags.append(
                f"HALLUCINATED CODES: {g_detail['hallucinated']} — codes returned by LLM "
                "that do not exist in Stage-1 candidates OR master CSV."
            )
        if g_detail["extrapolated"]:
            flags.append(
                f"EXTRAPOLATED CODES: {g_detail['extrapolated']} — valid codes but not in "
                "Stage-1 candidates (LLM sourced from injected CSV reference)."
            )
        if g_detail["top1_grounded"] == 0.0:
            flags.append(
                "TOP-1 NOT GROUNDED: rank-1 result was not in Stage-1 retrieval candidates."
            )
        if co_detail["score_descending"] < 1.0:
            flags.append(
                "INCOHERENT SCORES: LLM confidence scores are not monotonically decreasing."
            )
        if relevance < _FLAG_RELEVANCE_LOW:
            flags.append(
                f"LOW RELEVANCE: top result has very low lexical overlap with query "
                f"(jaccard={co_detail.get('weighted_jaccard', '?')})."
            )

        return flags

    # ── CSV loader ─────────────────────────────────────────────────────────

    def _load_master(self, path: Path) -> None:
        """Load anzsic_master.csv into {code: description} dict."""
        if not path.exists():
            logger.warning("ANZSICEvaluator: master CSV not found at %s — correctness checks disabled", path)
            return
        try:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    code = (row.get("anzsic_code") or row.get("ANZSIC_CODE") or "").strip()
                    desc = (row.get("anzsic_desc") or row.get("ANZSIC_DESC") or "").strip()
                    if code:
                        self._master[code] = desc
        except Exception as exc:
            logger.warning("ANZSICEvaluator: failed to load master CSV: %s", exc)

    # ── Edge-case helper ───────────────────────────────────────────────────

    @staticmethod
    def _empty_report(reason: str) -> EvaluationReport:
        return EvaluationReport(
            completeness=0.0,
            coherence=0.0,
            correctness=0.0,
            relevance=0.0,
            groundedness=0.0,
            overall=0.0,
            details={},
            flags=[f"CANNOT EVALUATE: {reason}"],
        )
