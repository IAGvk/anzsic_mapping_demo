"""
services/retriever.py
──────────────────────────────────────────────────────────────────────────────
Stage 1 of the classification pipeline: Hybrid Retrieval with RRF Fusion.

Architecture:
  • Accepts any EmbeddingPort and DatabasePort via constructor injection.
  • _compute_rrf() is a pure Python function — no I/O, easily unit-tested.
  • HybridRetriever.retrieve() is the single public entry point.

Reciprocal Rank Fusion formula:
  score(d) = Σᵢ  1 / (k + rankᵢ(d))

Where k = RRF_K (default 60) and rankᵢ(d) is the rank of document d
in system i.  Documents that appear in both systems receive a combined score.

References:
  Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009). Reciprocal Rank
  Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from prod.config.settings import Settings
from prod.domain.exceptions import EmbeddingError, RetrievalError
from prod.domain.models import Candidate
from prod.ports.database_port import DatabasePort
from prod.ports.embedding_port import EmbeddingPort

logger = logging.getLogger(__name__)


# ── RRF intermediate result ────────────────────────────────────────────────

@dataclass(frozen=True)
class _RRFResult:
    """Holds the fused score and provenance for a single ANZSIC code."""

    anzsic_code: str
    rrf_score: float
    in_vector: bool
    in_fts: bool
    vector_rank: int | None
    fts_rank: int | None


# ── Service class ──────────────────────────────────────────────────────────

class HybridRetriever:
    """Hybrid retrieval using vector ANN search + FTS fused via RRF.

    To change retrieval behaviour, swap the injected DatabasePort or
    EmbeddingPort — no code changes here.

    Args:
        db:       Any object satisfying DatabasePort.
        embedder: Any object satisfying EmbeddingPort.
        settings: Shared application settings.
    """

    def __init__(
        self,
        db: DatabasePort,
        embedder: EmbeddingPort,
        settings: Settings,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._rrf_k = settings.rrf_k
        logger.debug(
            "HybridRetriever init | embed_model=%s rrf_k=%d",
            embedder.model_name,
            self._rrf_k,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, n: int) -> list[Candidate]:
        """Run hybrid retrieval for a query.

        Workflow:
          1. Embed query  (RETRIEVAL_QUERY task type)
          2. Vector ANN search → top-n (code, rank) pairs
          3. FTS search        → top-n (code, rank) pairs
          4. RRF fusion        → merged, scored list (pure Python)
          5. Fetch full DB records for top-n fused codes
          6. Assemble and return Candidate objects

        Args:
            query: Natural-language query.
            n:     Maximum number of candidates to return.

        Returns:
            List of Candidate objects sorted by RRF score descending.

        Raises:
            EmbeddingError: If the embedding call fails.
            RetrievalError: If candidate assembly fails.
        """
        logger.info("Retrieving candidates | query=%r n=%d", query[:80], n)

        # ── 1. Embed ───────────────────────────────────────────────────────
        query_vec = self._embedder.embed_query(query)
        if not query_vec:
            raise EmbeddingError(f"embed_query returned empty result for: {query!r}")

        # ── 2 & 3. Search ─────────────────────────────────────────────────
        vec_hits = self._db.vector_search(query_vec, limit=n)
        fts_hits = self._db.fts_search(query, limit=n)
        logger.debug("vec_hits=%d  fts_hits=%d", len(vec_hits), len(fts_hits))

        # ── 4. RRF fusion (pure Python — no I/O) ──────────────────────────
        rrf_results = compute_rrf(vec_hits, fts_hits, k=self._rrf_k)
        top_rrf = sorted(rrf_results, key=lambda r: r.rrf_score, reverse=True)[:n]

        # ── 5. Fetch full records ──────────────────────────────────────────
        top_codes = [r.anzsic_code for r in top_rrf]
        records = self._db.fetch_by_codes(top_codes)

        # ── 6. Assemble Candidate objects ──────────────────────────────────
        candidates: list[Candidate] = []
        for rrf in top_rrf:
            rec = records.get(rrf.anzsic_code)
            if rec is None:
                logger.warning(
                    "Code %s in RRF results but missing from fetch_by_codes",
                    rrf.anzsic_code,
                )
                continue
            candidates.append(
                Candidate(
                    **rec,
                    rrf_score=round(rrf.rrf_score, 6),
                    in_vector=rrf.in_vector,
                    in_fts=rrf.in_fts,
                    vector_rank=rrf.vector_rank,
                    fts_rank=rrf.fts_rank,
                )
            )

        top_score = candidates[0].rrf_score if candidates else 0.0
        logger.info(
            "Retrieval complete | candidates=%d top_rrf=%.6f",
            len(candidates),
            top_score,
        )
        return candidates


# ── Pure function: RRF fusion ──────────────────────────────────────────────
# Extracted as a module-level function so unit tests can call it directly
# without instantiating HybridRetriever or any adapter.

def compute_rrf(
    vec_hits: list[tuple[str, int]],
    fts_hits: list[tuple[str, int]],
    k: int = 60,
) -> list[_RRFResult]:
    """Compute Reciprocal Rank Fusion scores for two ranked lists.

    This is a *pure function* — deterministic, no side-effects, no I/O.
    It is the ideal target for unit tests.

    Args:
        vec_hits: List of (anzsic_code, rank) from vector search.
                  rank is 1-indexed, 1 = best match.
        fts_hits: List of (anzsic_code, rank) from full-text search.
        k:        RRF smoothing constant (standard value = 60).

    Returns:
        List of _RRFResult objects with combined scores.
        Not sorted — caller decides sort order.

    Examples:
        >>> results = compute_rrf([("A", 1), ("B", 2)], [("A", 2), ("C", 1)])
        >>> scores = {r.anzsic_code: r.rrf_score for r in results}
        >>> scores["A"] > scores["B"]   # A appears in both systems
        True
        >>> scores["A"] > scores["C"]   # A's combined score beats FTS-only C
        True
    """
    vec_map: dict[str, int] = dict(vec_hits)
    fts_map: dict[str, int] = dict(fts_hits)
    all_codes = set(vec_map) | set(fts_map)

    results: list[_RRFResult] = []
    for code in all_codes:
        score = 0.0
        v_rank = vec_map.get(code)
        f_rank = fts_map.get(code)
        if v_rank is not None:
            score += 1.0 / (k + v_rank)
        if f_rank is not None:
            score += 1.0 / (k + f_rank)
        results.append(
            _RRFResult(
                anzsic_code=code,
                rrf_score=score,
                in_vector=v_rank is not None,
                in_fts=f_rank is not None,
                vector_rank=v_rank,
                fts_rank=f_rank,
            )
        )
    return results
