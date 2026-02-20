"""
services/classifier.py
──────────────────────────────────────────────────────────────────────────────
Pipeline orchestrator: wires Stage 1 (retrieval) + Stage 2 (reranking) into
a single classify(SearchRequest) → ClassifyResponse call.

This is the primary entry point for all interfaces (CLI, Streamlit, future API).
It knows nothing about infrastructure — it only speaks in domain objects.

SearchMode.FAST:
  Stage 1 only. Returns retrieval results sorted by RRF score, formatted as
  ClassifyResult objects. No LLM call. Suitable for interactive exploration.

SearchMode.HIGH_FIDELITY:
  Stage 1 + Stage 2. RRF candidates are re-ranked by the LLM, which adds a
  natural-language reason for each match. Recommended for production.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from prod.config.settings import Settings
from prod.domain.models import (
    Candidate,
    ClassifyResponse,
    ClassifyResult,
    SearchMode,
    SearchRequest,
)
from prod.services.retriever import HybridRetriever
from prod.services.reranker import LLMReranker

logger = logging.getLogger(__name__)


class ClassifierPipeline:
    """Two-stage ANZSIC classification pipeline.

    Inject via services/container.py — do not instantiate directly in
    application code.

    Args:
        retriever: HybridRetriever (Stage 1).
        reranker:  LLMReranker (Stage 2).
        settings:  Shared application settings.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: LLMReranker,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._settings = settings

    # ── Public API ─────────────────────────────────────────────────────────

    def classify(self, request: SearchRequest) -> ClassifyResponse:
        """Classify an occupation/business description into ANZSIC codes.

        Args:
            request: Validated SearchRequest (query, mode, top_k, retrieval_n).

        Returns:
            ClassifyResponse with ranked results and metadata.
        """
        logger.info(
            "classify | query=%r mode=%s top_k=%d retrieval_n=%d",
            request.query[:80],
            request.mode.value,
            request.top_k,
            request.retrieval_n,
        )

        # ── Stage 1: Hybrid Retrieval ──────────────────────────────────────
        candidates = self._retriever.retrieve(
            query=request.query,
            n=request.retrieval_n,
        )

        # ── Stage 2 (optional): LLM Re-ranking ────────────────────────────
        if request.mode == SearchMode.HIGH_FIDELITY:
            results = self._reranker.rerank(
                query=request.query,
                candidates=candidates,
                top_k=request.top_k,
            )
            llm_model = self._reranker._llm.model_name
        else:
            # FAST mode: convert top-k candidates directly to ClassifyResult
            results = [
                _candidate_to_result(c, rank=i + 1)
                for i, c in enumerate(candidates[: request.top_k])
            ]
            llm_model = ""

        return ClassifyResponse(
            query=request.query,
            mode=request.mode.value,
            results=results,
            candidates_retrieved=len(candidates),
            generated_at=datetime.now(tz=timezone.utc),
            embed_model=self._retriever._embedder.model_name,
            llm_model=llm_model,
        )


# ── Helper ─────────────────────────────────────────────────────────────────

def _candidate_to_result(candidate: Candidate, rank: int) -> ClassifyResult:
    """Convert a Stage 1 Candidate to a ClassifyResult for FAST mode."""
    return ClassifyResult(
        rank=rank,
        anzsic_code=candidate.anzsic_code,
        anzsic_desc=candidate.anzsic_desc,
        class_desc=candidate.class_desc,
        division_desc=candidate.division_desc,
        reason=f"RRF score: {candidate.rrf_score:.6f} "
               f"(vector={'✓' if candidate.in_vector else '✗'}, "
               f"fts={'✓' if candidate.in_fts else '✗'})",
    )
