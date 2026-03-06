"""
adapters/null_embedding.py
──────────────────────────────────────────────────────────────────────────────
A no-op EmbeddingPort adapter that signals "skip vector search".

Use case:
  When the embedding service is unavailable (e.g. Vertex AI blocked by
  corporate firewall), set EMBED_PROVIDER=none to fall back to FTS-only
  retrieval.  HybridRetriever detects the empty vector and skips the
  pgvector ANN step, running only the full-text search leg.

  Stage 2 (GENI / Gemini reranking) is unaffected — it operates on the
  FTS candidates as normal.

Pipeline behaviour with EMBED_PROVIDER=none:
  Stage 1 → FTS search only (no vector ANN)  ← degraded but functional
  Stage 2 → LLM reranking unchanged           ← full quality
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class NullEmbeddingAdapter:
    """EmbeddingPort implementation that disables vector search.

    Returns an empty list from embed_query() — HybridRetriever treats
    this as a signal to skip the vector search leg entirely.

    Injected via services/container.py when EMBED_PROVIDER=none.
    No other file in the codebase references this class.
    """

    model_name = "none (FTS-only mode)"
    dimensions = 0

    def __init__(self) -> None:
        logger.warning(
            "NullEmbeddingAdapter active — vector search is DISABLED. "
            "Retrieval will use FTS only. Set EMBED_PROVIDER=vertex or "
            "EMBED_PROVIDER=openai to re-enable hybrid search."
        )

    def embed_query(self, text: str) -> list[float]:
        """Return empty list — signals HybridRetriever to skip vector search."""
        return []

    def embed_document(self, text: str, title: str = "") -> list[float]:
        """Not called in FTS-only mode; returns empty list defensively."""
        return []

    def embed_documents_batch(
        self,
        texts: list[str],
        titles: list[str] | None = None,
    ) -> list[list[float] | None]:
        """Not called in FTS-only mode; returns list of None defensively."""
        return [None for _ in texts]
