"""
services/container.py
──────────────────────────────────────────────────────────────────────────────
Dependency Injection container.

THIS IS THE ONLY FILE THAT NAMES CONCRETE ADAPTER CLASSES.

Provider selection is driven entirely by environment variables — no code
changes are needed to switch between providers:

  EMBED_PROVIDER=vertex  (default) → VertexEmbeddingAdapter
  EMBED_PROVIDER=openai            → OpenAIEmbeddingAdapter

  LLM_PROVIDER=vertex   (default) → GeminiLLMAdapter
  LLM_PROVIDER=openai             → OpenAILLMAdapter

Mix-and-match is supported (e.g. OpenAI embeddings + Gemini LLM).
GCPAuthManager is only instantiated when at least one GCP adapter is used.

Replace the database:
  - from prod.adapters.postgres_db import PostgresDatabaseAdapter
  + from prod.adapters.weaviate_db import WeaviateDatabaseAdapter

Thread safety:
  @lru_cache(maxsize=1) makes get_pipeline() return the same instance across
  calls.  For Streamlit this is fine (single process, one pipeline per run).
  For FastAPI with multiple workers, each worker process gets its own pipeline
  instance (one per process — correct behaviour for psycopg2 connections).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from prod.adapters.postgres_db import PostgresDatabaseAdapter
from prod.config.settings import get_settings
from prod.domain.exceptions import ConfigurationError
from prod.ports.embedding_port import EmbeddingPort
from prod.ports.llm_port import LLMPort
from prod.services.classifier import ClassifierPipeline
from prod.services.reranker import LLMReranker
from prod.services.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def _build_embedder(settings) -> EmbeddingPort:
    """Instantiate the correct EmbeddingPort adapter based on EMBED_PROVIDER."""
    provider = settings.embed_provider.lower()
    if provider == "openai":
        from prod.adapters.openai_embedding import OpenAIEmbeddingAdapter
        logger.info("Embedding provider: OpenAI (%s)", settings.openai_embed_model)
        return OpenAIEmbeddingAdapter(settings)
    if provider == "vertex":
        from prod.adapters.gcp_auth import GCPAuthManager
        from prod.adapters.vertex_embedding import VertexEmbeddingAdapter
        logger.info("Embedding provider: Vertex AI (%s)", settings.gcp_embed_model)
        auth = GCPAuthManager(settings)
        return VertexEmbeddingAdapter(auth, settings)
    raise ConfigurationError(
        f"Unknown EMBED_PROVIDER '{settings.embed_provider}'. "
        "Valid values: 'vertex', 'openai'."
    )


def _build_llm(settings) -> LLMPort:
    """Instantiate the correct LLMPort adapter based on LLM_PROVIDER."""
    provider = settings.llm_provider.lower()
    if provider == "openai":
        from prod.adapters.openai_llm import OpenAILLMAdapter
        logger.info("LLM provider: OpenAI (%s)", settings.openai_llm_model)
        return OpenAILLMAdapter(settings)
    if provider == "vertex":
        from prod.adapters.gcp_auth import GCPAuthManager
        from prod.adapters.gemini_llm import GeminiLLMAdapter
        logger.info("LLM provider: Vertex AI Gemini (%s)", settings.gcp_gemini_model)
        auth = GCPAuthManager(settings)
        return GeminiLLMAdapter(auth, settings)
    raise ConfigurationError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
        "Valid values: 'vertex', 'openai'."
    )


@lru_cache(maxsize=1)
def get_pipeline() -> ClassifierPipeline:
    """Build and return the fully wired ClassifierPipeline singleton.

    Provider selection is read from ``EMBED_PROVIDER`` and ``LLM_PROVIDER``
    environment variables.  The ``@lru_cache`` ensures this runs only once
    per process lifetime.

    Returns:
        Fully initialised ClassifierPipeline ready for use.

    Raises:
        ConfigurationError: If an unknown provider name is given.
        AuthenticationError: If required API keys / credentials are missing.
    """
    settings = get_settings()
    logger.info(
        "Building ClassifierPipeline | embed_provider=%s llm_provider=%s",
        settings.embed_provider,
        settings.llm_provider,
    )

    # ── Infrastructure adapters (provider-selected) ────────────────────────
    embedder = _build_embedder(settings)   # EmbeddingPort
    llm      = _build_llm(settings)        # LLMPort
    db       = PostgresDatabaseAdapter(settings)   # DatabasePort

    # ── Services (receive only Port interfaces, not concrete types) ────────
    retriever = HybridRetriever(db=db, embedder=embedder, settings=settings)
    reranker  = LLMReranker(llm=llm, settings=settings)

    pipeline = ClassifierPipeline(
        retriever=retriever,
        reranker=reranker,
        settings=settings,
    )

    logger.info(
        "ClassifierPipeline ready | embedder=%s llm=%s",
        embedder.model_name,
        llm.model_name,
    )
    return pipeline
