"""
tests/conftest.py
──────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures and mock adapter implementations.

Mock adapters implement the Port Protocols via structural subtyping — they do
NOT inherit from any base class.  pytest uses them to test service logic
without any real GCP or database connections.

Fixture hierarchy:
  mock_embedder  → implements EmbeddingPort (deterministic fake vectors)
  mock_llm       → implements LLMPort (returns pre-baked JSON)
  mock_db        → implements DatabasePort (in-memory fixture data)
  mock_retriever → HybridRetriever wired with mock_embedder + mock_db
  mock_reranker  → LLMReranker wired with mock_llm
  pipeline       → ClassifierPipeline wired with both mocks
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from prod.config.settings import Settings
from prod.domain.models import SearchMode, SearchRequest
from prod.services.classifier import ClassifierPipeline
from prod.services.reranker import LLMReranker
from prod.services.retriever import HybridRetriever


# ── Settings fixture ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def settings() -> Settings:
    """Return a Settings instance with sane test defaults."""
    return Settings(
        gcp_project_id="test-project",
        gcp_location_id="us-central1",
        gcp_embed_model="text-embedding-test",
        gcp_gemini_model="gemini-test",
        gcloud_path="/usr/bin/gcloud",
        https_proxy="",
        db_dsn="dbname=anzsic_db",
        rrf_k=60,
        retrieval_n=10,
        top_k=3,
        embed_dim=8,  # tiny vectors in tests
        embed_batch_size=5,
    )


# ── Mock adapters ──────────────────────────────────────────────────────────

class MockEmbeddingAdapter:
    """Deterministic fake embedder.  Returns fixed 8-dim vectors."""

    model_name = "mock-embedding"
    dimensions = 8

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]

    def embed_document(self, text: str, title: str = "") -> list[float]:
        return [0.2, 0.1, 0.4, 0.3, 0.2, 0.1, 0.4, 0.3]

    def embed_documents_batch(
        self,
        texts: list[str],
        titles: list[str] | None = None,
    ) -> list[list[float] | None]:
        return [self.embed_document(t) for t in texts]


# ── In-memory DB fixture data ──────────────────────────────────────────────

_DB_RECORDS: dict[str, dict] = {
    "P7411_01": {
        "anzsic_code": "P7411_01",
        "anzsic_desc": "Motor Vehicle Parts and Accessories Retailing",
        "class_code": "P7411",
        "class_desc": "Motor Vehicle Parts and Accessories Retailing",
        "group_code": "P741",
        "group_desc": "Motor Vehicle Retailing",
        "subdivision_desc": "Motor Vehicle and Motor Vehicle Parts Retailing",
        "division_desc": "Retail Trade",
        "class_exclusions": "",
        "enriched_text": "Motor Vehicle Parts and Accessories Retailing — Retail Trade",
    },
    "S9419_03": {
        "anzsic_code": "S9419_03",
        "anzsic_desc": "Automotive Repair and Maintenance (own account)",
        "class_code": "S9419",
        "class_desc": "Other Repair and Maintenance",
        "group_code": "S941",
        "group_desc": "Repair and Maintenance",
        "subdivision_desc": "Repair and Maintenance",
        "division_desc": "Other Services",
        "class_exclusions": "",
        "enriched_text": "Automotive Repair and Maintenance — Other Services",
    },
    "S9411_01": {
        "anzsic_code": "S9411_01",
        "anzsic_desc": "Automotive Electrical Services",
        "class_code": "S9411",
        "class_desc": "Automotive Repair and Maintenance",
        "group_code": "S941",
        "group_desc": "Repair and Maintenance",
        "subdivision_desc": "Repair and Maintenance",
        "division_desc": "Other Services",
        "class_exclusions": "",
        "enriched_text": "Automotive Electrical Services — Other Services",
    },
}


class MockDatabaseAdapter:
    """In-memory fake database."""

    # Pretend these codes are ranked by vector similarity
    _VECTOR_ORDER = ["S9419_03", "S9411_01", "P7411_01"]
    # Pretend these codes are ranked by FTS
    _FTS_ORDER = ["P7411_01", "S9419_03"]

    def vector_search(
        self, embedding: list[float], limit: int
    ) -> list[tuple[str, int]]:
        return [(c, i + 1) for i, c in enumerate(self._VECTOR_ORDER[:limit])]

    def fts_search(
        self, query_text: str, limit: int
    ) -> list[tuple[str, int]]:
        return [(c, i + 1) for i, c in enumerate(self._FTS_ORDER[:limit])]

    def fetch_by_codes(self, codes: list[str]) -> dict[str, dict]:
        return {c: _DB_RECORDS[c] for c in codes if c in _DB_RECORDS}


class MockLLMAdapter:
    """Returns a pre-baked JSON re-rank response."""

    model_name = "mock-llm"

    # Build canned response once
    _RESPONSE = json.dumps(
        [
            {
                "rank": 1,
                "anzsic_code": "S9419_03",
                "anzsic_desc": "Automotive Repair and Maintenance (own account)",
                "class_desc": "Other Repair and Maintenance",
                "division_desc": "Other Services",
                "reason": "Direct match — mobile mechanic performs automotive repair.",
            },
            {
                "rank": 2,
                "anzsic_code": "S9411_01",
                "anzsic_desc": "Automotive Electrical Services",
                "class_desc": "Automotive Repair and Maintenance",
                "division_desc": "Other Services",
                "reason": "Secondary match — electrical and mechanical overlap.",
            },
        ]
    )

    def generate_json(self, system_prompt: str, user_message: str) -> str | None:
        return self._RESPONSE


class MockLLMAdapterEmpty:
    """Simulates Gemini returning no results (triggers CSV fallback)."""

    model_name = "mock-llm-empty"

    _call_count = 0

    def generate_json(self, system_prompt: str, user_message: str) -> str | None:
        MockLLMAdapterEmpty._call_count += 1
        # Return empty list first, then real results on second call
        if MockLLMAdapterEmpty._call_count == 1:
            return "[]"
        return MockLLMAdapter._RESPONSE


# ── pytest fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    return MockEmbeddingAdapter()


@pytest.fixture
def mock_db():
    return MockDatabaseAdapter()


@pytest.fixture
def mock_llm():
    return MockLLMAdapter()


@pytest.fixture
def mock_llm_empty():
    MockLLMAdapterEmpty._call_count = 0
    return MockLLMAdapterEmpty()


@pytest.fixture
def mock_retriever(mock_embedder, mock_db, settings):
    return HybridRetriever(db=mock_db, embedder=mock_embedder, settings=settings)


@pytest.fixture
def mock_reranker(mock_llm, settings, tmp_path):
    """LLMReranker backed by mock LLM; CSV reference disabled (empty path)."""
    # Patch settings to point to a non-existent CSV so no file I/O occurs
    patched = Settings(
        gcp_project_id=settings.gcp_project_id,
        gcp_location_id=settings.gcp_location_id,
        gcp_embed_model=settings.gcp_embed_model,
        gcp_gemini_model=settings.gcp_gemini_model,
        gcloud_path=settings.gcloud_path,
        https_proxy=settings.https_proxy,
        db_dsn=settings.db_dsn,
        rrf_k=settings.rrf_k,
        retrieval_n=settings.retrieval_n,
        top_k=settings.top_k,
        embed_dim=settings.embed_dim,
        embed_batch_size=settings.embed_batch_size,
        master_csv_path=tmp_path / "nonexistent.csv",
    )
    return LLMReranker(llm=mock_llm, settings=patched)


@pytest.fixture
def pipeline(mock_retriever, mock_reranker, settings):
    return ClassifierPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        settings=settings,
    )
