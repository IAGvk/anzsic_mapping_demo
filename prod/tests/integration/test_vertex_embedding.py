"""
tests/integration/test_vertex_embedding.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for VertexEmbeddingAdapter.

Requires:
  • Active gcloud auth (gcloud auth application-default login)
  • Network access to Vertex AI (or corporate proxy configured)
  • GCP_PROJECT_ID, GCP_LOCATION_ID, GCLOUD_PATH set (or defaults applied)

Run with:
  pytest -m integration prod/tests/integration/test_vertex_embedding.py -v

IMPORTANT: These tests make real API calls and incur GCP costs.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    from prod.adapters.gcp_auth import GCPAuthManager
    from prod.adapters.vertex_embedding import VertexEmbeddingAdapter
    from prod.config.settings import get_settings
    settings = get_settings()
    auth = GCPAuthManager(settings)
    return VertexEmbeddingAdapter(auth, settings)


class TestEmbedQuery:
    def test_returns_correct_dimension(self, embedder):
        vec = embedder.embed_query("mobile mechanic")
        assert len(vec) == 768

    def test_returns_floats(self, embedder):
        vec = embedder.embed_query("plumber")
        assert all(isinstance(v, float) for v in vec)

    def test_different_queries_produce_different_vectors(self, embedder):
        vec1 = embedder.embed_query("mobile mechanic")
        vec2 = embedder.embed_query("registered nurse")
        assert vec1 != vec2

    def test_similar_queries_produce_similar_vectors(self, embedder):
        """Cosine similarity between related queries should be high."""
        import math
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x ** 2 for x in a))
            nb = math.sqrt(sum(x ** 2 for x in b))
            return dot / (na * nb) if na and nb else 0.0

        v1 = embedder.embed_query("car mechanic")
        v2 = embedder.embed_query("automobile technician")
        v3 = embedder.embed_query("registered nurse")
        assert cosine(v1, v2) > cosine(v1, v3)


class TestEmbedDocumentsBatch:
    def test_batch_length_matches_input(self, embedder):
        texts = ["plumber", "electrician", "carpenter"]
        results = embedder.embed_documents_batch(texts)
        assert len(results) == len(texts)

    def test_batch_items_are_768_dim(self, embedder):
        texts = ["nurse", "doctor"]
        results = embedder.embed_documents_batch(texts)
        for vec in results:
            assert vec is not None
            assert len(vec) == 768

    def test_empty_batch_returns_empty(self, embedder):
        assert embedder.embed_documents_batch([]) == []


class TestModelProperties:
    def test_model_name(self, embedder):
        assert "embedding" in embedder.model_name.lower() or len(embedder.model_name) > 0

    def test_dimensions(self, embedder):
        assert embedder.dimensions == 768
