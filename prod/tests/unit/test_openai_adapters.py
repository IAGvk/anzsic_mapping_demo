"""
tests/unit/test_openai_adapters.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for OpenAIEmbeddingAdapter and OpenAILLMAdapter.

All HTTP calls are intercepted with unittest.mock.patch so these tests run
fully offline — no OPENAI_API_KEY required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from prod.adapters.openai_embedding import OpenAIEmbeddingAdapter
from prod.adapters.openai_llm import OpenAILLMAdapter
from prod.config.settings import Settings
from prod.domain.exceptions import AuthenticationError, EmbeddingError


# ── Shared settings fixtures ───────────────────────────────────────────────

@pytest.fixture
def openai_settings():
    """Settings configured for OpenAI providers."""
    return Settings(
        embed_provider="openai",
        llm_provider="openai",
        openai_api_key="sk-test-key",
        openai_embed_model="text-embedding-3-small",
        openai_llm_model="gpt-4o",
        gcp_project_id="unused",
        gcp_location_id="unused",
        gcp_embed_model="unused",
        gcp_gemini_model="unused",
        gcloud_path="/usr/bin/gcloud",
        https_proxy="",
        db_dsn="dbname=anzsic_db",
        embed_dim=8,
        embed_batch_size=3,
    )


@pytest.fixture
def settings_no_key():
    """Settings with no OPENAI_API_KEY — should raise AuthenticationError."""
    return Settings(
        embed_provider="openai",
        llm_provider="openai",
        openai_api_key="",
        openai_embed_model="text-embedding-3-small",
        openai_llm_model="gpt-4o",
        gcp_project_id="unused",
        gcp_location_id="unused",
        gcp_embed_model="unused",
        gcp_gemini_model="unused",
        gcloud_path="/usr/bin/gcloud",
        https_proxy="",
        db_dsn="dbname=anzsic_db",
        embed_dim=8,
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_embed_response(vectors: list[list[float]]) -> MagicMock:
    """Build a mock requests.Response for an OpenAI embeddings call."""
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "model": "text-embedding-3-small",
    }
    return mock_resp


def _make_chat_response(content: str) -> MagicMock:
    """Build a mock requests.Response for an OpenAI chat completions call."""
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "model": "gpt-4o",
    }
    return mock_resp


# ── OpenAIEmbeddingAdapter tests ───────────────────────────────────────────

class TestOpenAIEmbeddingAdapter:

    def test_raises_if_no_api_key(self, settings_no_key):
        """Missing OPENAI_API_KEY must raise AuthenticationError immediately."""
        with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
            OpenAIEmbeddingAdapter(settings_no_key)

    def test_model_name_and_dimensions(self, openai_settings):
        adapter = OpenAIEmbeddingAdapter(openai_settings)
        assert adapter.model_name == "text-embedding-3-small"
        assert adapter.dimensions == 8  # embed_dim set to 8 in fixture

    def test_embed_query_calls_correct_endpoint(self, openai_settings):
        vec = [0.1] * 8
        with patch("prod.adapters.openai_embedding.requests.post") as mock_post:
            mock_post.return_value = _make_embed_response([vec])
            adapter = OpenAIEmbeddingAdapter(openai_settings)
            result = adapter.embed_query("café owner")

        assert result == vec
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/embeddings"
        payload = call_args[1]["json"]
        assert payload["input"] == "café owner"
        assert payload["dimensions"] == 8
        assert payload["model"] == "text-embedding-3-small"

    def test_embed_document_ignores_title(self, openai_settings):
        """title parameter is accepted but not forwarded to the API."""
        vec = [0.2] * 8
        with patch("prod.adapters.openai_embedding.requests.post") as mock_post:
            mock_post.return_value = _make_embed_response([vec])
            adapter = OpenAIEmbeddingAdapter(openai_settings)
            result = adapter.embed_document("Motor vehicle repair", title="ANZSIC")

        assert result == vec
        payload = mock_post.call_args[1]["json"]
        assert "title" not in payload

    def test_embed_documents_batch_batches_correctly(self, openai_settings):
        """embed_batch_size=3 with 5 texts should make 2 API calls."""
        vecs = [[float(i)] * 8 for i in range(5)]
        responses = [
            _make_embed_response(vecs[:3]),
            _make_embed_response(vecs[3:]),
        ]
        with patch("prod.adapters.openai_embedding.requests.post", side_effect=responses):
            adapter = OpenAIEmbeddingAdapter(openai_settings)
            results = adapter.embed_documents_batch([f"text {i}" for i in range(5)])

        assert len(results) == 5
        assert results[0] == vecs[0]
        assert results[4] == vecs[4]

    def test_embed_documents_batch_empty_returns_empty(self, openai_settings):
        adapter = OpenAIEmbeddingAdapter(openai_settings)
        assert adapter.embed_documents_batch([]) == []

    def test_401_raises_authentication_error(self, openai_settings):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 401
        with patch("prod.adapters.openai_embedding.requests.post", return_value=mock_resp):
            adapter = OpenAIEmbeddingAdapter(openai_settings)
            with pytest.raises(AuthenticationError, match="401"):
                adapter.embed_query("test")

    def test_retries_on_429_then_succeeds(self, openai_settings):
        """Should retry on 429 and succeed on the second attempt."""
        rate_limit = MagicMock()
        rate_limit.ok = False
        rate_limit.status_code = 429

        vec = [0.5] * 8
        success = _make_embed_response([vec])

        with patch("prod.adapters.openai_embedding.requests.post", side_effect=[rate_limit, success]):
            with patch("prod.adapters.openai_embedding.time.sleep"):  # skip delay
                adapter = OpenAIEmbeddingAdapter(openai_settings)
                result = adapter.embed_query("retry test")

        assert result == vec

    def test_raises_embedding_error_after_all_retries(self, openai_settings):
        """Should raise EmbeddingError when all retries are exhausted."""
        always_fail = MagicMock()
        always_fail.ok = False
        always_fail.status_code = 503

        with patch("prod.adapters.openai_embedding.requests.post", return_value=always_fail):
            with patch("prod.adapters.openai_embedding.time.sleep"):
                adapter = OpenAIEmbeddingAdapter(openai_settings)
                with pytest.raises(EmbeddingError, match="failed after"):
                    adapter.embed_query("will fail")


# ── OpenAILLMAdapter tests ─────────────────────────────────────────────────

class TestOpenAILLMAdapter:

    def test_raises_if_no_api_key(self, settings_no_key):
        """Missing OPENAI_API_KEY must raise AuthenticationError immediately."""
        with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
            OpenAILLMAdapter(settings_no_key)

    def test_model_name(self, openai_settings):
        adapter = OpenAILLMAdapter(openai_settings)
        assert adapter.model_name == "gpt-4o"

    def test_generate_json_happy_path(self, openai_settings):
        payload_json = json.dumps([{"rank": 1, "anzsic_code": "S9419_03"}])
        with patch("prod.adapters.openai_llm.requests.post") as mock_post:
            mock_post.return_value = _make_chat_response(payload_json)
            adapter = OpenAILLMAdapter(openai_settings)
            result = adapter.generate_json("system prompt", "user message")

        assert result == payload_json

    def test_generate_json_sends_correct_payload(self, openai_settings):
        """Payload must include JSON mode and both message roles."""
        with patch("prod.adapters.openai_llm.requests.post") as mock_post:
            mock_post.return_value = _make_chat_response("{}")
            adapter = OpenAILLMAdapter(openai_settings)
            adapter.generate_json("sys", "usr")

        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"
        body = call_args[1]["json"]
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0.1
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["system", "user"]

    def test_returns_none_on_empty_choices(self, openai_settings):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": []}

        with patch("prod.adapters.openai_llm.requests.post", return_value=mock_resp):
            adapter = OpenAILLMAdapter(openai_settings)
            result = adapter.generate_json("sys", "usr")

        assert result is None

    def test_returns_none_on_non_ok_response(self, openai_settings):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 400   # Bad request — immediate failure, no retry
        mock_resp.text = "Bad Request"

        with patch("prod.adapters.openai_llm.requests.post", return_value=mock_resp):
            adapter = OpenAILLMAdapter(openai_settings)
            result = adapter.generate_json("sys", "usr")

        assert result is None

    def test_401_raises_authentication_error(self, openai_settings):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 401

        with patch("prod.adapters.openai_llm.requests.post", return_value=mock_resp):
            adapter = OpenAILLMAdapter(openai_settings)
            with pytest.raises(AuthenticationError, match="401"):
                adapter.generate_json("sys", "usr")

    def test_retries_on_429_then_succeeds(self, openai_settings):
        rate_limit = MagicMock()
        rate_limit.ok = False
        rate_limit.status_code = 429

        success = _make_chat_response('{"result": "ok"}')

        with patch("prod.adapters.openai_llm.requests.post", side_effect=[rate_limit, success]):
            with patch("prod.adapters.openai_llm.time.sleep"):
                adapter = OpenAILLMAdapter(openai_settings)
                result = adapter.generate_json("sys", "usr")

        assert result == '{"result": "ok"}'

    def test_returns_none_after_all_retries_exhausted(self, openai_settings):
        always_fail = MagicMock()
        always_fail.ok = False
        always_fail.status_code = 503

        with patch("prod.adapters.openai_llm.requests.post", return_value=always_fail):
            with patch("prod.adapters.openai_llm.time.sleep"):
                adapter = OpenAILLMAdapter(openai_settings)
                result = adapter.generate_json("sys", "usr")

        assert result is None


# ── container.py provider routing tests ───────────────────────────────────

class TestContainerProviderRouting:

    def test_unknown_embed_provider_raises(self):
        from prod.domain.exceptions import ConfigurationError
        from prod.services.container import _build_embedder

        bad_settings = Settings(
            embed_provider="cohere",
            llm_provider="vertex",
            openai_api_key="",
            gcp_project_id="p",
            gcp_location_id="l",
            gcp_embed_model="m",
            gcp_gemini_model="g",
            gcloud_path="/usr/bin/gcloud",
            https_proxy="",
            db_dsn="dbname=test",
        )
        with pytest.raises(ConfigurationError, match="EMBED_PROVIDER"):
            _build_embedder(bad_settings)

    def test_unknown_llm_provider_raises(self):
        from prod.domain.exceptions import ConfigurationError
        from prod.services.container import _build_llm

        bad_settings = Settings(
            embed_provider="vertex",
            llm_provider="anthropic",
            openai_api_key="",
            gcp_project_id="p",
            gcp_location_id="l",
            gcp_embed_model="m",
            gcp_gemini_model="g",
            gcloud_path="/usr/bin/gcloud",
            https_proxy="",
            db_dsn="dbname=test",
        )
        with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
            _build_llm(bad_settings)

    def test_openai_embed_provider_returns_openai_adapter(self, openai_settings):
        from prod.services.container import _build_embedder

        with patch("prod.adapters.openai_embedding.requests.post"):  # guard
            adapter = _build_embedder(openai_settings)

        assert adapter.__class__.__name__ == "OpenAIEmbeddingAdapter"

    def test_openai_llm_provider_returns_openai_adapter(self, openai_settings):
        from prod.services.container import _build_llm

        adapter = _build_llm(openai_settings)
        assert adapter.__class__.__name__ == "OpenAILLMAdapter"
