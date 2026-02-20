"""
adapters/openai_embedding.py
──────────────────────────────────────────────────────────────────────────────
Implements EmbeddingPort using the OpenAI Embeddings API.

Key behaviour:
  - Uses /v1/embeddings via raw requests (no openai SDK dependency)
  - Passes `dimensions=settings.embed_dim` so output matches whatever
    pgvector column width the DB was initialised with
  - embed_query and embed_document call the same endpoint (OpenAI embeddings
    are symmetric — no RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT distinction)
  - Batches embed_documents_batch to stay within 2048-token-per-item limit
  - Retries on 429 / 500 with exponential back-off

Required env vars:
  OPENAI_API_KEY        — your OpenAI secret key  (sk-...)
  OPENAI_EMBED_MODEL    — default: text-embedding-3-small
  EMBED_DIM             — default: 768; 1536 recommended for text-embedding-3-small

To enable:
  Set EMBED_PROVIDER=openai in your .env file.
"""
from __future__ import annotations

import logging
import time

import requests

from prod.config.settings import Settings
from prod.domain.exceptions import AuthenticationError, EmbeddingError

logger = logging.getLogger(__name__)

_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbeddingAdapter:
    """OpenAI text-embedding adapter.

    Injected into HybridRetriever via services/container.py when
    ``EMBED_PROVIDER=openai`` is set in the environment.

    Note on dimensions:
        OpenAI's ``text-embedding-3-*`` models accept a ``dimensions``
        parameter to reduce the output to any size ≤ the model's native
        dimension.  This adapter always passes ``settings.embed_dim`` so
        output vectors are compatible with the pgvector column width that
        the database was initialised with.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise AuthenticationError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        self._settings = settings
        self._headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        logger.debug(
            "OpenAIEmbeddingAdapter ready | model=%s dim=%d",
            settings.openai_embed_model,
            settings.embed_dim,
        )

    # ── EmbeddingPort implementation ───────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Name of the underlying OpenAI embedding model."""
        return self._settings.openai_embed_model

    @property
    def dimensions(self) -> int:
        """Vector dimensionality (controlled by ``EMBED_DIM`` env var)."""
        return self._settings.embed_dim

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        OpenAI embeddings are symmetric, so the same endpoint and model
        is used for queries and documents.

        Args:
            text: Natural-language query string.

        Returns:
            Dense float vector of length ``settings.embed_dim``.

        Raises:
            EmbeddingError: On API failure or unexpected response shape.
        """
        return self._embed_one(text)

    def embed_document(self, text: str, title: str = "") -> list[float]:
        """Embed a document for storage.

        The ``title`` parameter is accepted for interface compatibility
        but is not used by the OpenAI embeddings API.

        Args:
            text:  Document body text.
            title: Ignored (OpenAI API does not use document titles).

        Returns:
            Dense float vector of length ``settings.embed_dim``.
        """
        return self._embed_one(text)

    def embed_documents_batch(
        self,
        texts: list[str],
        titles: list[str] | None = None,
    ) -> list[list[float] | None]:
        """Embed multiple documents, batched to respect API limits.

        Args:
            texts:  List of document strings to embed.
            titles: Ignored (OpenAI API does not use document titles).

        Returns:
            List of float vectors; ``None`` for any item that failed.
        """
        if not texts:
            return []

        batch_size = self._settings.embed_batch_size
        all_results: list[list[float] | None] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            try:
                batch_results = self._embed_batch(chunk)
                all_results.extend(batch_results)
            except EmbeddingError:
                logger.warning(
                    "OpenAI batch embed failed for items %d–%d; "
                    "returning None for all %d items in chunk",
                    start,
                    start + len(chunk) - 1,
                    len(chunk),
                )
                all_results.extend([None] * len(chunk))

        return all_results

    # ── Private helpers ────────────────────────────────────────────────────

    def _embed_one(self, text: str) -> list[float]:
        """Call the OpenAI embeddings endpoint for a single text string."""
        payload = {
            "model": self._settings.openai_embed_model,
            "input": text,
            "dimensions": self._settings.embed_dim,
        }
        data = self._post_with_retry(payload)
        try:
            return data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(
                f"Unexpected OpenAI embed response shape: {list(data.keys())}"
            ) from exc

    def _embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Call the OpenAI embeddings endpoint for a list of strings."""
        payload = {
            "model": self._settings.openai_embed_model,
            "input": texts,
            "dimensions": self._settings.embed_dim,
        }
        data = self._post_with_retry(payload)
        items = data.get("data", [])
        # API returns items in index order but we validate just in case
        result: list[list[float] | None] = [None] * len(texts)
        for item in items:
            idx = item.get("index", -1)
            if 0 <= idx < len(texts):
                result[idx] = item.get("embedding")
        return result

    def _post_with_retry(
        self,
        payload: dict,
        retries: int = 3,
    ) -> dict:
        """POST to the OpenAI API with retry on 429 / 500."""
        delay = 2.0
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(
                    _OPENAI_EMBED_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=self._settings.embed_timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "OpenAI embed request error (attempt %d/%d): %s",
                    attempt, retries, exc,
                )
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 401:
                raise AuthenticationError(
                    "OpenAI returned 401 Unauthorised. "
                    "Check that OPENAI_API_KEY is valid."
                )

            if resp.status_code in (429, 500, 503):
                logger.warning(
                    "OpenAI embed %d (attempt %d/%d) — back-off %.1fs",
                    resp.status_code, attempt, retries, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue

            if not resp.ok:
                raise EmbeddingError(
                    f"OpenAI embed HTTP {resp.status_code}: {resp.text[:300]}"
                )

            return resp.json()

        raise EmbeddingError(
            f"OpenAI embed failed after {retries} attempts"
        ) from last_exc
