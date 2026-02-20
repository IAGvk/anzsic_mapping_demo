"""
adapters/vertex_embedding.py
──────────────────────────────────────────────────────────────────────────────
Implements EmbeddingPort using Vertex AI text-embedding-005.

Key behaviour:
  - embed_query  → RETRIEVAL_QUERY  task type (asymmetric retrieval)
  - embed_document → RETRIEVAL_DOCUMENT task type
  - embed_documents_batch → single API call for up to embed_batch_size items
  - Retries on transient HTTP errors (429, 503) with exponential back-off
  - Corporate proxy support via settings.https_proxy
  - Token 401 → triggers GCPAuthManager.invalidate() then retries once

To swap to a different embedding model (e.g. OpenAI text-embedding-3-large):
  1. Write OpenAIEmbeddingAdapter implementing EmbeddingPort
  2. Change ONE import in services/container.py
  3. Update EMBED_DIM in settings / .env
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from prod.adapters.gcp_auth import GCPAuthManager
from prod.config.settings import Settings
from prod.domain.exceptions import AuthenticationError, EmbeddingError

logger = logging.getLogger(__name__)

_TASK_QUERY = "RETRIEVAL_QUERY"
_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"


def _build_embed_url(settings: Settings) -> str:
    return (
        f"https://{settings.gcp_location_id}-aiplatform.googleapis.com"
        f"/v1/projects/{settings.gcp_project_id}"
        f"/locations/{settings.gcp_location_id}"
        f"/publishers/google/models/{settings.gcp_embed_model}:predict"
    )


class VertexEmbeddingAdapter:
    """Vertex AI text-embedding-005 adapter.

    Injected into HybridRetriever via services/container.py.
    """

    def __init__(self, auth: GCPAuthManager, settings: Settings) -> None:
        self._auth = auth
        self._settings = settings
        self._url = _build_embed_url(settings)
        self._proxies = (
            {"https": f"http://{settings.https_proxy}"}
            if settings.https_proxy
            else {}
        )
        logger.debug("VertexEmbeddingAdapter ready | url=%s", self._url)

    # ── EmbeddingPort implementation ───────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._settings.gcp_embed_model

    @property
    def dimensions(self) -> int:
        return self._settings.embed_dim

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with RETRIEVAL_QUERY task type."""
        return self._embed_single(text, task_type=_TASK_QUERY)

    def embed_document(self, text: str, title: str = "") -> list[float]:
        """Embed a document with RETRIEVAL_DOCUMENT task type."""
        return self._embed_single(text, task_type=_TASK_DOCUMENT, title=title)

    def embed_documents_batch(
        self,
        texts: list[str],
        titles: list[str] | None = None,
    ) -> list[list[float] | None]:
        """Embed multiple documents in batches."""
        if not texts:
            return []
        titles = titles or [""] * len(texts)
        batch_size = self._settings.embed_batch_size
        all_results: list[list[float] | None] = []

        for start in range(0, len(texts), batch_size):
            chunk_texts = texts[start : start + batch_size]
            chunk_titles = titles[start : start + batch_size]
            batch_results = self._embed_batch(chunk_texts, _TASK_DOCUMENT, chunk_titles)
            all_results.extend(batch_results)

        return all_results

    # ── Private helpers ────────────────────────────────────────────────────

    def _embed_single(
        self,
        text: str,
        task_type: str,
        title: str = "",
        retries: int = 3,
    ) -> list[float]:
        """Call the Vertex AI Predict endpoint for a single text."""
        instance: dict[str, Any] = {"content": text, "task_type": task_type}
        if title:
            instance["title"] = title
        payload = {"instances": [instance]}

        response_json = self._post_with_retry(payload, retries=retries)
        try:
            return response_json["predictions"][0]["embeddings"]["values"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(
                f"Unexpected embed response shape: {list(response_json.keys())}"
            ) from exc

    def _embed_batch(
        self,
        texts: list[str],
        task_type: str,
        titles: list[str],
    ) -> list[list[float] | None]:
        """Call Predict for a batch of texts; returns None for failed items."""
        instances = [
            {"content": t, "task_type": task_type, **({"title": tl} if tl else {})}
            for t, tl in zip(texts, titles)
        ]
        payload = {"instances": instances}
        try:
            response_json = self._post_with_retry(payload, retries=self._settings.embed_retries)
            predictions = response_json.get("predictions", [])
            results: list[list[float] | None] = []
            for pred in predictions:
                try:
                    results.append(pred["embeddings"]["values"])
                except (KeyError, TypeError):
                    results.append(None)
            # Pad with None if fewer predictions returned than requested
            while len(results) < len(texts):
                results.append(None)
            return results
        except EmbeddingError:
            logger.warning("Batch embed failed; returning None for all %d items", len(texts))
            return [None] * len(texts)

    def _post_with_retry(
        self,
        payload: dict,
        retries: int = 3,
    ) -> dict:
        """POST to Vertex AI with retry and token refresh on 401."""
        delay = 1.0
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            token = self._auth.get_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            try:
                resp = requests.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    proxies=self._proxies,
                    timeout=self._settings.embed_timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Embed HTTP error (attempt %d/%d): %s", attempt, retries, exc)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 401:
                logger.warning("Embed 401 — invalidating token and retrying")
                self._auth.invalidate()
                continue

            if resp.status_code in (429, 503):
                logger.warning("Embed %d (attempt %d/%d) — back-off %.1fs",
                               resp.status_code, attempt, retries, delay)
                time.sleep(delay)
                delay *= 2
                continue

            if not resp.ok:
                raise EmbeddingError(
                    f"Vertex AI Embed returned HTTP {resp.status_code}: {resp.text[:200]}"
                )

            return resp.json()

        raise EmbeddingError(
            f"Embed failed after {retries} attempts"
        ) from last_exc
