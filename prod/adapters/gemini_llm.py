"""
adapters/gemini_llm.py
──────────────────────────────────────────────────────────────────────────────
Implements LLMPort using Vertex AI Gemini (generateContent REST API).

Key behaviour:
  - Sends systemInstruction + contents in the Vertex AI REST format
  - Requests JSON output via responseMimeType: application/json
  - Token 401 → triggers GCPAuthManager.invalidate() then retries once
  - Retries on 429/503 with exponential back-off
  - Returns raw JSON string (caller parses)

To swap to OpenAI-compatible endpoints:
  1. Write OpenAILLMAdapter implementing LLMPort
  2. Change ONE import in services/container.py
"""
from __future__ import annotations

import logging
import time

import requests

from prod.adapters.gcp_auth import GCPAuthManager
from prod.config.settings import Settings
from prod.domain.exceptions import AuthenticationError, LLMError

logger = logging.getLogger(__name__)


def _build_gemini_url(settings: Settings) -> str:
    return (
        f"https://{settings.gcp_location_id}-aiplatform.googleapis.com"
        f"/v1/projects/{settings.gcp_project_id}"
        f"/locations/{settings.gcp_location_id}"
        f"/publishers/google/models/{settings.gcp_gemini_model}:generateContent"
    )


class GeminiLLMAdapter:
    """Vertex AI Gemini adapter.

    Injected into LLMReranker via services/container.py.
    """

    def __init__(self, auth: GCPAuthManager, settings: Settings) -> None:
        self._auth = auth
        self._settings = settings
        self._url = _build_gemini_url(settings)
        self._proxies = (
            {"https": f"http://{settings.https_proxy}"}
            if settings.https_proxy
            else {}
        )
        logger.debug("GeminiLLMAdapter ready | model=%s", settings.gcp_gemini_model)

    # ── LLMPort implementation ─────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._settings.gcp_gemini_model

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str | None:
        """Send a prompt and return the raw JSON response string.

        Args:
            system_prompt: System-level instruction for Gemini.
            user_message:  User-turn message content.

        Returns:
            Raw JSON string from the model, or None on recoverable failure.

        Raises:
            LLMError: On unrecoverable API failure.
        """
        payload = self._build_payload(system_prompt, user_message)
        return self._post_with_retry(payload)

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_payload(self, system_prompt: str, user_message: str) -> dict:
        return {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_message}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

    def _post_with_retry(
        self,
        payload: dict,
        retries: int = 3,
    ) -> str | None:
        """POST to Gemini with token refresh on 401 and back-off on 429/503."""
        delay = 2.0
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
                    timeout=self._settings.llm_timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Gemini HTTP error (attempt %d/%d): %s", attempt, retries, exc)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 401:
                logger.warning("Gemini 401 — invalidating token and retrying")
                self._auth.invalidate()
                continue

            if resp.status_code in (429, 503):
                logger.warning(
                    "Gemini %d (attempt %d/%d) — back-off %.1fs",
                    resp.status_code, attempt, retries, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue

            if not resp.ok:
                # Log but don't raise — caller can handle None gracefully
                logger.error("Gemini HTTP %d: %s", resp.status_code, resp.text[:300])
                return None

            return self._extract_text(resp.json())

        logger.error("Gemini failed after %d attempts", retries)
        return None

    def _extract_text(self, response_json: dict) -> str | None:
        """Pull the text content out of the Gemini generateContent response."""
        try:
            candidates = response_json.get("candidates", [])
            if not candidates:
                logger.warning("Gemini response contained no candidates")
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                logger.warning("Gemini candidate contained no parts")
                return None
            text = parts[0].get("text", "").strip()
            return text if text else None
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Failed to parse Gemini response structure: %s", exc)
            return None
