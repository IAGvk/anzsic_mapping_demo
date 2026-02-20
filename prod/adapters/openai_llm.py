"""
adapters/openai_llm.py
──────────────────────────────────────────────────────────────────────────────
Implements LLMPort using the OpenAI Chat Completions API.

Key behaviour:
  - Uses /v1/chat/completions via raw requests (no openai SDK dependency)
  - Requests JSON output via response_format={"type": "json_object"}
  - system_prompt → system role message; user_message → user role message
  - Retries on 429 / 500 with exponential back-off
  - Returns the raw JSON string (caller parses); None on recoverable failure

Required env vars:
  OPENAI_API_KEY     — your OpenAI secret key  (sk-...)
  OPENAI_LLM_MODEL   — default: gpt-4o

To enable:
  Set LLM_PROVIDER=openai in your .env file.
"""
from __future__ import annotations

import logging
import time

import requests

from prod.config.settings import Settings
from prod.domain.exceptions import AuthenticationError, LLMError

logger = logging.getLogger(__name__)

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


class OpenAILLMAdapter:
    """OpenAI GPT chat completions adapter.

    Injected into LLMReranker via services/container.py when
    ``LLM_PROVIDER=openai`` is set in the environment.

    JSON mode is enabled via ``response_format={"type": "json_object"}``,
    which guarantees the model returns syntactically valid JSON without
    needing to strip markdown fences.

    .. note::
        OpenAI's JSON mode requires the word "JSON" to appear somewhere in
        the prompt.  The existing ``build_system_prompt()`` in
        ``config/prompts.py`` already includes this — no prompt changes are
        needed.
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
        logger.debug("OpenAILLMAdapter ready | model=%s", settings.openai_llm_model)

    # ── LLMPort implementation ─────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Name of the underlying OpenAI chat model."""
        return self._settings.openai_llm_model

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str | None:
        """Send a prompt and return the raw JSON response string.

        Uses ``response_format={"type": "json_object"}`` to enforce valid
        JSON output from the model.  The caller is responsible for parsing
        the returned string.

        Args:
            system_prompt: System-level instruction for the model.
            user_message:  User-turn message content.

        Returns:
            Raw JSON string from the model, or ``None`` on recoverable failure.

        Raises:
            LLMError: On unrecoverable API failure (non-2xx after all retries).
        """
        payload = self._build_payload(system_prompt, user_message)
        return self._post_with_retry(payload)

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_payload(self, system_prompt: str, user_message: str) -> dict:
        """Build the OpenAI chat completions request body."""
        return {
            "model": self._settings.openai_llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

    def _post_with_retry(
        self,
        payload: dict,
        retries: int = 3,
    ) -> str | None:
        """POST to the OpenAI API with back-off on 429 / 500."""
        delay = 2.0
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(
                    _OPENAI_CHAT_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=self._settings.llm_timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "OpenAI LLM request error (attempt %d/%d): %s",
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
                    "OpenAI LLM %d (attempt %d/%d) — back-off %.1fs",
                    resp.status_code, attempt, retries, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue

            if not resp.ok:
                logger.error(
                    "OpenAI LLM HTTP %d: %s",
                    resp.status_code, resp.text[:300],
                )
                return None

            return self._extract_text(resp.json())

        logger.error("OpenAI LLM failed after %d attempts", retries)
        return None

    def _extract_text(self, response_json: dict) -> str | None:
        """Pull the content string out of the chat completions response."""
        try:
            choices = response_json.get("choices", [])
            if not choices:
                logger.warning("OpenAI response contained no choices")
                return None
            content = choices[0].get("message", {}).get("content", "").strip()
            return content if content else None
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Failed to parse OpenAI response structure: %s", exc)
            return None
