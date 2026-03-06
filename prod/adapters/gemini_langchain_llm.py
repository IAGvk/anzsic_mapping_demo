"""
adapters/gemini_langchain_llm.py
──────────────────────────────────────────────────────────────────────────────
Implements LLMPort using LangChain's ChatVertexAI integration.

Key behaviour:
  - Auth is handled entirely by LangChain via Application Default Credentials
    (ADC) — no GCPAuthManager / raw bearer token required.
  - run `gcloud auth application-default login` once (already done if you use
    GeminiLLMAdapter, since both rely on the same underlying GCP credentials).
  - Requests JSON output via `response_mime_type="application/json"`.
  - Retries on LangChain-surfaced rate-limit / service errors with exponential
    back-off (mirrors GeminiLLMAdapter behaviour).
  - Returns raw JSON string — caller (LLMReranker) parses it.

To activate: set  LLM_PROVIDER=langchain_gemini  in your .env or environment.
"""
from __future__ import annotations

import logging
import os
import time

from prod.adapters.gcp_auth import GCPAuthManager
from prod.config.settings import Settings
from prod.domain.exceptions import LLMError

logger = logging.getLogger(__name__)

# LangChain status codes / exception types that warrant a retry
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 503})


class _GCPAuthCredentials:
    """Thin google.auth.credentials.Credentials shim backed by GCPAuthManager.

    This bypasses google-auth's own token-refresh flow (which makes HTTPS
    requests to oauth2.googleapis.com and fails behind a corporate proxy)
    by delegating directly to GCPAuthManager.get_token(), which in turn calls
    `gcloud auth print-access-token` — the same path used by GeminiLLMAdapter.
    """

    def __init__(self, auth: GCPAuthManager) -> None:
        self._auth = auth

    # ── google.auth.credentials.Credentials interface ──────────────────────

    @property
    def token(self) -> str:
        return self._auth.get_token()

    @property
    def valid(self) -> bool:
        return True  # GCPAuthManager handles validity internally

    @property
    def expired(self) -> bool:
        return False  # GCPAuthManager handles expiry internally

    @property
    def universe_domain(self) -> str:
        return "googleapis.com"

    def refresh(self, request) -> None:  # noqa: ARG002
        """Called by google-auth when it thinks the token needs refreshing."""
        self._auth.invalidate()  # next .token call will fetch a fresh one

    def before_request(self, request, method, url, headers) -> None:  # noqa: ARG002
        self.apply(headers)

    def apply(self, headers, token: str | None = None) -> None:
        headers["Authorization"] = f"Bearer {token or self.token}"


class GeminiLangChainLLMAdapter:
    """LangChain ChatVertexAI adapter.

    Injected into LLMReranker via services/container.py when
    LLM_PROVIDER=langchain_gemini.

    Args:
        settings: Application settings (provides project/location/model).
    """

    def __init__(self, settings: Settings, auth: GCPAuthManager | None = None) -> None:
        """
        Args:
            settings: Application settings.
            auth:     GCPAuthManager instance.  When provided, its token is
                      injected directly into ChatVertexAI, bypassing
                      google-auth's own refresh flow (which fails behind a
                      corporate proxy that intercepts TLS).  Mirrors the
                      approach used by GeminiLLMAdapter.
        """
        # Lazy import so the package is only required when this adapter is used.
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-vertexai is not installed. "
                "Run: pip install langchain-google-vertexai"
            ) from exc

        self._settings = settings

        # ── Corporate proxy ────────────────────────────────────────────────
        # Must be set before ChatVertexAI initialises its REST transport so
        # that google-auth-httplib2 routes through the proxy.
        if settings.https_proxy:
            proxy_url = f"http://{settings.https_proxy}"
            os.environ.setdefault("HTTPS_PROXY", proxy_url)
            os.environ.setdefault("https_proxy", proxy_url)
            logger.debug("GeminiLangChainLLMAdapter: proxy set to %s", proxy_url)

        # ── Credentials ────────────────────────────────────────────────────
        # If a GCPAuthManager is provided, wrap it in our shim so that
        # google-auth never tries to refresh the token itself over HTTPS
        # (the refresh call to oauth2.googleapis.com also fails behind the
        # corporate proxy and is the source of RefreshError).
        credentials = _GCPAuthCredentials(auth) if auth else None

        # ── api_transport="rest" ───────────────────────────────────────────
        # Switches from gRPC to HTTPS/REST so that:
        #   1. The HTTPS_PROXY env var above is respected.
        #   2. Python's SSL stack (patched by ssl_corp_fix.py) is used.
        # gRPC has its own bundled OpenSSL that ignores both → SSL failure.
        self._llm = ChatVertexAI(
            model_name=settings.gcp_gemini_model,
            project=settings.gcp_project_id,
            location=settings.gcp_location_id,
            temperature=0.1,
            max_retries=0,           # we handle retries ourselves
            api_transport="rest",    # avoid gRPC SSL issues behind corporate proxy
            credentials=credentials, # None → falls back to ADC (non-proxy envs)
            model_kwargs={
                "generation_config": {
                    "response_mime_type": "application/json",
                }
            },
        )
        logger.debug(
            "GeminiLangChainLLMAdapter ready | model=%s project=%s location=%s",
            settings.gcp_gemini_model,
            settings.gcp_project_id,
            settings.gcp_location_id,
        )

    def get_raw_llm(self):
        """Return the underlying ChatVertexAI instance.

        Use this when you need direct access to LangChain-native methods that
        the LLMPort interface does not expose, for example:

            raw = adapter.get_raw_llm()
            llm_with_tools = raw.bind_tools([my_tool])
            llm_structured  = raw.with_structured_output(MySchema)

        The returned object has all corporate proxy and auth fixes applied —
        you do not need to reconfigure anything.
        """
        return self._llm

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
            system_prompt: System-level instruction for the model.
            user_message:  User-turn message content.

        Returns:
            Raw JSON string from the model, or None on recoverable failure.

        Raises:
            LLMError: On unrecoverable API failure.
        """
        _t_total = time.perf_counter()
        result = self._invoke_with_retry(system_prompt, user_message)
        logger.info(
            "⏱ [GeminiLangChain] operation=generate_json_total elapsed=%.3fs",
            time.perf_counter() - _t_total,
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_messages(self, system_prompt: str, user_message: str) -> list:
        """Construct the LangChain message list."""
        from langchain_core.messages import HumanMessage, SystemMessage

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

    def _invoke_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        retries: int = 3,
    ) -> str | None:
        """Invoke the LLM with exponential back-off on transient errors."""
        from langchain_core.exceptions import OutputParserException

        delay = 2.0
        messages = self._build_messages(system_prompt, user_message)

        # ── Log prompt sizes once ──────────────────────────────────────────
        _sys_chars = len(system_prompt)
        _usr_chars = len(user_message)
        logger.info(
            "⏱ [GeminiLangChain] prompt_size system_chars=%d user_chars=%d "
            "est_tokens≈%d",
            _sys_chars,
            _usr_chars,
            (_sys_chars + _usr_chars) // 4,
        )

        for attempt in range(1, retries + 1):
            try:
                _t0 = time.perf_counter()
                response = self._llm.invoke(messages)
                _resp_chars = len(response.content) if hasattr(response, "content") else 0
                logger.info(
                    "⏱ [GeminiLangChain] operation=llm_invoke attempt=%d "
                    "elapsed=%.3fs response_chars=%d",
                    attempt, time.perf_counter() - _t0, _resp_chars,
                )
                return self._extract_text(response)

            except OutputParserException as exc:
                # Non-retryable: the model returned something but it's unparseable
                logger.error("LangChain output parse error: %s", exc)
                return None

            except Exception as exc:  # noqa: BLE001
                exc_str = str(exc)
                is_retryable = any(
                    str(code) in exc_str for code in _RETRYABLE_STATUS_CODES
                )
                if is_retryable and attempt < retries:
                    logger.warning(
                        "LangChain Gemini transient error (attempt %d/%d) — "
                        "back-off %.1fs: %s",
                        attempt, retries, delay, exc_str[:200],
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                logger.error(
                    "LangChain Gemini failed (attempt %d/%d): %s",
                    attempt, retries, exc_str[:300],
                )
                return None

        logger.error("GeminiLangChainLLMAdapter failed after %d attempts", retries)
        return None

    def _extract_text(self, response) -> str | None:
        """Pull the text content from the LangChain AIMessage."""
        try:
            text = response.content
            if isinstance(text, str):
                text = text.strip()
                return text if text else None
            # content can be a list of dicts in multimodal responses
            if isinstance(text, list):
                parts = [p.get("text", "") for p in text if isinstance(p, dict)]
                joined = "".join(parts).strip()
                return joined if joined else None
        except (AttributeError, TypeError, KeyError) as exc:
            logger.error("Failed to extract text from LangChain response: %s", exc)
        return None
