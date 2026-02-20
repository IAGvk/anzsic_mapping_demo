"""
adapters/gcp_auth.py
──────────────────────────────────────────────────────────────────────────────
GCP authentication manager.

Wraps the `gcloud auth print-access-token` subprocess call, caches the token
in memory, and refreshes automatically when it is within TOKEN_REFRESH_MARGIN
seconds of expiry.

Design notes:
  - All adapters that need GCP auth receive a GCPAuthManager instance via DI.
  - A single GCPAuthManager is created in services/container.py and shared
    across VertexEmbeddingAdapter and GeminiLLMAdapter → one token, one call.
  - Thread-safe: uses a threading.Lock for token refresh in multi-threaded
    contexts (Streamlit, FastAPI worker threads).
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field

from prod.config.settings import Settings
from prod.domain.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

# How many seconds before expiry to proactively refresh the token.
TOKEN_REFRESH_MARGIN: int = 120  # 2 minutes

# GCP access tokens expire after 3600 seconds; we assume this conservatively.
TOKEN_TTL_SECONDS: int = 3600


@dataclass
class _TokenState:
    """Internal mutable token state (not frozen so we can refresh in place)."""

    value: str = ""
    expires_at: float = 0.0  # Unix timestamp


class GCPAuthManager:
    """Manages a GCP Application Default Credentials access token.

    Usage (injected by container.py — do not instantiate manually):
        auth = GCPAuthManager(settings)
        token = auth.get_token()          # fresh or cached
    """

    def __init__(self, settings: Settings) -> None:
        self._gcloud_path = settings.gcloud_path
        self._state = _TokenState()
        self._lock = threading.Lock()
        logger.debug("GCPAuthManager initialised | gcloud=%s", self._gcloud_path)

    # ── Public API ─────────────────────────────────────────────────────────

    def get_token(self) -> str:
        """Return a valid access token, refreshing if necessary.

        Returns:
            Bearer token string.

        Raises:
            AuthenticationError: If gcloud fails.
        """
        with self._lock:
            if self._needs_refresh():
                self._refresh()
            return self._state.value

    def invalidate(self) -> None:
        """Force the next call to get_token() to fetch a fresh token."""
        with self._lock:
            self._state.expires_at = 0.0
            logger.debug("GCPAuthManager: token invalidated")

    # ── Private helpers ────────────────────────────────────────────────────

    def _needs_refresh(self) -> bool:
        margin_time = time.time() + TOKEN_REFRESH_MARGIN
        return self._state.value == "" or self._state.expires_at <= margin_time

    def _refresh(self) -> None:
        logger.info("GCPAuthManager: refreshing access token …")
        try:
            result = subprocess.run(
                [self._gcloud_path, "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise AuthenticationError("gcloud timed out fetching access token") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "(no stderr)"
            raise AuthenticationError(
                f"gcloud auth print-access-token failed: {stderr}"
            ) from exc
        except FileNotFoundError as exc:
            raise AuthenticationError(
                f"gcloud not found at '{self._gcloud_path}'. "
                "Set the GCLOUD_PATH environment variable."
            ) from exc

        token = result.stdout.strip()
        if not token:
            raise AuthenticationError("gcloud returned an empty access token")

        self._state.value = token
        self._state.expires_at = time.time() + TOKEN_TTL_SECONDS
        logger.info("GCPAuthManager: token refreshed (expires in %ds)", TOKEN_TTL_SECONDS)
