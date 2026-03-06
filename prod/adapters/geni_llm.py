"""
adapters/geni_llm.py
──────────────────────────────────────────────────────────────────────────────
Implements LLMPort using the GENI platform (IAG internal LLM API).

Architecture notes (hexagonal):
  - This adapter is the ONLY file that knows about GENI.
  - It receives Settings via DI from services/container.py.
  - services/reranker.py only sees the LLMPort interface — zero changes there.

How GENI works vs direct Gemini:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Direct Gemini            │  GENI                                       │
  │  ─────────────────────── │  ─────────────────────────────────────────  │
  │  system_prompt → API      │  system_prompt baked into bot (UI config)   │
  │  user_message  → API      │  user_message sent as conversation question │
  │  synchronous response     │  async: create → post → poll → fetch        │
  └─────────────────────────────────────────────────────────────────────────┘

CSV reference strategy (two-path with automatic fallback):
  Preferred — file upload:
    On first call the ANZSIC master CSV is uploaded via POST /api/files.
    GENI validates the file and if it fits within the token limit it stores
    the bytes in GCS and returns a UUID.  That UUID is cached on the adapter
    instance (or pre-configured via GENI_CSV_FILE_ID to skip the upload
    entirely).  The UUID is passed as file_ids on every question so the LLM
    always has the full reference without bloating the message body.

  Fallback — inline text:
    GENI imposes a 128 K-token limit on uploaded files.  If the CSV exceeds
    this (currently ~632 K tokens), the upload is rejected with HTTP 422.
    The adapter catches this, marks the upload as permanently failed, and
    switches to the old sentinel/text-injection approach for the lifetime of
    the process: the CSV block is extracted from system_prompt (which the
    reranker always builds with include_reference=True) and appended directly
    to the user message.

  The GENI bot is configured with:
    - Instructions  : RERANK_SYSTEM_BASE from config/prompts.py
    - Language Model: Gemini 2.5 Flash
    - Temperature   : 0 (deterministic)
    - Output Format : JSON

To activate: set LLM_PROVIDER=geni in .env
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import requests

from prod.config.settings import Settings
from prod.domain.exceptions import LLMError

# GCP identity tokens are valid for 1 hour; we refresh with 5-minute margin.
_TOKEN_TTL_SECONDS = 55 * 60

logger = logging.getLogger(__name__)

# Sentinel that appears in CSV_REFERENCE_HEADER (config/prompts.py).
# Used to locate the CSV block in system_prompt for the text-fallback path.
_CSV_SENTINEL = "FULL ANZSIC REFERENCE"


class GeniLLMAdapter:
    """GENI platform adapter satisfying LLMPort.

    Injected into LLMReranker via services/container.py.
    No other file in the codebase references this class.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.geni_base_url.rstrip("/")
        self._domain = settings.geni_domain
        self._bot_version_id = settings.geni_bot_version_id
        self._poll_timeout = settings.geni_poll_timeout
        self._poll_interval = settings.geni_poll_interval

        if not self._bot_version_id:
            raise LLMError(
                "GENI_BOT_VERSION_ID is not set. "
                "Retrieve it via: GET /api/bots_v2/<url-suffix> "
                "and add it to your .env file."
            )
        # Cached file UUID for the ANZSIC master CSV.
        # Pre-populated from GENI_CSV_FILE_ID if set; otherwise uploaded lazily.
        self._csv_file_id: str | None = settings.geni_csv_file_id or None
        # Set to True permanently if GENI rejects the upload (e.g. token limit),
        # or immediately if GENI_DISABLE_CSV_UPLOAD=true in .env.
        self._csv_upload_failed: bool = settings.geni_disable_csv_upload
        if self._csv_upload_failed:
            logger.info(
                "GeniLLMAdapter: GENI_DISABLE_CSV_UPLOAD=true — "
                "skipping CSV upload, using inline-text fallback"
            )

        # ── Token cache ────────────────────────────────────────────────────
        # GCP identity tokens are valid for 1 hour.  We cache for 55 minutes
        # to avoid a ~1.4s subprocess call before every single HTTP request.
        self._cached_token: str | None = None
        self._token_expires_at: float = 0.0  # time.monotonic() value
        logger.debug(
            "GeniLLMAdapter ready | base_url=%s domain=%s bot_version_id=%s csv_file_id=%s",
            self._base_url,
            self._domain,
            self._bot_version_id,
            self._csv_file_id or "<will upload on first call>",
        )

    # ── LLMPort interface ──────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Identifier logged by LLMReranker on init."""
        return f"geni/gemini-2.5-pro (bot_version={self._bot_version_id})"

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str | None:
        """Implements LLMPort.generate_json().

        The GENI bot already holds the system instructions (RERANK_SYSTEM_BASE),
        so ``system_prompt`` is only used for the text-fallback path (see below).

        CSV reference is passed via one of two paths (automatic selection):

        1. **File path** (preferred): CSV uploaded to GENI once; UUID reused via
           ``file_ids`` on every question.  Zero extra message bytes.
        2. **Text fallback** (if upload rejected by GENI, e.g. token > 128 K):
           CSV block is extracted from ``system_prompt`` and appended to
           ``user_message`` inline — same as the original sentinel approach.

        Args:
            system_prompt: Built by build_system_prompt() in config/prompts.py.
                           Contains the CSV block when include_reference=True.
                           Used only by the text-fallback path.
            user_message:  Query + candidate block built by build_user_message().

        Returns:
            Raw JSON string from GENI, or None on recoverable failure.

        Raises:
            LLMError: On unrecoverable API failure.
        """
        # ── Step 1: resolve CSV delivery method ───────────────────────────
        csv_file_id: str | None = None
        if not self._csv_upload_failed:
            try:
                csv_file_id = self._ensure_csv_uploaded()
            except LLMError as exc:
                logger.warning(
                    "GeniLLMAdapter: CSV file upload rejected — switching to "
                    "inline text fallback for this process. Reason: %s", exc,
                )
                self._csv_upload_failed = True
                self._csv_file_id = None  # clear any partial state

        # ── Step 2: build final question content ──────────────────────────
        if csv_file_id:
            # File path: CSV lives in GENI/GCS, referenced by UUID.
            final_content = user_message
            file_ids: list[str] | None = [csv_file_id]
            logger.debug(
                "GeniLLMAdapter.generate_json [file path]: %d chars + file_id=%s",
                len(user_message), csv_file_id,
            )
        else:
            # Text fallback: extract CSV block from system_prompt and append inline.
            file_ids = None
            if _CSV_SENTINEL in system_prompt:
                csv_start = system_prompt.index(_CSV_SENTINEL)
                csv_block = system_prompt[csv_start:]
                final_content = (
                    f"{user_message}\n\n"
                    f"── ADDITIONAL REFERENCE ──\n"
                    f"{csv_block}"
                )
                logger.info(
                    "GeniLLMAdapter.generate_json [text fallback]: "
                    "appending %d chars CSV to message", len(csv_block),
                )
            else:
                final_content = user_message
                logger.debug(
                    "GeniLLMAdapter.generate_json [text fallback]: "
                    "no CSV sentinel in system_prompt, sending %d chars",
                    len(user_message),
                )

        # ── Step 3: execute GENI conversation flow ────────────────────────
        _t_total = time.perf_counter()
        try:
            conversation_id = self._create_conversation()
            question_id = self._post_question(
                conversation_id, final_content, file_ids=file_ids
            )
            result = self._poll_for_answer(question_id)
            logger.info(
                "⏱ [GeniLLM] operation=generate_json_total elapsed=%.3fs "
                "csv_path=%s",
                time.perf_counter() - _t_total,
                "file" if csv_file_id else "inline_text",
            )
            return result
        except LLMError:
            raise
        except Exception as exc:
            logger.error("GeniLLMAdapter unexpected error: %s", exc)
            raise LLMError(f"GENI call failed: {exc}") from exc

    # ── Auth ───────────────────────────────────────────────────────────────

    def _get_token(self) -> str:
        """Return a GCloud identity token, refreshing only when near-expiry.

        GCP identity tokens are valid for 1 hour.  We cache the token for
        55 minutes (_TOKEN_TTL_SECONDS) so the ~1.4s gcloud subprocess is
        only executed once per process rather than before every HTTP request.
        """
        now = time.monotonic()
        if self._cached_token and now < self._token_expires_at:
            logger.debug("GeniLLM: using cached token (expires in %.0fs)",
                         self._token_expires_at - now)
            return self._cached_token

        _t0 = time.perf_counter()
        try:
            result = subprocess.run(
                [self._settings.gcloud_path, "auth", "print-identity-token"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            token = result.stdout.strip()
            self._cached_token = token
            self._token_expires_at = time.monotonic() + _TOKEN_TTL_SECONDS
            logger.info(
                "⏱ [GeniLLM] operation=get_token elapsed=%.3fs (token refreshed, valid ~55min)",
                time.perf_counter() - _t0,
            )
            return token
        except subprocess.CalledProcessError as exc:
            raise LLMError(
                f"gcloud auth print-identity-token failed (exit {exc.returncode}). "
                "Run: gcloud auth login"
            ) from exc
        except FileNotFoundError as exc:
            raise LLMError(
                f"gcloud binary not found at '{self._settings.gcloud_path}'. "
                "Set GCLOUD_PATH in .env to the correct path."
            ) from exc

    def _headers(self) -> dict:
        """Base headers for all GENI requests (no Content-Type — let requests set it)."""
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "x-iag-domain": self._domain,
        }

    # ── CSV file management ────────────────────────────────────────────────

    def _ensure_csv_uploaded(self) -> str:
        """Return the cached CSV file UUID, uploading it first if needed.

        The upload only happens once per adapter instance (or per process if
        GENI_CSV_FILE_ID is set in .env, in which case it never uploads).

        Returns:
            GENI file UUID string.
        """
        if self._csv_file_id:
            return self._csv_file_id
        csv_path = self._settings.master_csv_path
        logger.info("GeniLLMAdapter: uploading CSV file '%s' to GENI...", csv_path)
        self._csv_file_id = self._upload_file(csv_path)
        logger.info(
            "GeniLLMAdapter: CSV uploaded successfully — file_id=%s "
            "(set GENI_CSV_FILE_ID=%s in .env to skip future uploads)",
            self._csv_file_id,
            self._csv_file_id,
        )
        return self._csv_file_id

    def _upload_file(self, path: Path) -> str:
        """Upload a local file to GENI via POST /api/files.

        Args:
            path: Local filesystem path to the file.

        Returns:
            GENI file UUID (``file.id`` from the response).

        Raises:
            LLMError: On non-2xx response or missing id field.
        """
        _t0 = time.perf_counter()
        with open(path, "rb") as fh:
            resp = requests.post(
                f"{self._base_url}/api/files",
                headers=self._headers(),  # no Content-Type — requests sets multipart
                files={"file": (path.name, fh, "text/csv")},
                timeout=60,
            )
        _raise_for_status(resp, "upload file")
        payload = resp.json()
        file_id: str = payload["file"]["id"]
        logger.info(
            "⏱ [GeniLLM] operation=upload_csv elapsed=%.3fs file_id=%s",
            time.perf_counter() - _t0,
            file_id,
        )
        logger.debug("GENI file upload response: %s", payload.get("file", {}))
        return file_id

    # ── GENI conversation flow ─────────────────────────────────────────────

    def _create_conversation(self) -> str:
        """Create a new single-use conversation tied to the bot version.

        Each classify call gets a fresh conversation — no cross-query
        contamination from conversation history.

        Returns:
            conversation_id string.
        """
        _t0 = time.perf_counter()
        resp = requests.post(
            f"{self._base_url}/api/crud/conversations",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={
                "conversation": {
                    "bot_version_id": self._bot_version_id,
                    "name": "anzsic-classification",
                }
            },
            timeout=30,
        )
        _raise_for_status(resp, "create conversation")
        conversation_id = resp.json()["conversation"]["id"]
        logger.info(
            "⏱ [GeniLLM] operation=create_conversation elapsed=%.3fs conversation_id=%s",
            time.perf_counter() - _t0,
            conversation_id,
        )
        return conversation_id

    def _post_question(
        self,
        conversation_id: str,
        content: str,
        file_ids: list[str] | None = None,
    ) -> str:
        """Post the user message as a question to the conversation.

        Args:
            conversation_id: ID from _create_conversation().
            content:         The full user message (query + candidates).
            file_ids:        Optional list of GENI file UUIDs to include in
                             the LLM prompt (e.g. the ANZSIC master CSV).

        Returns:
            question_id string.
        """
        question_payload: dict = {"content": content}
        if file_ids:
            question_payload["file_ids"] = file_ids

        _t0 = time.perf_counter()
        resp = requests.post(
            f"{self._base_url}/api/crud/conversations/{conversation_id}/questions",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"question": question_payload},
            timeout=30,
        )
        _raise_for_status(resp, "post question")
        question_id = resp.json()["question"]["id"]
        logger.info(
            "⏱ [GeniLLM] operation=post_question elapsed=%.3fs question_id=%s",
            time.perf_counter() - _t0,
            question_id,
        )
        return question_id

    def _poll_for_answer(self, question_id: str) -> str:
        """Poll the status endpoint until the answer is ready.

        Args:
            question_id: ID from _post_question().

        Returns:
            Raw answer string (JSON) from GENI.

        Raises:
            LLMError: If GENI returns an error in the answer payload.
            TimeoutError: If the answer does not arrive within poll_timeout.
        """
        deadline = time.monotonic() + self._poll_timeout
        _t0 = time.perf_counter()
        _poll_count = 0

        while time.monotonic() < deadline:
            _poll_count += 1
            status_resp = requests.get(
                f"{self._base_url}/api/crud/questions/{question_id}/status",
                headers=self._headers(),
                timeout=15,
            )
            _raise_for_status(status_resp, "poll status")

            if status_resp.json().get("has_answer"):
                _t_fetch = time.perf_counter()
                answer_resp = requests.get(
                    f"{self._base_url}/api/crud/questions/{question_id}",
                    headers=self._headers(),
                    timeout=15,
                )
                _raise_for_status(answer_resp, "fetch answer")

                answer_obj = answer_resp.json().get("answer", {})
                if answer_obj.get("error"):
                    raise LLMError(f"GENI returned error: {answer_obj['error']}")

                # GENI returns answer text in "content", not "answer"
                raw = answer_obj.get("content", "")
                _elapsed = time.perf_counter() - _t0
                logger.info(
                    "⏱ [GeniLLM] operation=poll_for_answer elapsed=%.3fs "
                    "poll_iterations=%d fetch_elapsed=%.3fs answer_chars=%d "
                    "question_id=%s",
                    _elapsed,
                    _poll_count,
                    time.perf_counter() - _t_fetch,
                    len(raw),
                    question_id,
                )
                return raw

            time.sleep(self._poll_interval)

        raise LLMError(
            f"GENI did not answer within {self._poll_timeout}s "
            f"(question_id={question_id})"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _raise_for_status(resp: requests.Response, context: str) -> None:
    """Raise LLMError with context on non-2xx responses."""
    if not resp.ok:
        raise LLMError(
            f"GENI API error during '{context}': "
            f"HTTP {resp.status_code} — {resp.text[:200]}"
        )
