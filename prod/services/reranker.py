"""
services/reranker.py
──────────────────────────────────────────────────────────────────────────────
Stage 2 of the classification pipeline: LLM Re-ranking.

Responsibilities:
  1. Build a structured prompt from Stage 1 candidates (via config/prompts.py).
  2. Call the LLMPort to generate a ranked JSON response.
  3. Parse and validate the JSON into ClassifyResult objects.
  4. CSV fallback: if Gemini returns an empty list, retry with the full
     ANZSIC master CSV injected into the system prompt.

Fallback strategy:
  - Normal call first (no CSV reference) — keeps the prompt concise.
  - If results list is empty, log a warning and retry with CSV injected.
  - Two failed attempts → return empty list (caller handles gracefully).

The CSV reference is loaded ONCE at construction time and reused across
all classify calls (amortised startup cost).
"""
from __future__ import annotations

import json
import logging
import csv
from pathlib import Path

from prod.config.prompts import build_system_prompt, build_user_message
from prod.config.settings import Settings
from prod.domain.exceptions import RerankError
from prod.domain.models import Candidate, ClassifyResult
from prod.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class LLMReranker:
    """Re-rank Stage 1 candidates using an LLM.

    Args:
        llm:      Any object satisfying LLMPort.
        settings: Shared application settings.
    """

    def __init__(self, llm: LLMPort, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings
        self._csv_reference = self._load_csv_reference()
        has_ref = bool(self._csv_reference)
        logger.debug(
            "LLMReranker init | model=%s csv_reference_loaded=%s",
            llm.model_name,
            has_ref,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> list[ClassifyResult]:
        """Re-rank Stage 1 candidates using the LLM.

        Args:
            query:      Original search query.
            candidates: Ordered list of Stage 1 Candidate objects.
            top_k:      How many results to return.

        Returns:
            Ordered list of ClassifyResult objects (best match first).
            Empty list if both LLM attempts fail.
        """
        if not candidates:
            logger.warning("LLMReranker.rerank called with no candidates")
            return []

        # ── Attempt 1: normal call ─────────────────────────────────────────
        results = self._call_llm(query, candidates, top_k, include_reference=False)
        if results:
            return results

        # ── Attempt 2: retry with CSV reference injected ───────────────────
        logger.warning(
            "Gemini returned empty results for %r — retrying with CSV fallback", query
        )
        results = self._call_llm(query, candidates, top_k, include_reference=True)
        if results:
            logger.info("CSV fallback succeeded for %r", query)
            return results

        logger.error("LLMReranker: both attempts failed for query %r", query)
        return []

    # ── Private helpers ────────────────────────────────────────────────────

    def _call_llm(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int,
        include_reference: bool,
    ) -> list[ClassifyResult]:
        """Build prompt, call LLM, parse response."""
        candidate_dicts = [c.model_dump() for c in candidates]
        system = build_system_prompt(
            include_reference=include_reference,
            csv_reference=self._csv_reference,
        )
        user = build_user_message(query, candidate_dicts, top_k)

        raw = self._llm.generate_json(system, user)
        if not raw:
            return []
        return self._parse_response(raw, top_k)

    def _parse_response(self, raw: str, top_k: int) -> list[ClassifyResult]:
        """Parse the LLM JSON response into ClassifyResult objects.

        The model may return a bare JSON array or an object wrapping one.
        Both formats are handled gracefully.

        Returns:
            List of ClassifyResult objects (empty on parse failure).
        """
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.error("LLMReranker: failed to parse JSON: %.200s", raw)
            return []

        # Unwrap if the model returned {"results": [...]} or similar
        if isinstance(parsed, dict):
            items = next(
                (v for v in parsed.values() if isinstance(v, list)),
                [],
            )
        elif isinstance(parsed, list):
            items = parsed
        else:
            logger.error("LLMReranker: unexpected JSON type %s", type(parsed).__name__)
            return []

        results: list[ClassifyResult] = []
        for item in items[:top_k]:
            try:
                results.append(ClassifyResult(**item))
            except Exception as exc:
                logger.warning("Skipping malformed result item %s: %s", item, exc)

        return results

    def _load_csv_reference(self) -> str:
        """Load the ANZSIC master CSV as a compact CODE: description string.

        Loads only anzsic_code + anzsic_desc to keep token count low.
        Returns empty string if the file is missing (fallback is simply skipped).
        """
        csv_path = Path(self._settings.master_csv_path)
        if not csv_path.exists():
            logger.warning(
                "master_csv_path not found: %s — CSV fallback disabled", csv_path
            )
            return ""

        try:
            lines: list[str] = []
            with csv_path.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    code = (row.get("anzsic_code") or "").strip()
                    desc = (row.get("anzsic_desc") or "").strip()
                    if code and desc:
                        lines.append(f"{code}: {desc}")
            reference = "\n".join(lines)
            logger.info(
                "CSV reference loaded: %d entries (%d chars)",
                len(lines),
                len(reference),
            )
            return reference
        except Exception as exc:
            logger.error("Failed to load CSV reference: %s", exc)
            return ""
