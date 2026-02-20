"""
tests/integration/test_gemini_llm.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for GeminiLLMAdapter.

Requires:
  • Active gcloud auth
  • Network access to Vertex AI
  • GCP_PROJECT_ID, GCP_LOCATION_ID, GCP_GEMINI_MODEL set

Run with:
  pytest -m integration prod/tests/integration/test_gemini_llm.py -v

IMPORTANT: These tests make real API calls and incur GCP costs.
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration

_SYSTEM_PROMPT = (
    "You are an ANZSIC classification assistant. "
    "Return a JSON array of objects with fields: rank, anzsic_code, anzsic_desc, reason."
)

_USER_PROMPT = (
    'Classify the occupation "mobile mechanic" into ANZSIC codes. '
    "Return top 2 results as a JSON array."
)


@pytest.fixture(scope="module")
def llm():
    from prod.adapters.gcp_auth import GCPAuthManager
    from prod.adapters.gemini_llm import GeminiLLMAdapter
    from prod.config.settings import get_settings
    settings = get_settings()
    auth = GCPAuthManager(settings)
    return GeminiLLMAdapter(auth, settings)


class TestGenerateJson:
    def test_returns_string(self, llm):
        raw = llm.generate_json(_SYSTEM_PROMPT, _USER_PROMPT)
        assert raw is None or isinstance(raw, str)

    def test_response_is_valid_json(self, llm):
        raw = llm.generate_json(_SYSTEM_PROMPT, _USER_PROMPT)
        assert raw is not None, "Gemini returned None — check auth and model config"
        parsed = json.loads(raw)  # Must not raise
        assert isinstance(parsed, (list, dict))

    def test_response_contains_anzsic_code(self, llm):
        raw = llm.generate_json(_SYSTEM_PROMPT, _USER_PROMPT)
        assert raw is not None
        parsed = json.loads(raw)
        items = parsed if isinstance(parsed, list) else next(
            (v for v in parsed.values() if isinstance(v, list)), []
        )
        assert len(items) > 0
        # At least one item should have an anzsic_code-like field
        assert any("anzsic" in str(item).lower() for item in items)

    def test_model_name_property(self, llm):
        assert len(llm.model_name) > 0
