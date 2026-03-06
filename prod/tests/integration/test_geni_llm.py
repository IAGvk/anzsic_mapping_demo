"""
tests/integration/test_geni_llm.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for GeniLLMAdapter.

Requires:
  • Active gcloud auth  (run: gcloud auth login)
  • GENI_BOT_VERSION_ID set in .env  (retrieve via curl — see .env.example)
  • LLM_PROVIDER=geni in .env
  • Network access to GENI platform

Run with:
  pytest -m integration prod/tests/integration/test_geni_llm.py -v

IMPORTANT: These tests make real GENI API calls.
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration

# ── Shared test prompts (mirrors test_gemini_llm.py) ──────────────────────────

# NOTE: system_prompt is passed but intentionally IGNORED by GeniLLMAdapter —
# the GENI bot already holds the instructions. We pass RERANK_SYSTEM_BASE here
# to test that the adapter is robust to receiving it (matches production flow).
_SYSTEM_PROMPT = (
    "You are an ANZSIC classification assistant. "
    "Return a JSON array of objects with fields: rank, anzsic_code, anzsic_desc, reason."
)

_USER_PROMPT = (
    'Classify the occupation "mobile mechanic" into ANZSIC codes. '
    "Return top 2 results as a JSON array."
)

# A user prompt with a realistic candidate block (matches production format)
_USER_PROMPT_WITH_CANDIDATES = """\
User input: "mobile mechanic"

Candidates (3 total):

[1] Code: S9419_03
    Occupation: Automotive Repair and Maintenance (own account)
    Class: Other Repair and Maintenance
    Group: Repair and Maintenance
    Subdivision: Repair and Maintenance
    Division: Other Services

[2] Code: S9411_01
    Occupation: Automotive Electrical Services
    Class: Automotive Repair and Maintenance
    Group: Repair and Maintenance
    Subdivision: Repair and Maintenance
    Division: Other Services

[3] Code: P7411_01
    Occupation: Motor Vehicle Parts and Accessories Retailing
    Class: Motor Vehicle Parts and Accessories Retailing
    Group: Motor Vehicle Retailing
    Subdivision: Motor Vehicle and Motor Vehicle Parts Retailing
    Division: Retail Trade
    Not included: New motor vehicle retailing

Return the top 2 matches as a JSON array.\
"""


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def llm():
    """Build a real GeniLLMAdapter using settings from .env."""
    from prod.adapters.geni_llm import GeniLLMAdapter
    from prod.config.settings import get_settings
    settings = get_settings()
    return GeniLLMAdapter(settings)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGeniLLMAdapter:

    def test_model_name_property(self, llm):
        """model_name should include 'geni' and the bot_version_id."""
        name = llm.model_name
        assert "geni" in name.lower()
        assert len(name) > 0

    def test_returns_string(self, llm):
        """generate_json should return a non-empty string."""
        raw = llm.generate_json(_SYSTEM_PROMPT, _USER_PROMPT)
        assert raw is not None, (
            "GeniLLMAdapter returned None — check auth (gcloud auth login) "
            "and GENI_BOT_VERSION_ID in .env"
        )
        assert isinstance(raw, str)
        assert len(raw) > 0

    def test_response_is_valid_json(self, llm):
        """GENI bot (JSON output mode) must return parseable JSON."""
        raw = llm.generate_json(_SYSTEM_PROMPT, _USER_PROMPT)
        assert raw is not None, "GENI returned None"
        parsed = json.loads(raw)  # Raises on invalid JSON
        assert isinstance(parsed, (list, dict)), (
            f"Expected JSON list or dict, got {type(parsed).__name__}: {raw[:200]}"
        )

    def test_response_contains_anzsic_fields(self, llm):
        """Response items should contain expected ANZSIC classification fields."""
        raw = llm.generate_json(_SYSTEM_PROMPT, _USER_PROMPT_WITH_CANDIDATES)
        assert raw is not None

        parsed = json.loads(raw)
        items = parsed if isinstance(parsed, list) else next(
            (v for v in parsed.values() if isinstance(v, list)), []
        )

        assert len(items) > 0, f"Expected at least 1 result, got empty list. Raw: {raw[:300]}"

        first = items[0]
        assert "anzsic_code" in first, f"Missing 'anzsic_code' in: {first}"
        assert "anzsic_desc" in first, f"Missing 'anzsic_desc' in: {first}"
        assert "rank" in first, f"Missing 'rank' in: {first}"

    def test_system_prompt_ignored_gracefully(self, llm):
        """Passing a non-empty system_prompt must not raise — it is simply ignored."""
        raw = llm.generate_json(
            system_prompt="This system prompt should be silently ignored by GENI.",
            user_message=_USER_PROMPT_WITH_CANDIDATES,
        )
        assert raw is not None

    def test_csv_sentinel_rerouted_to_user_message(self, llm):
        """When system_prompt contains the CSV sentinel, adapter appends it to
        user message instead of dropping it — verify no exception is raised
        and a valid response is still returned."""
        csv_fallback_system = (
            _SYSTEM_PROMPT
            + "\n\n────────────────────────────────────────────────────────"
            + "\nFULL ANZSIC REFERENCE — the candidate list above may be insufficient.\n"
            + "S9419_03: Automotive Repair and Maintenance (own account)\n"
            + "S9411_01: Automotive Electrical Services\n"
        )
        raw = llm.generate_json(csv_fallback_system, _USER_PROMPT_WITH_CANDIDATES)
        assert raw is not None
        parsed = json.loads(raw)
        assert isinstance(parsed, (list, dict))
