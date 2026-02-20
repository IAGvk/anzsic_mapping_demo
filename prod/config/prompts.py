"""
config/prompts.py
──────────────────────────────────────────────────────────────────────────────
All LLM prompt strings in one place.

Why centralise prompts?
  • Easy to diff and review prompt changes in version control
  • Swap or tune a prompt without touching service logic
  • Single source for prompt versioning / A-B testing

To change the re-ranking prompt: edit RERANK_SYSTEM_BASE below.
To support a different output schema: change RERANK_OUTPUT_SCHEMA.
"""
from __future__ import annotations

# ── Re-ranking system prompt ───────────────────────────────────────────────────
RERANK_SYSTEM_BASE = """\
You are an expert ANZSIC (Australian and New Zealand Standard Industrial \
Classification) coder.
Your job is to match a poorly-worded occupation description provided by a \
non-expert to the correct ANZSIC occupation codes.

You will be given:
1. The user's raw input description
2. A list of candidate ANZSIC codes retrieved by a search system (each with \
its description, class, group, subdivision, division, and a "NOT included" \
exclusion note)

Your task:
- Carefully read each candidate.
- Use the "Not included" exclusion text to ELIMINATE candidates that are \
explicitly ruled out.
- Select the TOP 5 best-matching codes, ranked from most to least likely.
- For each selected code provide a short plain-English reason (1–2 sentences) \
explaining WHY it matches (or why you ranked it above other options).
- If fewer than 5 candidates genuinely match, return fewer — do not pad with \
poor matches.

Respond ONLY with a JSON array of objects in this exact schema \
(no markdown fences):
[
  {
    "rank": 1,
    "anzsic_code": "X1234_56",
    "anzsic_desc": "...",
    "class_desc": "...",
    "division_desc": "...",
    "reason": "..."
  },
  ...
]
"""

# ── CSV reference section (appended when retrieval confidence is low) ──────────
CSV_REFERENCE_HEADER = """\

{divider}
FULL ANZSIC REFERENCE — the candidate list above may be insufficient.
All 5,236 codes are listed below as:  CODE: description
Use this reference to find a better match if none of the candidates fit.
{divider}
"""

# ── User message template ──────────────────────────────────────────────────────
RERANK_USER_TEMPLATE = """\
User input: "{query}"

Candidates ({n_candidates} total):
{candidate_block}

Return the top {top_k} matches as a JSON array.\
"""

# ── Candidate block line template ──────────────────────────────────────────────
CANDIDATE_BLOCK_TEMPLATE = """\
[{idx}] Code: {anzsic_code}
    Occupation: {anzsic_desc}
    Class: {class_desc}
    Group: {group_desc}
    Subdivision: {subdivision_desc}
    Division: {division_desc}
"""

CANDIDATE_EXCLUSION_LINE = "    Not included: {exclusions}\n"


def build_system_prompt(include_reference: bool, csv_reference: str) -> str:
    """Assembles the Gemini system prompt, optionally appending the full CSV.

    Args:
        include_reference: When True, appends all 5,236 ANZSIC codes as a
                           fallback lookup table for low-confidence queries.
        csv_reference:     Pre-loaded CSV reference string (code: desc lines).

    Returns:
        Complete system prompt string ready to send to the LLM.
    """
    if not include_reference or not csv_reference:
        return RERANK_SYSTEM_BASE

    divider = "─" * 77
    header = CSV_REFERENCE_HEADER.format(divider=divider)
    return RERANK_SYSTEM_BASE + header + csv_reference


def build_candidate_block(candidates: list[dict]) -> str:
    """Renders the numbered candidate list for the LLM user message.

    Args:
        candidates: List of candidate dicts (from Candidate.model_dump()).

    Returns:
        Formatted multi-line string.
    """
    lines = []
    for i, c in enumerate(candidates, 1):
        block = CANDIDATE_BLOCK_TEMPLATE.format(
            idx=i,
            anzsic_code=c.get("anzsic_code", ""),
            anzsic_desc=c.get("anzsic_desc", ""),
            class_desc=c.get("class_desc", ""),
            group_desc=c.get("group_desc", ""),
            subdivision_desc=c.get("subdivision_desc", ""),
            division_desc=c.get("division_desc", ""),
        )
        if c.get("class_exclusions"):
            block += CANDIDATE_EXCLUSION_LINE.format(
                exclusions=c["class_exclusions"]
            )
        lines.append(block)
    return "\n".join(lines)


def build_user_message(query: str, candidates: list[dict], top_k: int) -> str:
    """Assembles the user-turn message for the LLM.

    Args:
        query:      Raw input description from the user.
        candidates: List of candidate dicts.
        top_k:      Number of results to request from the LLM.

    Returns:
        Formatted user message string.
    """
    return RERANK_USER_TEMPLATE.format(
        query=query,
        n_candidates=len(candidates),
        candidate_block=build_candidate_block(candidates),
        top_k=top_k,
    )
