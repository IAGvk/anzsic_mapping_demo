# ADR 005 — Gemini for LLM Re-ranking (Stage 2)

<span class="adr-status-accepted">Accepted</span> &nbsp; **Date:** 2025 &nbsp; **Deciders:** CEP AI Team

---

## Context

Stage 2 takes the top-20 Stage 1 candidates and asks an LLM to:

- Read each candidate's description, class, group, division, and exclusions
- Select the top-5 best matches ranked by relevance
- Provide a plain-English reason for each selection

This task requires a model that can:

- Follow a strict JSON output schema
- Reason about subtle distinctions (e.g. "own account" vs "employed" mechanics)
- Process ~20 candidate records (~3,000 tokens) in a single call
- Return results within a 5–10 second budget

Models evaluated:

| Model | Context window | JSON mode | Region | Notes |
|---|---|---|---|---|
| **Gemini 2.5 Flash** | 1M tokens | ✅ native | `australia-southeast1` | GCP auth reused |
| GPT-4o | 128K tokens | ✅ native | US/EU only | Separate API key; data residency concern |
| GPT-4o-mini | 128K tokens | ✅ | US/EU only | Cheaper but less accurate |
| Claude 3.5 Sonnet | 200K tokens | ✅ (tool use) | US/EU only | No AUS region |
| Gemini 2.0 Flash 001 | 1M tokens | ✅ | `australia-southeast1` | Model unavailable in region (tested, failed) |

---

## Decision

We chose **Vertex AI Gemini 2.5 Flash** (`gemini-2.5-flash`).

Reasons:

1. **Existing GCP auth reused** — The same `GCPAuthManager` and bearer token
   used for embeddings serves Gemini. No additional secrets management.

2. **`australia-southeast1` region** — Data stays in Australian infrastructure.
   (Note: `gemini-2.0-flash-001` was tested first but was unavailable in this
   region — see below.)

3. **Native JSON mode** — `responseMimeType: application/json` guarantees
   syntactically valid JSON output, eliminating the need to strip markdown
   fences or handle malformed responses.

4. **1M token context window** — The CSV fallback strategy (injecting all 5,236
   codes, ~63K tokens) fits comfortably within the context budget.

5. **Speed** — Flash-tier models balance quality and latency better than
   Pro-tier for this structured re-ranking task.

---

## Model availability issue (resolved)

During development, `gemini-2.0-flash-001` returned HTTP 404 for all requests
to `australia-southeast1`. Investigation confirmed the model was not deployed
in that region at the time. We switched to `gemini-2.5-flash` which is
available. The model is configured via the `GCP_GEMINI_MODEL` environment
variable — changing model requires zero code changes.

---

## CSV fallback strategy

The standard re-ranking prompt contains only the 20 Stage 1 candidates
(~2K tokens). On rare queries where Gemini returns an empty result array
(i.e. no candidate matched), we retry with all 5,236 ANZSIC codes injected
as a reference lookup (~63K tokens).

**Why retry-on-empty rather than always injecting?**

- The 63K-token prompt was initially always sent, but this caused token limit
  errors (148K > 128K context for an older model)
- Even with the new 1M context model, a 65K-token prompt degrades Gemini's
  attention on the 20 relevant candidates
- Testing showed > 95% of queries succeed on the first (compact) call
- The retry adds latency only for the rare case where it is genuinely needed

This strategy is implemented in `LLMReranker._call_llm()` and tested in
`prod/tests/unit/test_reranker.py::TestRerankerCsvFallback`.

---

## Consequences

**Positive:**

- GCP auth shared with embeddings — one token, two adapters
- Temperature 0.1 produces consistent, reproducible re-rankings across runs
- Provenance: `llm_model` field in `ClassifyResponse` records which model
  produced the result — important for reproducibility audits

**Negative / trade-offs:**

- Stage 2 adds 2–5 seconds of latency — Fast mode (Stage 1 only) is available
  when speed matters more than explanation quality
- Gemini API availability affects production reliability — Fast mode is the
  graceful degradation path

**When to revisit:**

The `LLMPort` abstraction makes swapping Gemini for another model a one-line
change in `container.py`. Good candidates for future evaluation:
GPT-4o (if data residency requirements relax) or a fine-tuned smaller model
(if Gemini API costs become a concern at scale).
