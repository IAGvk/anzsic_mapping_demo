# ADR 006 — Multi-Provider Architecture (Vertex AI / OpenAI)

<span class="adr-status-accepted">Accepted</span> &nbsp; **Date:** 2025 &nbsp; **Deciders:** CEP AI Team

---

## Context

The initial implementation used Vertex AI for both embeddings and LLM re-ranking.
This restricted the tool to users who have:

1. A GCP project with Vertex AI APIs enabled
2. `gcloud` installed and authenticated
3. Network access to `australia-southeast1-aiplatform.googleapis.com`

This covered the AI team developers but excluded the broader organisation —
business analysts, data engineers, and other teams who have an OpenAI API key
but no GCP access.

The Hexagonal Architecture already provided `EmbeddingPort` and `LLMPort`
abstractions. The question was: how to let users choose their provider
without modifying any code?

---

## Decision

We added **environment-variable-driven provider selection** to `container.py`:

```bash
# GCP users (default — no changes required)
EMBED_PROVIDER=vertex
LLM_PROVIDER=vertex

# OpenAI users
EMBED_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Mix-and-match is valid
EMBED_PROVIDER=openai
LLM_PROVIDER=vertex
```

Two new adapters implement the existing Ports:

| New file | Port implemented | Technology |
|---|---|---|
| `adapters/openai_embedding.py` | `EmbeddingPort` | OpenAI `/v1/embeddings` |
| `adapters/openai_llm.py` | `LLMPort` | OpenAI `/v1/chat/completions` |

`container.py` was refactored into two private factory functions
(`_build_embedder`, `_build_llm`) that read the provider env vars and
return the appropriate adapter. `GCPAuthManager` is only instantiated when
at least one GCP adapter is active — OpenAI-only users incur zero GCP
import overhead.

No other file changed.

---

## Key design choices

### No OpenAI SDK dependency

Both adapters use `requests` directly, consistent with the existing Vertex AI
adapters. This keeps the dependency footprint minimal — `openai` Python package
would add ~80 MB of transitive dependencies for functionality we don't need
(streaming, assistants API, etc.).

### JSON mode parity

| Provider | JSON guarantee mechanism |
|---|---|
| Gemini (Vertex) | `responseMimeType: application/json` |
| GPT-4o (OpenAI) | `response_format: {"type": "json_object"}` |

Both mechanisms guarantee syntactically valid JSON without post-processing.
The `LLMReranker._parse_response()` function is unchanged — it already
handles the `None` / empty / invalid-JSON cases defensively.

### Dimension flexibility

OpenAI's `text-embedding-3` models accept a `dimensions` parameter.
`OpenAIEmbeddingAdapter` always passes `dimensions=settings.embed_dim`,
making output compatible with whatever pgvector column the database
was initialised with. If the database was set up with Vertex AI's 768-dim
vectors, set `EMBED_DIM=768` — OpenAI will reduce its output accordingly.

### No prompt changes

The existing `build_system_prompt()` in `config/prompts.py` contains the
word "JSON" — satisfying OpenAI JSON mode's requirement that the system
prompt explicitly mention JSON. No prompt modifications were needed.

---

## Consequences

**Positive:**

- Any user with an `OPENAI_API_KEY` can now run the full pipeline
- Zero code changes needed to switch providers — only `.env` changes
- Mix-and-match allows hybrid setups (e.g. OpenAI embeddings + Gemini LLM
  for users migrating incrementally)
- GCP credentials are never required for OpenAI-only deployments
- 35 new unit tests cover both adapters, retry logic, and provider routing

**Negative / trade-offs:**

- OpenAI and Vertex AI embeddings live in **different vector spaces** —
  a database populated with Vertex AI embeddings cannot be queried with
  OpenAI embeddings (the cosine similarities would be meaningless)
- A user switching providers must re-embed and re-ingest all 5,236 records
  with the new embedding model
- Two providers means two sets of API credentials to manage in production

**Neutral:**

- `gpt-4o` is not in Australian infrastructure; if data residency for the
  LLM prompt is required, keep `LLM_PROVIDER=vertex`

**When to add a third provider:**

The `_build_embedder` / `_build_llm` factory functions in `container.py`
use a simple `if/else` chain. Adding Cohere, Azure OpenAI, or any other
provider requires only:
1. A new adapter file implementing the relevant Port
2. An additional `elif` branch in the factory function
