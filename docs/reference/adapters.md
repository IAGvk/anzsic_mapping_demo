# Adapters

Adapters are the concrete implementations of the Ports. Each adapter binds
the system to one specific external technology.

!!! warning "Container is the only entry point"
    Never import an adapter class directly in your application code.
    Always go through `services/container.py` — that is the single wiring
    point where adapters are named. This preserves swappability.

---

## Provider overview

Two providers are supported for both embedding and LLM. Selection is controlled
entirely by environment variables — no code changes required.

| Component | `EMBED_PROVIDER=vertex` (default) | `EMBED_PROVIDER=openai` |
|---|---|---|
| Embedding adapter | `VertexEmbeddingAdapter` | `OpenAIEmbeddingAdapter` |
| Auth required | gcloud ADC token | `OPENAI_API_KEY` |
| Model default | `text-embedding-005` | `text-embedding-3-small` |

| Component | `LLM_PROVIDER=vertex` (default) | `LLM_PROVIDER=openai` |
|---|---|---|
| LLM adapter | `GeminiLLMAdapter` | `OpenAILLMAdapter` |
| Auth required | gcloud ADC token | `OPENAI_API_KEY` |
| Model default | `gemini-2.5-flash` | `gpt-4o` |

Mix-and-match is supported (e.g. `EMBED_PROVIDER=openai LLM_PROVIDER=vertex`).

---

## GCPAuthManager

Shared across both GCP adapters (`VertexEmbeddingAdapter` and `GeminiLLMAdapter`).
Manages the bearer token lifecycle — fetching, caching, and refreshing — so
that both adapters share a single token rather than making separate `gcloud`
subprocess calls.

Only instantiated when `EMBED_PROVIDER=vertex` or `LLM_PROVIDER=vertex`.

::: prod.adapters.gcp_auth
    options:
      members:
        - GCPAuthManager

---

## VertexEmbeddingAdapter

Implements `EmbeddingPort` using the Vertex AI Predict REST API.

Key behaviours:

- `embed_query` uses `RETRIEVAL_QUERY` task type (asymmetric retrieval)
- `embed_document` uses `RETRIEVAL_DOCUMENT` task type
- Batch calls chunk to `embed_batch_size` (default 50) to stay within API limits
- HTTP 401 → invalidates the cached token and retries once
- HTTP 429 / 503 → exponential back-off with up to `embed_retries` attempts

::: prod.adapters.vertex_embedding
    options:
      members:
        - VertexEmbeddingAdapter

---

## GeminiLLMAdapter

Implements `LLMPort` using the Vertex AI `generateContent` REST API.

Key behaviours:

- Requests `responseMimeType: application/json` — guaranteed valid JSON output
- `temperature: 0.1` — low temperature for consistent, deterministic re-ranking
- HTTP 401 → token refresh + retry
- HTTP 429 / 503 → exponential back-off

::: prod.adapters.gemini_llm
    options:
      members:
        - GeminiLLMAdapter

---

## OpenAIEmbeddingAdapter

Implements `EmbeddingPort` using the OpenAI Embeddings REST API.
Activated when `EMBED_PROVIDER=openai`.

Key behaviours:

- Uses `/v1/embeddings` directly via `requests` (no SDK dependency)
- Passes `dimensions=EMBED_DIM` to the API so output matches the pgvector column width
- `embed_query` and `embed_document` use the same endpoint (OpenAI embeddings are symmetric)
- HTTP 401 → raises `AuthenticationError` immediately (key is wrong, no retry)
- HTTP 429 / 500 / 503 → exponential back-off

!!! note "Dimension compatibility"
    If your database was initialised with `vector(768)` (the Vertex AI default), set
    `EMBED_DIM=768` when using OpenAI models. `text-embedding-3-small` and
    `text-embedding-3-large` both accept a `dimensions` parameter to reduce their
    native output to any target size. For a fresh installation with OpenAI, setting
    `EMBED_DIM=1536` gives better quality from `text-embedding-3-small`.

::: prod.adapters.openai_embedding
    options:
      members:
        - OpenAIEmbeddingAdapter

---

## OpenAILLMAdapter

Implements `LLMPort` using the OpenAI Chat Completions REST API.
Activated when `LLM_PROVIDER=openai`.

Key behaviours:

- Uses `/v1/chat/completions` with `response_format: {"type": "json_object"}` —
  guarantees syntactically valid JSON output (same guarantee as Gemini's
  `responseMimeType: application/json`)
- `temperature: 0.1` — consistent with the Gemini adapter
- The existing `build_system_prompt()` already includes the word "JSON",
  satisfying OpenAI's JSON mode requirement with no prompt changes needed
- HTTP 401 → raises `AuthenticationError` immediately
- HTTP 429 / 500 / 503 → exponential back-off

::: prod.adapters.openai_llm
    options:
      members:
        - OpenAILLMAdapter

---

## PostgresDatabaseAdapter

Implements `DatabasePort` using `psycopg2` and the `pgvector` extension.

Key behaviours:

- Connection is opened lazily on first query and reused
- On `OperationalError`, reconnects once automatically
- `vector_search` uses the HNSW index via the `<=>` cosine operator
- `fts_search` uses the GIN-indexed `tsvector` column
- `fetch_by_codes` uses `ANY(%s)` for a single round-trip to fetch N records

!!! tip "Scaling to FastAPI"
    The current implementation uses a single persistent connection — appropriate
    for Streamlit (single process, single thread). For a multi-threaded FastAPI
    deployment, replace the connection management here with
    `psycopg2.pool.ThreadedConnectionPool`. No other file needs to change.

::: prod.adapters.postgres_db
    options:
      members:
        - PostgresDatabaseAdapter
