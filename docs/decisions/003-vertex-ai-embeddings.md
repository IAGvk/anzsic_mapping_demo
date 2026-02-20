# ADR 003 — Vertex AI Embeddings (text-embedding-005)

<span class="adr-status-accepted">Accepted</span> &nbsp; **Date:** 2025 &nbsp; **Deciders:** CEP AI Team

---

## Context

High-quality semantic embeddings are the backbone of Stage 1 retrieval.
The embedding model determines:

- Whether colloquial queries ("barista", "runs a café") match correctly
- The vector dimension (affects index size and memory)
- Latency and cost per query

We evaluated:

| Model | Dimensions | Task types | Auth | Latency |
|---|---|---|---|---|
| **Vertex AI text-embedding-005** | 768 | Asymmetric (QUERY / DOCUMENT) | gcloud ADC | ~150–200 ms |
| OpenAI text-embedding-3-large | 3072 | Symmetric | API key | ~80–120 ms |
| OpenAI text-embedding-3-small | 1536 | Symmetric | API key | ~50–80 ms |
| Cohere embed-v3 | 1024 | Asymmetric | API key | ~100–150 ms |
| Local (sentence-transformers) | 384–768 | Symmetric | None | ~5–50 ms |

---

## Decision

We chose **Vertex AI `text-embedding-005`**.

Reasons:

1. **Existing GCP auth infrastructure** — The organisation already uses GCP
   (`top-arc-65ca`, `australia-southeast1`). gcloud Application Default
   Credentials were already configured. No new API keys to manage.

2. **Asymmetric task types** — `RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT` task
   types optimise the embedding space for retrieval asymmetry. A short query
   embedding and a long document embedding are in compatible spaces. OpenAI's
   text-embedding-3 models do not support this distinction.

3. **Data residency** — `australia-southeast1` keeps embeddings generation
   within Australian GCP infrastructure, satisfying corporate data policy.

4. **768 dimensions** — Smaller than OpenAI's 3072-dim model, reducing
   HNSW index memory and cosine distance computation time, with negligible
   quality difference at this dataset size.

---

## Asymmetric retrieval explained

```
Query:    "mobile mechanic"           → RETRIEVAL_QUERY embedding
Documents: "Automotive Repair and    → RETRIEVAL_DOCUMENT embeddings
            Maintenance (own account)    (pre-computed at ingest time)
            Motor Vehicle Parts..."
```

The model is trained so that RETRIEVAL_QUERY vectors lie close to
RETRIEVAL_DOCUMENT vectors for semantically related text, even when the
surface forms are completely different. This is why "barista" finds
"Cafes and Restaurants" rather than just keyword matches.

---

## Consequences

**Positive:**

- Works within existing GCP auth (`gcloud auth application-default login`)
- Embeddings remain within Australian infrastructure
- `GCPAuthManager` is shared with Gemini — one token serves both adapters

**Negative / trade-offs:**

- Requires network access (or proxy) to `australia-southeast1-aiplatform.googleapis.com`
- Cannot run fully offline (unlike local sentence-transformers)
- gcloud subprocess for auth adds ~30 ms to the first call per session

**When to revisit:**

If offline operation becomes a requirement, swap in a local embedding model
via a new adapter. The `EmbeddingPort` abstraction makes this a one-file change.
`EMBED_DIM` in settings handles the dimension change transparently.
