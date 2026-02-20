# ADR 002 — pgvector Hybrid Search (Vector + FTS in PostgreSQL)

<span class="adr-status-accepted">Accepted</span> &nbsp; **Date:** 2025 &nbsp; **Deciders:** CEP AI Team

---

## Context

We needed a retrieval backend that could:

- Store 5,236 ANZSIC records with 768-dimensional embeddings
- Perform fast approximate nearest-neighbour (ANN) search on the embeddings
- Perform keyword / full-text search on the descriptions
- Run on developer laptops and a corporate Mac without external services
- Be replaced in the future if requirements change

Candidates evaluated:

| Option | Vector search | FTS | Self-hosted | Notes |
|---|---|---|---|---|
| **PostgreSQL + pgvector** | ✅ HNSW | ✅ tsvector | ✅ | Already installed; SQL familiarity |
| **Weaviate** | ✅ HNSW | ✅ BM25 | ✅ | Docker dependency; extra ops overhead |
| **Pinecone** | ✅ | ❌ (no FTS) | ❌ SaaS | Corporate data policy concerns; cost |
| **Qdrant** | ✅ | Partial | ✅ | Newer; less community docs |
| **Elasticsearch** | Partial | ✅ | ✅ | Heavy; no native pgvector interop |

---

## Decision

We chose **PostgreSQL 15 + pgvector 0.8.0**.

Reasons:

1. **Single dependency** — PostgreSQL was already running on developer machines.
   Adding pgvector required one `CREATE EXTENSION vector` command.

2. **Both search types in one query engine** — The FTS (`tsvector` + GIN index)
   and vector search (HNSW index via `<=>`) live in the same database. There is
   no data synchronisation, no dual writes, no consistency lag.

3. **HNSW performance** — At 5,236 rows, HNSW is fast enough that latency is
   dominated by the Vertex AI API calls, not the database. The HNSW index
   (m=16, ef_construction=64) returns results in < 10 ms.

4. **Corporate data policy** — ANZSIC data and query logs stay on-premises.
   No external SaaS dependency.

5. **SQL familiarity** — The entire team can inspect, debug, and extend the
   retrieval queries with standard SQL knowledge.

---

## Database schema (key columns)

```sql
CREATE TABLE anzsic_codes (
    anzsic_code      TEXT PRIMARY KEY,
    anzsic_desc      TEXT,
    class_code       TEXT,
    class_desc       TEXT,
    group_code       TEXT,
    group_desc       TEXT,
    subdivision_desc TEXT,
    division_desc    TEXT,
    class_exclusions TEXT,
    enriched_text    TEXT,
    embedding        vector(768),   -- pgvector HNSW
    fts_vector       tsvector       -- GIN full-text
);

CREATE INDEX ON anzsic_codes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX ON anzsic_codes USING gin (fts_vector);
```

---

## Consequences

**Positive:**

- Zero additional infrastructure beyond what already existed
- SQL debugging: `EXPLAIN ANALYZE` works on vector queries
- Both search systems share the same transaction semantics

**Negative / trade-offs:**

- PostgreSQL is not purpose-built for vector search; Weaviate or Qdrant would
  be faster at 10M+ records
- HNSW is an approximate index — exact nearest-neighbour is not guaranteed
  (acceptable for classification; we fuse with FTS anyway)
- pgvector compilation required `PG_SYSROOT` override on macOS Sequoia
  (documented in `ingest.py` header)

**When to revisit:**

Migrate to a dedicated vector database (Weaviate/Qdrant) when the code
catalogue grows beyond ~50,000 records or when sub-5ms p99 vector search
latency becomes a requirement.

The `DatabasePort` abstraction makes this migration a one-file change.
