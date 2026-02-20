# ADR 001 — Hexagonal Architecture (Ports & Adapters)

<span class="adr-status-accepted">Accepted</span> &nbsp; **Date:** 2025 &nbsp; **Deciders:** CEP AI Team

---

## Context

We needed a production-grade code structure for an AI classification pipeline
that would:

- Be easy to test without live GCP or database connections
- Allow the embedding model, LLM, and database to be swapped independently
- Scale from a Streamlit prototype to a REST API service without a rewrite
- Communicate *why* architectural decisions were made to future maintainers

We evaluated three structural patterns:

| Pattern | Description |
|---|---|
| **A — Hexagonal (Ports & Adapters)** | Domain at centre; infrastructure at edges; Protocols as contracts |
| **B — Layered (N-tier)** | Presentation → Service → Repository → Infrastructure |
| **C — Feature Slice** | One folder per feature (classify/, ingest/, etc.) |

---

## Decision

We chose **Hexagonal Architecture (Option A)**.

The decisive factors were:

1. **Testability without infrastructure** — Port Protocols let us write mock
   adapters that satisfy contracts without inheritance. 65 unit tests run in
   0.10 seconds with no GCP or PostgreSQL connection.

2. **Single-line component swaps** — `services/container.py` is the *only* file
   that names concrete adapter classes. Replacing Gemini with GPT-4o, or
   PostgreSQL with Weaviate, requires changing one import line in one file.

3. **Natural FastAPI evolution** — Hexagonal architecture treats the web layer
   as just another interface adapter. Adding `interfaces/api.py` requires zero
   changes to services, adapters, or domain.

4. **Enforced dependency direction** — The rule `Services → Ports ← Adapters`
   is structurally enforced: if a service accidentally imports an adapter class,
   the circular dependency becomes immediately visible.

---

## Consequences

**Positive:**

- All business logic (RRF fusion, CSV fallback, search mode routing) is testable
  in pure Python with no I/O
- The `prod/` folder structure is self-documenting — the layer a file belongs to
  is its directory name
- New developers can understand the data flow by reading the domain models alone

**Negative / trade-offs:**

- More files than a simple script (`anzsic_agent.py` was 407 lines in one file;
  the `prod/` equivalent spans ~15 files)
- Requires discipline: developers must not import adapters directly in services
- The `container.py` indirection is unfamiliar to developers who have only worked
  with flat scripts

**Neutral:**

- The original `anzsic_agent.py` and `app.py` remain untouched — they continue
  to work. The `prod/` folder is an additive layer, not a replacement.
