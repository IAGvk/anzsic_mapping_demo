# ANZSIC Classifier — Test Plan

## Overview

This document describes the testing strategy for the production ANZSIC
classifier (`prod/`), built using Hexagonal Architecture (Ports & Adapters).

---

## 1. Testing Philosophy

### Core principle: test behaviour, not implementation

Tests assert on **observable outcomes** (output given input) rather than
internal implementation details (e.g. which SQL query was run).  This means:
- Mock adapters implement Port *protocols* — they are interchangeable with real ones
- Services are tested independently of infrastructure
- Infrastructure (DB, GCP) is tested independently of services

### The three layers

| Layer        | Test type        | Dependencies           | Speed  |
|--------------|------------------|------------------------|--------|
| Pure logic   | Unit             | None                   | < 1ms  |
| Services     | Unit (mocked)    | Mock adapters only     | < 10ms |
| Adapters     | Integration      | Real GCP / DB          | 1–5s   |
| Full pipeline| E2E (mocked)     | Mock adapters          | < 50ms |
| Full pipeline| E2E (live)       | Real GCP / DB          | 3–15s  |

---

## 2. Test Structure

```
prod/tests/
├── conftest.py                      # Shared fixtures, mock adapters
├── fixtures/
│   ├── sample_queries.txt           # 15 representative queries
│   └── golden_results.json          # Expected top codes per query
│
├── unit/                            # Fast, pure-logic tests (no I/O)
│   ├── test_rrf_fusion.py           # compute_rrf() pure function
│   ├── test_reranker.py             # JSON parsing, fallback logic
│   ├── test_models.py               # Pydantic model validation
│   └── test_classifier.py          # Pipeline orchestration (mocked)
│
├── integration/                     # Adapter tests (requires live services)
│   ├── test_postgres_adapter.py     # DB: vector_search, fts_search, fetch
│   ├── test_vertex_embedding.py     # Embedding: dimension, cosine similarity
│   └── test_gemini_llm.py          # LLM: JSON response, model name
│
└── e2e/
    └── test_classify_pipeline.py   # Full pipeline (mocked adapters)
```

---

## 3. Running Tests

### Standard run (unit + E2E, no live services needed)

```bash
cd /Users/s748779/CEP_AI/anzsic_mapping
source .venv/bin/activate

# Run all fast tests
pytest prod/tests/ -v

# Run only unit tests
pytest prod/tests/unit/ -v

# Run with coverage
pytest prod/tests/ --cov=prod --cov-report=term-missing
```

### Integration tests (requires PostgreSQL + GCP)

```bash
# Integration tests are skipped by default; opt in with the marker
pytest prod/tests/integration/ -m integration -v

# Database only
pytest prod/tests/integration/test_postgres_adapter.py -m integration -v

# GCP Vertex AI only (incurs cost)
pytest prod/tests/integration/test_vertex_embedding.py \
       prod/tests/integration/test_gemini_llm.py \
       -m integration -v
```

### E2E against live services

```bash
# Override the mock adapters in conftest.py to use real adapters
# (advanced: see section 6 below)
pytest prod/tests/e2e/ -m e2e -v
```

---

## 4. Coverage Targets

| Module                          | Target |
|---------------------------------|--------|
| `prod/services/retriever.py`    | 95%    |
| `prod/services/reranker.py`     | 90%    |
| `prod/services/classifier.py`   | 90%    |
| `prod/domain/models.py`         | 100%   |
| `prod/domain/exceptions.py`     | 80%    |
| `prod/config/prompts.py`        | 85%    |
| `prod/adapters/` (all)          | 70% (integration tests) |
| **Overall**                     | **85%** |

---

## 5. Mock Adapter Design

All mock adapters in `conftest.py` satisfy the Port Protocols via structural
subtyping — they do NOT inherit from any class.

### MockEmbeddingAdapter
- Returns deterministic 8-dim float vectors
- Always succeeds (no network calls)
- Used in: unit/test_classifier.py, e2e/test_classify_pipeline.py

### MockDatabaseAdapter
- Returns hardcoded ranked tuples from `_DB_RECORDS` fixture data
- `vector_search` returns 3 records in fixed order
- `fts_search` returns 2 records in fixed order
- Used in: all unit tests for retriever and classifier

### MockLLMAdapter
- Returns a pre-baked JSON string with 2 ranked results
- No network call, deterministic output
- Used in: unit/test_reranker.py, e2e/test_classify_pipeline.py

### MockLLMAdapterEmpty
- Returns `"[]"` on first call (simulates empty Gemini response)
- Returns real data on second call (simulates successful retry)
- Used to test CSV fallback logic

---

## 6. Adding New Tests

### New unit test for a service
1. Add to `prod/tests/unit/test_<service>.py`
2. Use only fixtures from `conftest.py` (no real I/O)
3. Focus on one behaviour per test method

### New integration test for an adapter
1. Add to `prod/tests/integration/test_<adapter>.py`
2. Decorate with `@pytest.mark.integration`
3. Use `scope="module"` fixtures to avoid repeated auth/connection overhead

### New query to golden results
1. Run the live pipeline manually:
   ```python
   from prod.services.container import get_pipeline
   from prod.domain.models import SearchRequest
   r = get_pipeline().classify(SearchRequest(query="new query"))
   print(r.to_dict())
   ```
2. Verify the top result is correct
3. Add an entry to `prod/tests/fixtures/golden_results.json`

---

## 7. CI/CD Integration

### GitHub Actions example (`.github/workflows/test.yml`)

```yaml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e "prod/[dev]"
      - run: pytest prod/tests/unit/ prod/tests/e2e/ -v --tb=short

  integration-tests:
    runs-on: ubuntu-latest
    environment: gcp-integration  # Protected environment with secrets
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - run: pip install -e "prod/[dev]"
      - run: pytest prod/tests/integration/ -m integration -v
```

---

## 8. Known Limitations

| Limitation | Mitigation |
|---|---|
| Gemini non-deterministic | Tests assert on structure, not exact content |
| Embedding vectors vary by model version | Integration tests use cosine similarity thresholds, not exact equality |
| CSV reference load time at startup | Measured; ~0.1s for 5,236 rows — acceptable |
| psycopg2 not thread-safe per connection | Each worker process gets its own adapter instance (fine for Streamlit/FastAPI workers) |

---

## 9. Performance Benchmarks (indicative)

| Operation                    | Target p50 | Target p99 |
|------------------------------|-----------|-----------|
| embed_query (Vertex AI)      | 200ms     | 800ms     |
| vector_search (PostgreSQL)   | 10ms      | 50ms      |
| fts_search (PostgreSQL)      | 5ms       | 20ms      |
| llm rerank (Gemini)          | 2s        | 8s        |
| Full HIGH_FIDELITY classify  | 2.5s      | 10s       |
| Full FAST classify            | 250ms     | 900ms     |

Measure with:
```bash
pytest prod/tests/ -v --durations=10
```
