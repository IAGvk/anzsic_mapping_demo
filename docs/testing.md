# Testing Guide

The `prod/` package has a multi-layer test suite that can run entirely offline — no GCP
credentials and no live PostgreSQL connection required.

---

## Test layout

```
prod/tests/
├── conftest.py                  ← shared fixtures (mock adapters, sample data)
├── fixtures/
│   └── sample_data.py           ← 20 representative ANZSIC rows for unit tests
├── unit/
│   ├── test_models.py           ← domain model validation, source_label, field defaults
│   ├── test_rrf.py              ← compute_rrf arithmetic, tie-breaking, k constant
│   ├── test_retriever.py        ← HybridRetriever with mock DatabasePort
│   ├── test_reranker.py         ← LLMReranker, CSV fallback, _parse_response edge cases
│   └── test_classifier.py       ← ClassifierPipeline FAST/HIGH_FIDELITY modes, error propagation
├── integration/
│   └── test_pipeline.py         ← wired pipeline with mock adapters, end-to-end data flow
└── e2e/
    └── test_live.py             ← live GCP + PostgreSQL (skipped when env vars absent)
```

---

## Running the tests

=== "All unit + integration tests"

    ```bash
    cd /path/to/anzsic_mapping
    source .venv/bin/activate
    pytest prod/tests/unit prod/tests/integration -v
    ```

=== "Unit tests only (fastest)"

    ```bash
    pytest prod/tests/unit -v
    ```

=== "With coverage report"

    ```bash
    pytest prod/tests/unit prod/tests/integration \
        --cov=prod --cov-report=term-missing
    ```

=== "Live E2E (requires GCP + DB)"

    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
    export POSTGRES_DSN="postgresql://user:pass@localhost:5432/anzsic_db"
    pytest prod/tests/e2e -v
    ```

---

## What each layer tests

### Unit tests

Each unit test file targets a single class/function with mock adapters:

| File | What is tested |
|---|---|
| `test_models.py` | `SearchMode` validation, `Candidate.source_label`, `ClassifyResult` field defaults |
| `test_rrf.py` | RRF score arithmetic against manually computed values, k=60 constant, empty-list edge case |
| `test_retriever.py` | `retrieve()` calls both DB methods, merges results, calls `compute_rrf` |
| `test_reranker.py` | `rerank()` happy path, `_parse_response(None)` safety, CSV fallback trigger |
| `test_classifier.py` | FAST mode skips LLM, HIGH_FIDELITY mode calls both stages, `DatabaseError` propagation |

### Integration tests

`test_pipeline.py` wires the full pipeline with mock adapters:

- A `MockDatabaseAdapter` returns deterministic vector and FTS results
- A `MockLLMAdapter` returns a fixed JSON ranking
- The test verifies the full `classify()` → `ClassifyResponse` structure end-to-end

### E2E tests

`test_live.py` runs against live systems:

```python
@pytest.mark.skipif(
    not os.getenv("POSTGRES_DSN"),
    reason="No live database — set POSTGRES_DSN to run"
)
def test_live_classify():
    ...
```

E2E tests are skipped automatically in CI if environment variables are absent.

---

## Mock adapters

`conftest.py` provides ready-made mock adapters:

```python
# Returns 5 fake candidates for any query
class MockDatabaseAdapter:
    def vector_search(self, embedding, n): ...
    def fts_search(self, query, n): ...
    def healthcheck(self): return True

# Returns a fixed JSON ranking for any candidate list
class MockLLMAdapter:
    def rerank(self, prompt): return '[{"rank":1,...}]'

# Returns a zero vector for any text
class MockEmbeddingAdapter:
    def embed_query(self, text): return [0.0] * 768
    def embed_documents(self, texts): return [[0.0]*768]*len(texts)
```

---

## Test counts and runtime

As of the last full run:

| Suite | Tests | Runtime |
|---|---|---|
| `unit/test_models.py` | 14 | < 0.1 s |
| `unit/test_rrf.py` | 12 | < 0.1 s |
| `unit/test_retriever.py` | 8 | < 0.1 s |
| `unit/test_reranker.py` | 11 | < 0.1 s |
| `unit/test_classifier.py` | 10 | < 0.1 s |
| `integration/test_pipeline.py` | 10 | < 0.5 s |
| **Total** | **65** | **< 1 s** |

---

## Full test plan

For the detailed test plan including acceptance criteria, edge cases, and
regression scenarios, see `prod/tests/TEST_PLAN.md` in the repository root.
