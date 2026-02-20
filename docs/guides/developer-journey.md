# Developer Journey — From Python Script to Production AI System

**Audience:** A Python developer who can write and read Python code comfortably,
but has not yet built a production-grade system.

**Goal:** Walk through every architectural decision, coding pattern, and terminal
command used to build this system — so you can reason about *why* things are
structured the way they are, not just *what* is there.

---

## 1. Where we started — the "script problem"

The first version of this classifier lived in two files:

```
anzsic_agent.py   ← 407 lines: database calls, API calls, retry logic, prompts — all mixed together
app.py            ← Streamlit UI calling functions from anzsic_agent.py
```

This works fine as a prototype. It becomes a problem when:

- **You want to run tests** — to test the RRF logic you also have to have a live database running
- **You want to swap the LLM** — changing from Gemini to GPT-4o means hunting across 400 lines
- **A second person joins** — there's no structure telling them where to put new code
- **You need a REST API** — you have to rewrite everything around a Flask/FastAPI app

The `prod/` folder solves all four problems by imposing a structure before they
become painful. Every decision from here on is motivated by one of these four goals.

---

## 2. Setting up the Python environment

Before writing any production code, you need an isolated Python environment.
This is non-negotiable — never install project packages into your system Python.

```bash
# Create a virtual environment in a folder called .venv
python3 -m venv .venv

# Activate it (your terminal prompt will change to show (.venv))
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# Confirm you are in the venv — should print a path inside .venv/
which python

# Deactivate when you leave the project
deactivate
```

**Why a venv?** Every project has different dependency versions.
`psycopg2>=2.9` here might conflict with `psycopg2==2.7` in another project on
the same machine. A venv gives each project its own isolated Python with its
own packages.

### Installing dependencies

```bash
# Install from a requirements list
pip install psycopg2-binary pgvector requests python-dotenv pydantic streamlit

# Or, if the project has pyproject.toml (our case):
pip install -e prod/        # -e = editable install (code changes take effect immediately)
pip install -e "prod/[dev]" # include dev tools: pytest, ruff, mypy
```

**The `pyproject.toml` file** (`prod/pyproject.toml`)
is the modern replacement for `requirements.txt`. It also defines the package
name, Python version requirement, and entry-point commands (`anzsic-classify`).

---

## 3. The folder structure — one concept per layer

```
prod/
├── config/          ← What the app is configured with (settings, prompts)
├── domain/          ← What the app works on (data shapes, error types)
├── ports/           ← What the app needs (abstract interface contracts)
├── adapters/        ← How those needs are met (concrete API/DB code)
├── services/        ← What the app does (business logic: RRF, reranking)
├── interfaces/      ← How humans talk to the app (CLI, Streamlit)
└── tests/           ← Proof that each layer does what it claims
```

This layout is called **Hexagonal Architecture** (also "Ports & Adapters").
The rule is: **dependencies only point inward**.

```
interfaces  →  services  →  ports  ←  adapters
                    ↑
                  domain
```

- `services` know about `ports` and `domain` — nothing else
- `adapters` know about `ports` and `domain` — nothing else
- `interfaces` know about `services` and `domain` — nothing else
- **No layer imports from the layer outside it**

This rule is what lets you swap Gemini for GPT-4o by changing one line.

### Why `__init__.py` in every folder?

```bash
# Every subfolder needs this (can be empty) to be a Python package
touch prod/__init__.py
touch prod/config/__init__.py
touch prod/domain/__init__.py
# ... and so on
```

Without `__init__.py`, Python won't let you write `from prod.domain.models import Candidate`.
The file signals "this folder is a Python package, not just a folder".

---

## 4. OOP concept: Dataclasses — typed data containers

Before Pydantic, Python's standard library offers `dataclasses` for structured
data. We use `@dataclass(frozen=True)` in settings:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)          # frozen=True → fields are read-only after creation
class Settings:
    db_dsn: str = field(default_factory=lambda: os.getenv("DB_DSN", "dbname=anzsic_db"))
    rrf_k: int  = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
```

**`frozen=True`** means `settings.rrf_k = 99` will raise a `FrozenInstanceError`
at runtime. This prevents accidental mutation of configuration — settings should
be read-only after the app starts.

**`field(default_factory=...)`** is used instead of a plain default value when
the default is computed (e.g. reading an env var). Plain defaults like `rrf_k: int = 60`
are evaluated once at class definition time, not at instance creation.

### OOP concept: `@lru_cache` — singleton via function

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

`@lru_cache(maxsize=1)` memoises the function — it runs once, caches the result,
and returns the same object on every subsequent call. This gives us a singleton
Settings object without writing a Singleton class. Import `get_settings` anywhere
in the codebase and you always get the same instance.

---

## 5. OOP concept: Pydantic models — validated data

The domain models (`prod/domain/models.py`) use Pydantic:

```python
from pydantic import BaseModel, Field, field_validator

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)    # ge = greater-or-equal, le = less-or-equal

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()                   # run automatically on every instantiation
```

**What Pydantic adds over a plain dataclass:**

| Feature | Plain dataclass | Pydantic |
|---|---|---|
| Type hints enforced at runtime | ❌ | ✅ |
| Built-in validation (min/max) | ❌ | ✅ |
| Auto-strip whitespace | Manual | `@field_validator` |
| Serialise to JSON / dict | Manual | `.model_dump()` |
| Generate JSON Schema | ❌ | ✅ (free FastAPI docs later) |

### OOP concept: Enums — named constants with type safety

```python
from enum import Enum

class SearchMode(str, Enum):
    FAST           = "fast"
    HIGH_FIDELITY  = "high_fidelity"
```

`SearchMode.FAST` is better than the string `"fast"` throughout the codebase
because your IDE can autocomplete it, and misspelling `"fsat"` raises an error
immediately instead of silently doing the wrong thing at runtime.

`str, Enum` (inheriting from both `str` and `Enum`) means instances compare
equal to their string value: `SearchMode.FAST == "fast"` is `True`. This
makes them work transparently with JSON serialisation.

---

## 6. OOP concept: Protocols — interface contracts without inheritance

This is the most important concept in the entire codebase.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingPort(Protocol):
    """Any class that has these methods is a valid EmbeddingPort."""

    @property
    def model_name(self) -> str: ...

    def embed_query(self, text: str) -> list[float]: ...
    def embed_document(self, text: str, title: str = "") -> list[float]: ...
```

**The key insight:** `HybridRetriever` accepts an `EmbeddingPort` in its
constructor. It never imports `VertexEmbeddingAdapter` or `OpenAIEmbeddingAdapter`.
It only knows the *contract* (the Port).

This is called **structural subtyping** (also "duck typing with types").
Any class that implements `model_name`, `embed_query`, and `embed_document`
satisfies `EmbeddingPort` — even if it never inherits from it.

**Compare to traditional OOP inheritance:**

```python
# Traditional (Java-style) — tightly coupled
class VertexEmbeddingAdapter(BaseEmbedder):  # inherits from a base class
    ...

# Protocol-style — loosely coupled
class VertexEmbeddingAdapter:                # no inheritance at all
    def embed_query(self, text: str) -> list[float]:
        ...  # just implement the methods
```

The Protocol approach means swapping adapters has zero inheritance hierarchy
to maintain. Your mock test adapter is also just a plain class with the right methods.

---

## 7. Setting up PostgreSQL and pgvector

```bash
# Install PostgreSQL (macOS with Homebrew)
brew install postgresql@15
brew services start postgresql@15

# Create the database
createdb anzsic_db

# Connect to it
psql anzsic_db

# Inside psql — install the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
\q    # quit psql
```

### Why pgvector inside PostgreSQL?

The `vector(768)` column type stores a 768-dimensional float array. The HNSW
index makes approximate nearest-neighbour search fast:

```sql
-- Create the table with both a vector column and a full-text column
CREATE TABLE anzsic_codes (
    anzsic_code  TEXT PRIMARY KEY,
    anzsic_desc  TEXT,
    embedding    vector(768),   -- pgvector
    fts_vector   tsvector       -- PostgreSQL built-in full-text
);

-- HNSW index for vector search (cosine distance)
CREATE INDEX ON anzsic_codes
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX ON anzsic_codes USING gin (fts_vector);
```

**HNSW** (Hierarchical Navigable Small World) is a graph-based ANN algorithm.
`m=16` is the number of bidirectional links per node; `ef_construction=64` is
the search width during index build. Larger values = better recall but slower
build time.

### Useful psql commands for debugging

```bash
psql anzsic_db                      # connect
\dt                                 # list all tables
\d anzsic_codes                     # describe table structure + indexes
SELECT COUNT(*) FROM anzsic_codes;  # count rows
\timing on                          # show query execution time
EXPLAIN ANALYZE SELECT ...;         # show query plan + actual timing
\q                                  # quit
```

---

## 8. OOP concept: Dependency Injection — wiring without coupling

Look at `HybridRetriever`'s constructor:

```python
class HybridRetriever:
    def __init__(
        self,
        db: DatabasePort,
        embedder: EmbeddingPort,
        settings: Settings,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._settings = settings
```

`HybridRetriever` doesn't know or care *which* database or embedder it gets.
It receives them as constructor arguments — this is **Dependency Injection (DI)**.

The caller who *creates* `HybridRetriever` decides which concrete implementations
to inject. That caller is `services/container.py` — the only file allowed to
name concrete adapter classes:

```python
# container.py — the single wiring point
def _build_embedder(settings) -> EmbeddingPort:
    if settings.embed_provider == "openai":
        from prod.adapters.openai_embedding import OpenAIEmbeddingAdapter
        return OpenAIEmbeddingAdapter(settings)
    else:
        from prod.adapters.vertex_embedding import VertexEmbeddingAdapter
        auth = GCPAuthManager(settings)
        return VertexEmbeddingAdapter(auth, settings)

@lru_cache(maxsize=1)
def get_pipeline() -> ClassifierPipeline:
    settings = get_settings()
    embedder = _build_embedder(settings)   # EmbeddingPort
    llm      = _build_llm(settings)        # LLMPort
    db       = PostgresDatabaseAdapter(settings)
    retriever = HybridRetriever(db=db, embedder=embedder, settings=settings)
    reranker  = LLMReranker(llm=llm, settings=settings)
    return ClassifierPipeline(retriever=retriever, reranker=reranker, settings=settings)
```

**The layering rule enforced by container.py:**

```
services/container.py  ← ONLY file that imports adapter classes
         ↓
    services/*         ← import only Ports (EmbeddingPort, LLMPort, DatabasePort)
         ↓
    adapters/*         ← import only Ports + domain models
```

If you ever find yourself importing `VertexEmbeddingAdapter` inside
`services/retriever.py` — that's a rule violation.

---

## 9. Writing an adapter — the pattern

Every adapter follows the same pattern. Here is `OpenAILLMAdapter` stripped to its skeleton:

```python
class OpenAILLMAdapter:
    """Implements LLMPort. Injected via container.py."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise AuthenticationError("OPENAI_API_KEY is not set.")
        self._settings = settings
        self._headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    # ── LLMPort implementation (these method signatures must match the Protocol)
    @property
    def model_name(self) -> str:
        return self._settings.openai_llm_model

    def generate_json(self, system_prompt: str, user_message: str) -> str | None:
        payload = self._build_payload(system_prompt, user_message)
        return self._post_with_retry(payload)

    # ── Private helpers (named with _ prefix = "internal, don't call these directly")
    def _build_payload(self, system_prompt: str, user_message: str) -> dict:
        return {
            "model": self._settings.openai_llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
        }

    def _post_with_retry(self, payload: dict, retries: int = 3) -> str | None:
        delay = 2.0
        for attempt in range(1, retries + 1):
            resp = requests.post(self._url, headers=self._headers, json=payload)
            if resp.status_code == 429:        # rate limited — wait and retry
                time.sleep(delay); delay *= 2  # exponential back-off
                continue
            if resp.ok:
                return self._extract_text(resp.json())
        return None
```

**Naming conventions used throughout:**

| Pattern | Meaning |
|---|---|
| `_method()` | Private — internal to this class, don't call from outside |
| `__method()` | Name-mangled — strongly private (used rarely) |
| `UPPER_CASE` | Module-level constants |
| `_UPPER_CASE` | Module-level private constants |

### Custom exceptions — the hierarchy

```python
# domain/exceptions.py
class ANZSICError(Exception):          # root — catch everything with: except ANZSICError
    ...
class AuthenticationError(ANZSICError): ...  # bad API key
class EmbeddingError(ANZSICError):      ...  # API call failed
class DatabaseError(ANZSICError):       ...  # DB query failed
```

Raising a specific exception (`raise EmbeddingError(...)`) instead of a
generic `Exception` lets callers decide how broadly or narrowly to catch:

```python
try:
    result = embedder.embed_query(text)
except EmbeddingError:
    # handle specifically
except ANZSICError:
    # catch anything from our app
except Exception:
    # catch truly unexpected failures
```

---

## 10. The `from __future__ import annotations` line

You will see this at the top of nearly every file:

```python
from __future__ import annotations
```

This enables **postponed evaluation of type annotations** — Python 3.10+
behaviour backported to 3.9/3.8. It lets you write:

```python
def retrieve(self, request: SearchRequest) -> list[Candidate]:
    ...
```

...even in a file where `SearchRequest` or `Candidate` might be defined *later*
in the same file, or where the type is a forward reference. Without this import,
you'd need to wrap the type in quotes: `-> "list[Candidate]"`. With it, all
annotations are strings at import time and only evaluated when needed.

---

## 11. The two-stage pipeline — RRF fusion

The core algorithm is in `services/retriever.py`:

```python
def compute_rrf(
    vector_results: list[tuple[str, int]],   # [(code, rank), ...]
    fts_results:    list[tuple[str, int]],
    k: int = 60,
) -> list[_RRFResult]:
    scores: dict[str, float] = {}

    for code, rank in vector_results:
        scores[code] = scores.get(code, 0.0) + 1.0 / (k + rank)

    for code, rank in fts_results:
        scores[code] = scores.get(code, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

`compute_rrf` is a **pure function**: same inputs always produce same outputs,
no database calls, no API calls, no side effects. This makes it trivially
testable:

```python
def test_code_in_both_systems_has_highest_score():
    vec_results = [("CODE_A", 1), ("CODE_B", 2)]
    fts_results = [("CODE_A", 1), ("CODE_C", 2)]
    results = compute_rrf(vec_results, fts_results, k=60)
    assert results[0].anzsic_code == "CODE_A"  # appears in both → wins
```

**Design principle:** extract pure logic into standalone functions.
They are the easiest to test and the safest to refactor.

---

## 12. Environment variables and `.env` files

Never hardcode secrets or configuration values in source code. Use environment
variables instead:

```bash
# Check what an env var is set to
echo $OPENAI_API_KEY

# Set one in the current shell session (lost when you close the terminal)
export OPENAI_API_KEY="sk-..."

# Or put it in .env (loaded by python-dotenv at startup)
echo "OPENAI_API_KEY=sk-..." >> .env
```

The `.env` file must **never be committed to git**:

```bash
# .gitignore — create this at the repo root if it doesn't exist
echo ".env" >> .gitignore
echo ".venv/" >> .gitignore
echo "site/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

`prod/.env.example` is the
safe version — it shows the *shape* of the configuration with placeholder values.
New developers copy it and fill in real values:

```bash
cp prod/.env.example .env
# then edit .env with your actual keys
```

**How python-dotenv loads it:**

```python
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")
# This runs at import time in settings.py — before any Settings() is created
```

---

## 13. Testing — the three layers

```bash
prod/tests/
├── unit/          ← Tests one class/function in complete isolation (no I/O)
├── integration/   ← Tests one adapter against a real external system
└── e2e/           ← Tests the full pipeline end-to-end
```

### Running tests

```bash
# Run everything
pytest prod/tests/unit prod/tests/integration -v

# Run only unit tests (fast — no network, no DB)
pytest prod/tests/unit -v

# Run a single test file
pytest prod/tests/unit/test_rrf_fusion.py -v

# Run a single test by name
pytest prod/tests/unit/test_rrf_fusion.py::TestComputeRRF::test_rrf_scores_are_positive -v

# Run with coverage report
pytest prod/tests/unit --cov=prod --cov-report=term-missing
```

### How mock adapters work

The key to fast unit tests is mock adapters in `tests/conftest.py`.
A mock adapter is a plain Python class that satisfies a Port Protocol
without doing any real I/O:

```python
class MockEmbeddingAdapter:
    """Satisfies EmbeddingPort — returns a fixed fake vector instantly."""
    model_name = "mock-embedding"
    dimensions = 8

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]  # always the same

    def embed_document(self, text: str, title: str = "") -> list[float]:
        return [0.2, 0.1, 0.4, 0.3, 0.2, 0.1, 0.4, 0.3]
```

When `HybridRetriever` gets a `MockEmbeddingAdapter` injected instead of a
`VertexEmbeddingAdapter`, it cannot tell the difference — both satisfy the
`EmbeddingPort` Protocol. This is dependency injection + Protocols making
tests effortless.

### `unittest.mock.patch` — intercepting HTTP calls

For the OpenAI adapter tests, we don't want real API calls. `patch` replaces
`requests.post` with a `MagicMock` for the duration of one test:

```python
from unittest.mock import patch, MagicMock

def test_embed_query_calls_correct_endpoint(openai_settings):
    vec = [0.1] * 8
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"data": [{"index": 0, "embedding": vec}]}

    with patch("prod.adapters.openai_embedding.requests.post", return_value=mock_resp):
        adapter = OpenAIEmbeddingAdapter(openai_settings)
        result = adapter.embed_query("café owner")

    assert result == vec
    # Verify the right URL was called
    assert mock_resp.json.called
```

`with patch(...) as mock_post:` — everything inside the `with` block runs
with the real `requests.post` replaced by the mock. After the `with` block,
the real `requests.post` is restored automatically.

### `pytest` fixtures — reusable test setup

```python
# conftest.py — shared across all test files in the same folder
import pytest
from prod.config.settings import Settings

@pytest.fixture
def openai_settings():
    """Return a Settings instance configured for OpenAI."""
    return Settings(openai_api_key="sk-test", embed_provider="openai", ...)

# In a test file — pytest injects it automatically by name
def test_something(openai_settings):   # ← pytest sees the name, finds the fixture
    adapter = OpenAIEmbeddingAdapter(openai_settings)
    ...
```

Fixtures in `conftest.py` are automatically available to all test files in
the same directory and all subdirectories — no import needed.

---

## 14. Logging — the right way to print in production

Never use `print()` in production code. Use Python's `logging` module:

```python
import logging
logger = logging.getLogger(__name__)   # __name__ = "prod.adapters.openai_llm"

logger.debug("Sending request | url=%s model=%s", url, model)   # dev detail
logger.info("OpenAI adapter ready | model=%s", model_name)      # normal ops
logger.warning("Rate limited (429) — back-off %.1fs", delay)    # worth noting
logger.error("API failed after %d retries", retries)            # needs attention
```

`logging.getLogger(__name__)` uses the module's dotted path as the logger name.
This lets operators filter logs by module:

```bash
# Show only warnings and above from the whole app
python -m prod.interfaces.cli "barista" --log-level WARNING

# Show debug logs only from the embedding adapter
# (configured in logging config — not covered here)
```

**Why not `print()`?**

- `print()` always goes to stdout, always
- `logging` can be directed to files, cloud log sinks, suppressed in tests
- Log levels let you turn up detail in production without code changes

---

## 15. Building the documentation

```bash
# Install MkDocs and plugins
pip install mkdocs-material "mkdocstrings[python]" mkdocs-mermaid2-plugin ruff

# Serve locally with live-reload (best for writing docs)
.venv/bin/mkdocs serve
# Open http://127.0.0.1:8000 in your browser

# Build static HTML into site/
.venv/bin/mkdocs build

# Build with strict mode — treats any warning as an error
.venv/bin/mkdocs build --strict
```

**How `mkdocstrings` works:**

Any docstring you write in the code is automatically rendered into the docs
by placing a `:::` directive in a markdown file:

```markdown
::: prod.adapters.openai_llm
    options:
      members:
        - OpenAILLMAdapter
```

This renders the class docstring, all method docstrings, and their type
signatures — no duplication between code and docs.

**What makes a good docstring:**

```python
def embed_query(self, text: str) -> list[float]:
    """Embed a search query.

    Uses RETRIEVAL_QUERY task type so the vector is optimised for
    matching against document embeddings.

    Args:
        text: Natural-language query string.

    Returns:
        Dense vector of floats with length ``settings.embed_dim``.

    Raises:
        EmbeddingError: On API failure or empty response.
    """
```

Google-style docstring format (Args / Returns / Raises sections) is what
`mkdocstrings` expects and renders beautifully.

---

## 16. Pivotal commands — quick reference

### Environment
```bash
python3 -m venv .venv                  # create venv
source .venv/bin/activate              # activate
pip install -e "prod/[dev]"            # install project + dev tools
```

### Database
```bash
createdb anzsic_db                     # create the database
psql anzsic_db                         # connect
CREATE EXTENSION IF NOT EXISTS vector; # enable pgvector (inside psql)
\d anzsic_codes                        # inspect table schema
SELECT COUNT(*) FROM anzsic_codes;     # count rows
```

### Tests
```bash
pytest prod/tests/unit -v              # unit tests (fast, no I/O)
pytest prod/tests/unit --cov=prod      # with coverage
pytest -k "test_rrf"                   # run tests matching a name pattern
pytest --tb=short                      # shorter tracebacks on failure
```

### Docs
```bash
mkdocs serve                           # live preview at localhost:8000
mkdocs build --strict                  # production build (zero warnings)
mkdocs gh-deploy --force               # deploy to GitHub Pages
```

### Linting & formatting
```bash
ruff check prod/                       # lint check (find code style issues)
ruff format prod/                      # auto-format all Python files
ruff check --fix prod/                 # auto-fix fixable issues
```

### Git essentials for a new project
```bash
git init                               # initialise a new repo
git add .                              # stage all changes
git status                             # see what's staged / changed
git commit -m "feat: add OpenAI adapters"  # commit
git push origin main                   # push to GitHub
```

---

## 17. The `prod/` layering — a visual map

```
                    ┌──────────────────────────────────┐
                    │         INTERFACES                │
                    │   cli.py    streamlit_app.py      │
                    └─────────────┬────────────────────┘
                                  │ calls
                    ┌─────────────▼────────────────────┐
                    │           SERVICES                │
                    │  ClassifierPipeline               │
                    │    ├── HybridRetriever            │
                    │    │     └── compute_rrf()        │
                    │    └── LLMReranker                │
                    └──────┬──────────────┬────────────┘
                           │ uses         │ uses
               ┌───────────▼──┐    ┌──────▼───────────┐
               │    PORTS     │    │     PORTS         │
               │ EmbeddingPort│    │  LLMPort          │
               │ DatabasePort │    │                   │
               └───────┬──────┘    └──────┬────────────┘
                       │ implemented by   │ implemented by
          ┌────────────▼──────────────────▼────────────┐
          │              ADAPTERS                       │
          │  VertexEmbeddingAdapter  OpenAIEmbedding    │
          │  GeminiLLMAdapter        OpenAILLMAdapter   │
          │  PostgresDatabaseAdapter                    │
          │  GCPAuthManager                             │
          └────────────────────────────────────────────┘
                       ↑ wired together by
               ┌───────────────────┐
               │  container.py     │  ← only file that names concrete adapters
               └───────────────────┘
```

Everything rests on:

```
┌─────────────────────────────────────────────────────┐
│                     DOMAIN                          │
│  SearchRequest  Candidate  ClassifyResult           │
│  SearchMode  ANZSICError hierarchy                  │
│  (pure Python — no external imports)                │
└─────────────────────────────────────────────────────┘
```

---

## 18. What to learn next

| Topic | Where to look |
|---|---|
| FastAPI — add a REST API layer | Add `prod/interfaces/api.py`; domain models already produce JSON Schema |
| async Python | Replace `requests` with `httpx` + `async def`; adapters become `async` |
| Docker — containerise the app | Write a `Dockerfile`, `docker-compose.yml` for the DB |
| CI/CD — auto-run tests on push | `.github/workflows/test.yml` with `pytest` + `mkdocs gh-deploy` |
| Connection pooling — scale the DB | Swap `psycopg2` for `psycopg2.pool.ThreadedConnectionPool` in `postgres_db.py` |
| Weaviate / Qdrant — dedicated vector DB | Write a `WeaviateDatabaseAdapter` implementing `DatabasePort` |

Every one of these is a *one-file* change or addition — the hexagonal
structure was deliberately designed so that the scope of any future change
is as small as possible.
