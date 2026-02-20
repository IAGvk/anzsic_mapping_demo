# Quickstart

Get the ANZSIC Classifier running on your machine in under 5 minutes.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | |
| PostgreSQL | 15 | Must have pgvector 0.8+ extension installed |
| gcloud CLI | Latest | Must be authenticated (`gcloud auth login`) |
| Network | — | Corporate proxy configured if on the IAG network |

---

## 1 — Clone and set up the environment

```bash
cd /Users/s748779/CEP_AI/anzsic_mapping

# The virtual environment is already created at .venv
# Activate it
source .venv/bin/activate

# Verify the package is importable
python -c "from prod.services.container import get_pipeline; print('OK')"
```

---

## 2 — Configure environment variables

Copy the example env file and fill in your values:

```bash
cp prod/.env.example .env
```

The only values you must change from the defaults:

```bash title=".env"
GCP_PROJECT_ID=top-arc-65ca          # (1)
GCP_LOCATION_ID=australia-southeast1 # (2)
GCLOUD_PATH=/path/to/gcloud          # (3)
HTTPS_PROXY=cloudproxy.auiag.corp:8080  # (4)
```

1. Your GCP project ID
2. The Vertex AI region — must support `text-embedding-005` and `gemini-2.5-flash`
3. Run `which gcloud` to find this
4. Remove this line entirely if you are not behind a corporate proxy

!!! tip "Already configured"
    If the pipeline was previously working (i.e. the `.env` file already exists
    at the project root), you do not need to do anything here.

---

## 3 — Verify the database is running

```bash
psql dbname=anzsic_db -c "SELECT count(*) FROM anzsic_codes;"
```

You should see `5236`. If the table is empty or missing, run the ingest script:

```bash
python ingest.py           # loads anzsic_master.csv → PostgreSQL
python embed.py            # generates embeddings (uses anzsic_embeddings.json checkpoint)
```

---

## 4 — Run your first classification

=== "CLI — single query"

    ```bash
    python -m prod.interfaces.cli --query "mobile mechanic"
    ```

    Expected output:

    ```
    ────────────────────────────────────────────────────────────
    Query : mobile mechanic
    Mode  : high_fidelity  |  Candidates: 20
    ────────────────────────────────────────────────────────────
      #1  [S9419_03] Automotive Repair and Maintenance (own account)
           Class: Other Repair and Maintenance
           Division: Other Services
           Reason: A mobile mechanic who works independently performing
                   automotive repair at a customer's location maps directly
                   to own-account automotive repair.
    ```

=== "CLI — fast mode (no LLM)"

    ```bash
    python -m prod.interfaces.cli --query "mobile mechanic" --mode fast
    ```

=== "Streamlit UI"

    ```bash
    streamlit run prod/interfaces/streamlit_app.py
    ```

    Then open [http://localhost:8501](http://localhost:8501) in your browser.

=== "Python API"

    ```python
    from prod.services.container import get_pipeline
    from prod.domain.models import SearchRequest, SearchMode

    pipeline = get_pipeline()
    response = pipeline.classify(
        SearchRequest(query="mobile mechanic", mode=SearchMode.HIGH_FIDELITY)
    )

    for result in response.results:
        print(f"#{result.rank}  [{result.anzsic_code}]  {result.anzsic_desc}")
        print(f"      Reason: {result.reason}")
    ```

---

## 5 — Run the test suite

```bash
# Unit + E2E tests (no GCP or DB needed)
python -m pytest prod/tests/unit/ prod/tests/e2e/ -v

# Integration tests (requires live PostgreSQL + GCP auth)
python -m pytest prod/tests/integration/ -m integration -v
```

Expected: **65 passed** for the unit/e2e suite.

---

## Troubleshooting

??? failure "`AuthenticationError: gcloud not found`"
    Set `GCLOUD_PATH` in your `.env` file to the full path returned by `which gcloud`.

??? failure "`DatabaseError: Cannot connect to database`"
    Ensure PostgreSQL is running: `brew services start postgresql@15`
    and that the `anzsic_db` database exists: `createdb anzsic_db`

??? failure "`EmbeddingError: Vertex AI Embed returned HTTP 403`"
    Your GCP token has expired or you are not authenticated.
    Run: `gcloud auth application-default login`

??? failure "Proxy / SSL errors"
    On the IAG corporate network, ensure `HTTPS_PROXY=cloudproxy.auiag.corp:8080`
    is set in your `.env` file. Do not set `HTTPS_PROXY` when working off-network.
