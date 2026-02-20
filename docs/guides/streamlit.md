# Streamlit UI

The ANZSIC Classifier has a full browser-based interface built with Streamlit.
It supports both single-query and batch classification, with results in card,
table, or JSON format and CSV download.

---

## Starting the UI

```bash
cd /Users/s748779/CEP_AI/anzsic_mapping
streamlit run prod/interfaces/streamlit_app.py
```

The browser opens automatically at [http://localhost:8501](http://localhost:8501).

!!! note "First run"
    The first request takes a few extra seconds while the pipeline loads and
    establishes the database connection. Subsequent requests are fast because
    the pipeline is cached for the lifetime of the Streamlit session
    (`@st.cache_resource`).

---

## Sidebar controls

The left sidebar contains all configuration options.

### Search Type

| Option | Description |
|---|---|
| **Single Query** | Type one description and get results immediately |
| **Batch (file upload)** | Upload a `.txt` file with multiple queries |

### Search Mode

| Mode | Pipeline stages | Typical speed | Best for |
|---|---|---|---|
| **High Fidelity (+ Gemini)** | Retrieval + LLM re-ranking | 2–5 seconds | Final classification, needs explanations |
| **Fast (retrieval only)** | Retrieval only (RRF) | < 500 ms | Exploration, large batches |

### ANZSIC Type

Currently locked to **6-digit (unit groups)** — the most granular level of
the ANZSIC hierarchy. Future releases may support 4-digit (class) or
2-digit (division) classification.

### Max Results

Number of ANZSIC codes returned per query.

- In **High Fidelity** mode: Gemini selects the best matches up to this limit.
  Setting this lower does *not* speed up the call (all candidates are still
  sent to Gemini).
- In **Fast** mode: the top-N by RRF score are returned directly.

### Retrieval Pool

Controls how many candidates Stage 1 returns before re-ranking.
A larger pool gives Gemini more to choose from, at the cost of a slightly
longer prompt. Default **20** is optimal for most queries. Increase to
**30–50** for rare or highly specific occupations.

---

## Single query mode

1. Type your description in the text box
2. Press **Classify**
3. Results appear in three tabs:

### 🃏 Cards tab

Each result is shown as a card with:

- **Rank badge** — `#1` is highlighted in green
- **ANZSIC code** (monospace, e.g. `S9419_03`)
- **ANZSIC description** — the official occupation title
- **Class and Division** — hierarchical context
- **Reason** — Gemini's plain-English explanation (High Fidelity mode only)

### 📋 Table tab

All results in a sortable, filterable data table. Includes a **Download CSV**
button to export the current results.

### { } JSON tab

The raw `ClassifyResponse` object rendered as interactive, collapsible JSON.
Useful for developers who want to inspect all metadata fields (RRF score,
source systems, model versions, timestamp).

---

## Batch mode

1. Select **Batch (file upload)** in the sidebar
2. Upload a `.txt` file — one description per line, lines starting with `#` ignored
3. A preview shows the first 10 queries
4. Press **Run Batch**
5. A progress bar tracks each query as it processes
6. Results aggregate into a single table with a **Query** column

### Batch file format

```text title="queries.txt"
# ANZSIC classification batch — Feb 2026
mobile mechanic
café owner
registered nurse
software engineer
primary school teacher
delivery driver
```

!!! warning "Batch and rate limits"
    Each query in a batch makes one embedding API call and (in High Fidelity
    mode) one Gemini API call. For batches larger than 50 queries, consider
    using **Fast mode** in the UI, or use the [CLI batch mode](cli.md#batch-processing-tips)
    which gives you more control over rate limiting.

---

## Metrics row

After each single query, a row of metric tiles appears:

| Metric | What it shows |
|---|---|
| **Results** | Number of ANZSIC codes returned |
| **Candidates** | Stage 1 retrieval pool size used |
| **Latency** | Wall-clock time for the full classify() call |
| **Mode** | `Fast` or `High Fidelity` |

---

## Customising the UI

The Streamlit app is a single file at `prod/interfaces/streamlit_app.py`.
It imports only from `prod.services.container` and `prod.domain.models`
— no infrastructure knowledge.

To embed the classifier in another Streamlit app:

```python
from prod.services.container import get_pipeline
from prod.domain.models import SearchRequest, SearchMode

# In your Streamlit page:
pipeline = get_pipeline()  # cached automatically
result = pipeline.classify(SearchRequest(query=user_input))
```
