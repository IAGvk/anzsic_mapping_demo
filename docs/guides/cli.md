# CLI Reference

The `anzsic-classify` command-line interface classifies occupation and business
descriptions into ANZSIC codes from your terminal.

---

## Invocation

```bash
# Via Python module (always works from the repo root)
python -m prod.interfaces.cli [OPTIONS]

# Via installed entry-point (if installed with pip install -e .)
anzsic-classify [OPTIONS]
```

---

## Options

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--query` | `-q` | `TEXT` | — | Single query to classify |
| `--file` | `-f` | `FILE` | — | Path to a text file (one query per line) |
| `--mode` | `-m` | `fast` \| `high_fidelity` | `high_fidelity` | Search mode |
| `--top-k` | `-k` | `INT` | `5` | Number of results to return |
| `--candidates` | `-c` | `INT` | `20` | Stage 1 retrieval pool size |
| `--json` | | flag | off | Output results as JSON |
| `--verbose` | `-v` | flag | off | Enable debug logging |

!!! info "Either `--query` or `--file` is required"
    Running without either prints the help message and exits.

---

## Search modes

### `high_fidelity` (default)

Runs both pipeline stages:

1. **Stage 1** — Hybrid vector + FTS search, RRF fusion → 20 candidates
2. **Stage 2** — Gemini reads all candidates and selects the top-k with reasons

Best for: production use, final classification decisions, cases where you need
an explanation for *why* a code was chosen.

Typical latency: **2–5 seconds**

### `fast`

Runs Stage 1 only. Results are the top-k by RRF score, with the reason field
showing the raw score and source systems.

Best for: interactive exploration, large batch jobs, cases where speed matters
more than explanation quality.

Typical latency: **200–400 ms**

---

## Examples

=== "Single query"

    ```bash
    python -m prod.interfaces.cli --query "mobile mechanic"
    ```

    ```
    ────────────────────────────────────────────────────────────
    Query : mobile mechanic
    Mode  : high_fidelity  |  Candidates: 20
    ────────────────────────────────────────────────────────────
      #1  [S9419_03] Automotive Repair and Maintenance (own account)
           Class: Other Repair and Maintenance
           Division: Other Services
           Reason: Mobile mechanics who work independently on customers'
                   vehicles map directly to own-account automotive repair.

      #2  [S9411_01] Automotive Electrical Services
           Class: Automotive Repair and Maintenance
           Division: Other Services
           Reason: Secondary match for mechanics who specialise in
                   electrical diagnostics.
    ```

=== "Fast mode, fewer results"

    ```bash
    python -m prod.interfaces.cli -q "café owner" -m fast -k 3
    ```

=== "JSON output"

    ```bash
    python -m prod.interfaces.cli -q "registered nurse" --json
    ```

    ```json
    {
      "query": "registered nurse",
      "mode": "high_fidelity",
      "results": [
        {
          "rank": 1,
          "anzsic_code": "Q8531_01",
          "anzsic_desc": "Nursing Care Facilities",
          "reason": "..."
        }
      ],
      "candidates_retrieved": 20,
      "generated_at": "2026-02-20T04:12:33.001Z",
      "embed_model": "text-embedding-005",
      "llm_model": "gemini-2.5-flash"
    }
    ```

=== "Batch from file"

    ```bash
    python -m prod.interfaces.cli --file queries.txt --mode high_fidelity --json
    ```

    `queries.txt` format:
    ```text title="queries.txt"
    # Lines starting with # are ignored
    mobile mechanic
    cafe owner
    registered nurse
    software engineer
    ```

=== "Larger retrieval pool"

    ```bash
    # Pull 50 candidates before re-ranking (useful for rare occupations)
    python -m prod.interfaces.cli -q "farrier" --candidates 50
    ```

=== "Debug logging"

    ```bash
    python -m prod.interfaces.cli -q "plumber" --verbose
    ```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All queries classified successfully |
| `1` | One or more queries failed (error printed to stderr) |
| `2` | Bad arguments (e.g. neither `--query` nor `--file` provided) |

---

## Batch processing tips

For large batch jobs, use `--mode fast` to avoid rate limits on the Gemini API:

```bash
python -m prod.interfaces.cli \
    --file large_batch.txt \
    --mode fast \
    --top-k 3 \
    --json > results.json
```

Then post-process `results.json` — each line is a complete `ClassifyResponse`
object for one query.

For production-quality results on large batches, run High Fidelity mode but
add a brief sleep between queries to stay within Gemini's QPS limits:

```python
import time, subprocess, json

queries = open("queries.txt").read().splitlines()
results = []
for q in queries:
    r = subprocess.run(
        ["python", "-m", "prod.interfaces.cli", "--query", q, "--json"],
        capture_output=True, text=True
    )
    results.append(json.loads(r.stdout))
    time.sleep(0.5)  # 2 QPS rate limit headroom
```
