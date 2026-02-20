# 🏭 ANZSIC Occupation Code Classifier

> Classify any free-text occupation or business description into the correct
> 6-digit ANZSIC code — in under 3 seconds.

[![Deploy MkDocs to GitHub Pages](https://github.com/IAGvk/anzsic_mapping_demo/actions/workflows/github-pages.yml/badge.svg)](https://github.com/IAGvk/anzsic_mapping_demo/actions/workflows/github-pages.yml)
[![Docs](https://img.shields.io/badge/docs-live-blue?logo=readthedocs)](https://iagvk.github.io/anzsic_mapping_demo/)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org)

---

## What it does

The ANZSIC Classifier takes unstructured text like *"mobile mechanic"* or
*"bloke who fixes cars"* and maps it to the precise 6-digit ANZSIC unit-group
code that best describes that economic activity — with a plain-English explanation
of why.

It runs in two stages:

| Stage | What happens | Technology |
|---|---|---|
| **1 · Hybrid Retrieval** | Vector ANN search + full-text search fused via Reciprocal Rank Fusion (RRF) | pgvector · PostgreSQL |
| **2 · Re-ranking** | LLM reads the candidate codes and selects the best matches with reasons | Vertex AI Gemini |

**Fast mode** runs Stage 1 only (< 300 ms).  
**High Fidelity mode** adds Stage 2 (2–5 s) for production-quality results.

---

## Screenshots

### App home screen
![App home screen](docs/screenshots/App%20Homescreen.png)

### Search results — Cards view
![Search results cards](docs/screenshots/App%20Search%20Results%20-%20Card%20View.png)

### Search results — Table view
![Search results table](docs/screenshots/App%20Search%20Results%20-%20Table%20View.png)

### Search results — JSON view
![Search results JSON](docs/screenshots/App%20Search%20Results%20-%20JSON%20View.png)

### Full results screen
![Full results screen](docs/screenshots/App%20Search%20Results%20Screen.png)

### Search settings sidebar
![Search settings sidebar](docs/screenshots/Search%20Options%20-%20Sidebar.png)

---

## Key features

- **5,236 ANZSIC codes** indexed with 768-dimensional semantic embeddings (Vertex AI `text-embedding-005`)
- **Handles colloquial descriptions** — *"barista"*, *"tradie"*, *"runs a cafe"* — that FTS-only systems miss
- **Streamlit UI** — single query and batch mode, Cards / Table / JSON tabs, CSV download
- **CLI** — `anzsic-classify "mobile mechanic"` from the terminal
- **Multi-provider** — swap between Vertex AI and OpenAI via a single env var
- **Hexagonal architecture** — DB, embedding model, and LLM are all hot-swappable
- **65 automated tests** — 100 % pass, no live services needed for unit tests

---

## Quick start

```bash
git clone https://github.com/IAGvk/anzsic_mapping_demo.git
cd anzsic_mapping_demo

python3.12 -m venv .venv && source .venv/bin/activate

cd prod && pip install -e . && cd ..

# Copy and fill in your credentials
cp .env.example .env   # edit DB_DSN, GCP_PROJECT_ID, etc.

streamlit run prod/interfaces/streamlit_app.py
```

The UI opens at **http://localhost:8501**.

---

## Documentation

Full documentation is published at **https://iagvk.github.io/anzsic_mapping_demo/**

- [Quickstart guide](https://iagvk.github.io/anzsic_mapping_demo/guides/quickstart/)
- [Architecture overview](https://iagvk.github.io/anzsic_mapping_demo/architecture/)
- [CLI reference](https://iagvk.github.io/anzsic_mapping_demo/guides/cli/)
- [Streamlit UI guide](https://iagvk.github.io/anzsic_mapping_demo/guides/streamlit/)
- [API reference](https://iagvk.github.io/anzsic_mapping_demo/reference/domain/)

---

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | Vertex AI `text-embedding-005` / OpenAI `text-embedding-3-small` |
| Vector store | PostgreSQL 15 + pgvector |
| Re-ranking LLM | Vertex AI Gemini 2.5 Flash / OpenAI GPT-4o |
| UI | Streamlit |
| Architecture | Hexagonal (Ports & Adapters) |
| Tests | pytest · 65 tests · no live services for unit tests |
| Docs | MkDocs Material → GitHub Pages |
