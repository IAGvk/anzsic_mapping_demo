"""
interfaces/api.py
──────────────────────────────────────────────────────────────────────────────
FastAPI HTTP API for the ANZSIC classifier.

Designed for multi-tenant, API-first usage (e.g. called by an upstream
platform serving 30-40 concurrent users).

Key design decisions:
  - Blocking pipeline calls run via asyncio.to_thread() so the event loop
    never stalls.  Uvicorn's default thread-pool handles concurrency.
  - The ClassifierPipeline singleton is shared across all requests in a
    process (stateless between calls — safe to share).
  - PostgresDatabaseAdapter uses ThreadedConnectionPool (see postgres_db.py)
    so concurrent threads each get their own DB connection.
  - Run with multiple Uvicorn workers for horizontal scaling:
      uvicorn prod.interfaces.api:app --workers 4 --host 0.0.0.0 --port 8000

Endpoints:
  POST /classify          → classify a single description
  GET  /health            → liveness probe  (always 200 if process is alive)
  GET  /readiness         → readiness probe (checks pipeline is initialised)

Run locally (single worker, hot-reload for dev):
  uvicorn prod.interfaces.api:app --reload --port 8000

Run production (4 workers — each gets its own pipeline + DB pool):
  uvicorn prod.interfaces.api:app --workers 4 --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Domain models — already Pydantic, serialise straight to JSON
from prod.domain.models import ClassifyResponse, SearchMode, SearchRequest
from prod.services.container import get_pipeline

logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ANZSIC Classifier API",
    description=(
        "Maps occupation / business descriptions to ANZSIC codes.\n\n"
        "Backed by pgvector hybrid search + Gemini LLM re-ranking."
    ),
    version="1.0.0",
)

# Allow all origins for internal/intranet usage — tighten in prod if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    """Warm up the pipeline singleton so the first real request is not slow."""
    logger.info("Warming up ClassifierPipeline…")
    await asyncio.to_thread(get_pipeline)
    logger.info("ClassifierPipeline ready")


# ── Request model ──────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    """Body for POST /classify."""

    query: str = Field(
        ...,
        description="Occupation or business description to classify.",
        min_length=1,
        max_length=500,
        examples=["mobile mechanic"],
    )
    mode: str = Field(
        default="high_fidelity",
        description="'high_fidelity' (retrieval + LLM) or 'fast' (retrieval only).",
        pattern="^(high_fidelity|fast)$",
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return.",
        ge=1,
        le=20,
    )
    retrieval_n: int = Field(
        default=20,
        description="Stage 1 candidate pool size.",
        ge=5,
        le=100,
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _run_classify(body: ClassifyRequest) -> ClassifyResponse:
    """Run the blocking pipeline synchronously (called via asyncio.to_thread)."""
    pipeline = get_pipeline()
    req = SearchRequest(
        query=body.query,
        mode=SearchMode(body.mode),
        top_k=body.top_k,
        retrieval_n=body.retrieval_n,
    )
    return pipeline.classify(req)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassifyResponse)
async def classify(body: ClassifyRequest) -> ClassifyResponse:
    """Classify an occupation or business description into ANZSIC codes.

    Runs the blocking pipeline in a thread pool worker so many concurrent
    requests are handled without blocking the Uvicorn event loop.

    Returns the full ClassifyResponse domain object as JSON.
    """
    t0 = time.perf_counter()
    logger.info("classify | query=%r mode=%s top_k=%d", body.query, body.mode, body.top_k)

    try:
        response = await asyncio.to_thread(_run_classify, body)
    except Exception as exc:
        logger.exception("classify failed for query=%r", body.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "classify | query=%r results=%d latency=%.0fms",
        body.query,
        len(response.results),
        elapsed_ms,
    )
    return response


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe — returns 200 if the process is running."""
    return JSONResponse({"status": "ok"})


@app.get("/readiness")
async def readiness() -> JSONResponse:
    """Readiness probe — verifies the pipeline singleton is initialised."""
    try:
        await asyncio.to_thread(get_pipeline)
        return JSONResponse({"status": "ready"})
    except Exception as exc:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail=f"Pipeline not ready: {exc}") from exc
