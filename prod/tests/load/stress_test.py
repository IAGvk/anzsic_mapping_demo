"""
tests/load/stress_test.py
──────────────────────────────────────────────────────────────────────────────
Async latency / stress test for the ANZSIC Classifier FastAPI endpoint.

Simulates N concurrent users each sending one classify request, then reports
per-request latency and aggregate statistics (P50/P95/P99).

No browsers needed — pure Python using httpx + asyncio.

Prerequisites
─────────────
  1. Start the API server in a separate terminal:
       uvicorn prod.interfaces.api:app --workers 4 --port 8000

  2. Run this script (from repo root, with .venv active):
       python -m prod.tests.load.stress_test            # default 30 concurrent users
       python -m prod.tests.load.stress_test --users 40 --mode fast
       python -m prod.tests.load.stress_test --queries fixtures/sample_queries.txt

Corporate proxy note
────────────────────
The HTTPS_PROXY env var is only used for outbound calls TO Vertex AI / GCP.
Requests from this script TO localhost:8000 bypass the proxy entirely.
→ Round-trip latency measured here = network to localhost + pipeline execution
  (DB + embedding + LLM through proxy).  That IS the real-world figure.

To isolate proxy latency:
  Run with --mode fast (skips LLM call entirely) — any latency diff between
  fast and high_fidelity = LLM call time, which includes proxy overhead.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── httpx (install: pip install httpx) ────────────────────────────────────
try:
    import httpx
except ImportError:
    print("ERROR: httpx is not installed.  Run: pip install httpx")
    sys.exit(1)

# ── Default test queries ───────────────────────────────────────────────────
_DEFAULT_QUERIES = [
    "mobile mechanic",
    "plumber",
    "registered nurse",
    "software engineer",
    "electrician in construction",
    "retail store manager",
    "primary school teacher",
    "truck driver",
    "accountant",
    "real estate agent",
    "chef in a restaurant",
    "dentist",
    "insurance broker",
    "IT support technician",
    "landscape gardener",
    "solicitor",
    "civil engineer",
    "financial planner",
    "project manager construction",
    "warehouse supervisor",
    "pharmacist",
    "veterinarian",
    "security guard",
    "taxi driver",
    "marine engineer",
    "builder carpenter",
    "nurse practitioner",
    "HR manager",
    "bank teller",
    "data analyst",
]

# ── Result types ───────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    query: str
    status_code: int
    latency_ms: float
    top_code: Optional[str] = None
    top_desc: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and self.error is None


# ── Worker ─────────────────────────────────────────────────────────────────

async def _classify_one(
    client: httpx.AsyncClient,
    query: str,
    mode: str,
    top_k: int,
    user_id: int,
    base_url: str,
) -> RequestResult:
    """Send one classify request and return timing + result."""
    payload = {"query": query, "mode": mode, "top_k": top_k}
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/classify", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            body = resp.json()
            results = body.get("results", [])
            top = results[0] if results else {}
            print(
                f"  [user {user_id:02d}] ✓  {latency_ms:6.0f}ms  "
                f"{top.get('anzsic_code','?'):12s}  {query!r}"
            )
            return RequestResult(
                query=query,
                status_code=200,
                latency_ms=latency_ms,
                top_code=top.get("anzsic_code"),
                top_desc=top.get("anzsic_desc"),
            )
        else:
            print(
                f"  [user {user_id:02d}] ✗  HTTP {resp.status_code}  {query!r}"
            )
            return RequestResult(
                query=query,
                status_code=resp.status_code,
                latency_ms=latency_ms,
                error=resp.text[:200],
            )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  [user {user_id:02d}] ✗  EXCEPTION  {query!r}: {exc}")
        return RequestResult(
            query=query,
            status_code=0,
            latency_ms=latency_ms,
            error=str(exc),
        )


# ── Load test orchestration ────────────────────────────────────────────────

async def run_load_test(
    queries: list[str],
    n_users: int,
    mode: str,
    top_k: int,
    base_url: str,
    ramp_secs: float,
) -> list[RequestResult]:
    """Fire n_users concurrent requests (one per virtual user).

    If n_users > len(queries), queries are cycled round-robin.

    Args:
        queries:    Pool of queries to draw from.
        n_users:    How many simultaneous virtual users.
        mode:       'high_fidelity' or 'fast'.
        top_k:      Results per request.
        base_url:   API base URL.
        ramp_secs:  Stagger user start times over this many seconds (0 = burst).
    """
    # Use a generous timeout — LLM calls can take 30s+ through a corp proxy
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Optional ramp-up: spread user starts to avoid a pure burst
        async def _user_task(user_id: int) -> RequestResult:
            if ramp_secs > 0:
                delay = (user_id / max(n_users - 1, 1)) * ramp_secs
                await asyncio.sleep(delay)
            query = queries[user_id % len(queries)]
            return await _classify_one(client, query, mode, top_k, user_id, base_url)

        tasks = [asyncio.create_task(_user_task(i)) for i in range(n_users)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return results  # type: ignore[return-value]


# ── Statistics printer ─────────────────────────────────────────────────────

def _pct(data: list[float], p: float) -> float:
    """Return pth-percentile of data (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def print_summary(results: list[RequestResult], wall_secs: float) -> None:
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    latencies = [r.latency_ms for r in ok]

    print("\n" + "═" * 62)
    print("  STRESS TEST RESULTS")
    print("═" * 62)
    print(f"  Total requests  : {len(results)}")
    print(f"  Successful      : {len(ok)}  ({100*len(ok)/max(len(results),1):.0f}%)")
    print(f"  Failed          : {len(fail)}")
    print(f"  Wall time       : {wall_secs:.1f}s")
    if ok:
        print(f"  Throughput      : {len(ok)/wall_secs:.2f} req/s")
        print()
        print(f"  Latency (ms)    min={min(latencies):.0f}  "
              f"median(P50)={_pct(latencies,50):.0f}  "
              f"P90={_pct(latencies,90):.0f}  "
              f"P95={_pct(latencies,95):.0f}  "
              f"P99={_pct(latencies,99):.0f}  "
              f"max={max(latencies):.0f}")
        print(f"  Stdev           : {statistics.stdev(latencies):.0f}ms" if len(latencies) > 1 else "")
    if fail:
        print()
        print("  Failed requests:")
        for r in fail:
            print(f"    [{r.status_code}] {r.query!r}  error={r.error}")
    print("═" * 62)

    # Emit JSON summary for easy parsing / CI integration
    summary = {
        "total": len(results),
        "ok": len(ok),
        "fail": len(fail),
        "wall_secs": round(wall_secs, 2),
        "latency_ms": {
            "min":  round(min(latencies), 1) if ok else None,
            "p50":  round(_pct(latencies, 50), 1) if ok else None,
            "p90":  round(_pct(latencies, 90), 1) if ok else None,
            "p95":  round(_pct(latencies, 95), 1) if ok else None,
            "p99":  round(_pct(latencies, 99), 1) if ok else None,
            "max":  round(max(latencies), 1) if ok else None,
        },
    }
    print("\n  JSON summary:")
    print("  " + json.dumps(summary, indent=4).replace("\n", "\n  "))


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async latency / stress test for the ANZSIC Classifier API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url",     default="http://localhost:8000",
                   help="API base URL. (default: http://localhost:8000)")
    p.add_argument("--users",   type=int, default=30,
                   help="Number of concurrent virtual users. (default: 30)")
    p.add_argument("--mode",    choices=["high_fidelity", "fast"], default="high_fidelity",
                   help="Classification mode. Use 'fast' to isolate non-LLM latency.")
    p.add_argument("--top-k",   type=int, default=5, dest="top_k",
                   help="Results per request. (default: 5)")
    p.add_argument("--queries", type=Path, default=None, dest="queries_file",
                   help="Path to a text file with one query per line. "
                        "Defaults to built-in 30 sample queries.")
    p.add_argument("--ramp",    type=float, default=0.0,
                   help="Ramp-up seconds: spread user starts over this window "
                        "(0 = pure burst). (default: 0)")
    p.add_argument("--warmup",  action="store_true",
                   help="Send one request first to warm up the pipeline before timing.")
    return p.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    # Load queries
    if args.queries_file:
        lines = args.queries_file.read_text().splitlines()
        queries = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        if not queries:
            print(f"ERROR: no queries found in {args.queries_file}")
            sys.exit(1)
    else:
        queries = _DEFAULT_QUERIES

    print(f"\nANZSIC Classifier — Stress Test")
    print(f"  URL        : {args.url}")
    print(f"  Users      : {args.users}")
    print(f"  Mode       : {args.mode}")
    print(f"  Query pool : {len(queries)} queries")
    print(f"  Ramp-up    : {args.ramp}s")

    # Readiness check
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{args.url}/readiness")
            if r.status_code != 200:
                print(f"\nWARN: readiness check failed ({r.status_code}) — proceeding anyway")
            else:
                print(f"  Server     : ready ✓")
        except Exception as exc:
            print(f"\nERROR: Cannot reach {args.url} — is the server running?\n  {exc}")
            sys.exit(1)

    # Optional single-request warmup (not timed)
    if args.warmup:
        print("\n  Warming up (1 request, not timed)…")
        async with httpx.AsyncClient(timeout=120.0) as client:
            await _classify_one(client, queries[0], args.mode, args.top_k, -1, args.url)

    # Main load test
    print(f"\n  Firing {args.users} concurrent requests…\n")
    t_wall_start = time.perf_counter()
    results = await run_load_test(
        queries=queries,
        n_users=args.users,
        mode=args.mode,
        top_k=args.top_k,
        base_url=args.url,
        ramp_secs=args.ramp,
    )
    wall_secs = time.perf_counter() - t_wall_start

    print_summary(results, wall_secs)


def main() -> None:
    args = _parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
