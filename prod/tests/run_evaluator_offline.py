"""
tests/run_evaluator_offline.py
──────────────────────────────────────────────────────────────────────────────
Offline evaluation runner — scores frozen LLM responses with zero LLM/DB calls.

Loads saved responses from tests/fixtures/frozen_responses.json, reconstructs
the domain objects, runs ANZSICEvaluator, and prints a colour-coded report.

Usage (from repo root, venv active):
  python -m prod.tests.run_evaluator_offline
  python -m prod.tests.run_evaluator_offline --fixture path/to/other.json
  python -m prod.tests.run_evaluator_offline --csv path/to/anzsic_master.csv
  python -m prod.tests.run_evaluator_offline --json     # machine-readable output

Capture live responses for later offline evaluation
────────────────────────────────────────────────────
If you have a working LLM session and want to save responses to evaluate later:

    import json
    from prod.services.container import get_pipeline
    from prod.domain.models import SearchRequest

    pipeline = get_pipeline()
    saves = []
    for q in ["plumber", "nurse", "software engineer"]:
        resp = pipeline.classify(SearchRequest(query=q))
        saves.append({
            "query": q,
            "top_k": 5,
            "candidates": [c.model_dump() for c in resp.candidates_retrieved],
            "results":    [r.model_dump() for r in resp.results],
        })
    Path("my_responses.json").write_text(json.dumps(saves, indent=2))

Then run:
    python -m prod.tests.run_evaluator_offline --fixture my_responses.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Resolve repo root ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prod.domain.models import Candidate, ClassifyResult
from prod.services.evaluator import ANZSICEvaluator

# ── Defaults ───────────────────────────────────────────────────────────────
_DEFAULT_FIXTURE = (
    Path(__file__).parent / "fixtures" / "frozen_responses.json"
)
_DEFAULT_CSV = (
    Path(__file__).parent.parent.parent / "anzsic_master.csv"
)

# ── ANSI colours (disabled when --no-colour or piped) ─────────────────────
_COLOUR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text

def _green(t: str)  -> str: return _c(t, "32")
def _yellow(t: str) -> str: return _c(t, "33")
def _red(t: str)    -> str: return _c(t, "31")
def _bold(t: str)   -> str: return _c(t, "1")
def _dim(t: str)    -> str: return _c(t, "2")

def _colour_score(score: float) -> str:
    s = f"{score:.3f}"
    if score >= 0.80:
        return _green(s)
    if score >= 0.50:
        return _yellow(s)
    return _red(s)

# Column width without ANSI escape chars
_COL_W = 13
# Extra bytes ANSI colour adds per coloured cell
_ANSI_PAD = 10 if _COLOUR else 0


# ── Loader helpers ─────────────────────────────────────────────────────────

def _load_candidate(d: dict) -> Candidate:
    return Candidate(
        anzsic_code=d["anzsic_code"],
        anzsic_desc=d.get("anzsic_desc", ""),
        rrf_score=float(d.get("rrf_score", 0.0)),
        in_vector=bool(d.get("in_vector", False)),
        in_fts=bool(d.get("in_fts", False)),
        vector_rank=d.get("vector_rank"),
        fts_rank=d.get("fts_rank"),
    )


def _load_result(d: dict) -> ClassifyResult:
    return ClassifyResult(
        rank=int(d["rank"]),
        anzsic_code=d["anzsic_code"],
        anzsic_desc=d.get("anzsic_desc", ""),
        class_desc=d.get("class_desc"),
        division_desc=d.get("division_desc"),
        reason=d.get("reason"),
        score=d.get("score"),
        group_desc=d.get("group_desc"),
        subdivision_desc=d.get("subdivision_desc"),
        rrf_score=d.get("rrf_score"),
        in_vector=d.get("in_vector"),
        in_fts=d.get("in_fts"),
    )


# ── Printer helpers ────────────────────────────────────────────────────────

_DIM_NAMES = ["completeness", "coherence", "correctness", "relevance", "groundedness"]

def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def print_header() -> None:
    cols = "  ".join(f"{n:>{_COL_W}}" for n in _DIM_NAMES)
    print()
    print(_bold(f"{'Query':<26}  {cols}   {'Overall':>{_COL_W}}  Flags"))
    print("─" * 120)


def print_row(query: str, scenario: str, report) -> None:
    scores = [getattr(report, dim) for dim in _DIM_NAMES]
    scored_cols = "  ".join(
        f"{_colour_score(s):>{_COL_W + _ANSI_PAD}}" for s in scores
    )
    flag_summary = f"{len(report.flags)} flag(s)" if report.flags else _green("✓")
    label = f"{query[:24]:<24}"
    print(f"  {label}  {scored_cols}   {_colour_score(report.overall):>{_COL_W + _ANSI_PAD}}  {flag_summary}")
    print(_dim(f"    [{scenario[:90]}]"))


def print_detail(report) -> None:
    """Print per-dimension breakdowns and flags."""
    dims = ["completeness", "coherence", "correctness", "relevance", "groundedness"]
    for dim in dims:
        detail = report.details.get(dim, {})
        detail_str = "  ".join(f"{k}={v}" for k, v in detail.items()
                               if not isinstance(v, (list, dict)))
        bar = _bar(getattr(report, dim))
        print(f"    {_bold(dim):>14}  {bar}  {_colour_score(getattr(report, dim))}  {_dim(detail_str)}")
    if report.flags:
        for flag in report.flags:
            print(f"    {_red('⚑ ' + flag)}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def run(fixture_path: Path, csv_path: Path, verbose: bool, as_json: bool) -> None:
    # Load fixture
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}")
        sys.exit(1)
    items: list[dict] = json.loads(fixture_path.read_text())
    print(f"\nLoaded {len(items)} frozen response(s) from {fixture_path.name}")

    # Build evaluator
    evaluator = ANZSICEvaluator(csv_path)
    print(f"Master CSV: {csv_path.name}  ({len(evaluator._master)} codes loaded)")

    all_reports = []

    if not as_json:
        print_header()

    for item in items:
        query     = item["query"]
        top_k     = item.get("top_k", 5)
        scenario  = item.get("_scenario", "")
        candidates = [_load_candidate(c) for c in item.get("candidates", [])]
        results    = [_load_result(r) for r in item.get("results", [])]

        report = evaluator.evaluate(
            query=query,
            results=results,
            candidates=candidates,
            top_k=top_k,
        )

        all_reports.append({
            "query":        query,
            "scenario":     scenario,
            "report":       report,
            "n_results":    len(results),
            "n_candidates": len(candidates),
        })

        if not as_json:
            print_row(query, scenario, report)
            if verbose:
                print_detail(report)

    # ── Summary stats ──────────────────────────────────────────────────────
    if not as_json:
        overalls = [x["report"].overall for x in all_reports]
        print()
        print("─" * 120)
        print(_bold(f"  Summary across {len(all_reports)} responses:"))
        print(f"    Mean overall:   {sum(overalls)/len(overalls):.3f}")
        print(f"    Best:           {max(overalls):.3f}  ({all_reports[overalls.index(max(overalls))]['query']})")
        print(f"    Worst:          {min(overalls):.3f}  ({all_reports[overalls.index(min(overalls))]['query']})")
        flagged = [x for x in all_reports if x["report"].flags]
        print(f"    Responses with flags: {len(flagged)} / {len(all_reports)}")
        for x in flagged:
            for flag in x["report"].flags:
                print(f"      [{x['query']}]  {_red(flag)}")

        # Per-dimension averages
        print()
        print(_bold("  Per-dimension averages:"))
        for dim in _DIM_NAMES:
            avg = sum(getattr(x["report"], dim) for x in all_reports) / len(all_reports)
            print(f"    {dim:>14}:  {_bar(avg, 30)}  {_colour_score(avg)}")
        print()

    else:
        # Machine-readable JSON output
        out = []
        for x in all_reports:
            r = x["report"]
            out.append({
                "query":    x["query"],
                "scenario": x["scenario"],
                "scores": {
                    "completeness": r.completeness,
                    "coherence":    r.coherence,
                    "correctness":  r.correctness,
                    "relevance":    r.relevance,
                    "groundedness": r.groundedness,
                    "overall":      r.overall,
                },
                "flags":   r.flags,
                "details": r.details,
            })
        print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score frozen LLM responses with ANZSICEvaluator (offline, no LLM/DB).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--fixture", type=Path, default=_DEFAULT_FIXTURE,
                    help=f"Path to frozen_responses.json. (default: {_DEFAULT_FIXTURE.name})")
    ap.add_argument("--csv", type=Path, default=_DEFAULT_CSV,
                    help=f"Path to anzsic_master.csv. (default: {_DEFAULT_CSV.name})")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print per-dimension breakdowns for every response.")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="Emit machine-readable JSON instead of the table.")
    args = ap.parse_args()
    run(args.fixture, args.csv, args.verbose, args.as_json)


if __name__ == "__main__":
    main()
