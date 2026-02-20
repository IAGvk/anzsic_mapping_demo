"""
interfaces/cli.py
──────────────────────────────────────────────────────────────────────────────
Command-line interface for the ANZSIC classifier.

Usage:
  # Single query (high-fidelity mode, 5 results)
  python -m prod.interfaces.cli --query "Mobile Mechanic"

  # Single query, fast mode, 3 results
  python -m prod.interfaces.cli --query "plumber" --mode fast --top-k 3

  # Batch file (one query per line)
  python -m prod.interfaces.cli --file queries.txt --top-k 5

  # JSON output
  python -m prod.interfaces.cli --query "nurse" --json

  # Via installed entry-point (pyproject.toml [project.scripts])
  anzsic-classify --query "nurse"

Exit codes:
  0 — success
  1 — fatal error (auth, DB, etc.)
  2 — argument error
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from prod.domain.models import SearchMode, SearchRequest
from prod.services.container import get_pipeline

logger = logging.getLogger(__name__)


# ── Argument parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anzsic-classify",
        description="Classify an occupation or business description into ANZSIC codes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--query", "-q",
        metavar="TEXT",
        help="Single query to classify.",
    )
    p.add_argument(
        "--file", "-f",
        metavar="FILE",
        type=Path,
        help="Path to a text file with one query per line.",
    )
    p.add_argument(
        "--mode", "-m",
        choices=["fast", "high_fidelity"],
        default="high_fidelity",
        help="Search mode. 'fast' skips LLM re-ranking. (default: high_fidelity)",
    )
    p.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results to return per query. (default: 5)",
    )
    p.add_argument(
        "--candidates", "-c",
        type=int,
        default=20,
        dest="retrieval_n",
        help="Retrieval pool size (Stage 1 candidates). (default: 20)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


# ── Formatting helpers ─────────────────────────────────────────────────────

def _print_results_text(response) -> None:
    """Pretty-print a ClassifyResponse to stdout."""
    print(f"\n{'─' * 60}")
    print(f"Query : {response.query}")
    print(f"Mode  : {response.mode}  |  Candidates: {response.candidates_retrieved}")
    print(f"{'─' * 60}")
    for r in response.results:
        print(f"  #{r.rank}  [{r.anzsic_code}] {r.anzsic_desc}")
        if r.class_desc:
            print(f"       Class: {r.class_desc}")
        if r.division_desc:
            print(f"       Division: {r.division_desc}")
        if r.reason:
            print(f"       Reason: {r.reason}")
    print()


def _print_results_json(response) -> None:
    """Print a ClassifyResponse as JSON to stdout."""
    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))


# ── Main logic ─────────────────────────────────────────────────────────────

def _load_queries_from_file(path: Path) -> list[str]:
    """Read queries from a text file, one per line, skip blank/comment lines."""
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def run(args: argparse.Namespace) -> int:
    """Execute classification for the given arguments.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    # Determine queries
    if args.query:
        queries = [args.query]
    elif args.file:
        queries = _load_queries_from_file(args.file)
    else:
        print("ERROR: provide --query or --file", file=sys.stderr)
        return 2

    mode = SearchMode.FAST if args.mode == "fast" else SearchMode.HIGH_FIDELITY
    printer = _print_results_json if args.json_output else _print_results_text

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        logger.exception("Failed to initialise pipeline")
        print(f"ERROR: Pipeline initialisation failed: {exc}", file=sys.stderr)
        return 1

    exit_code = 0
    for query in queries:
        try:
            request = SearchRequest(
                query=query,
                mode=mode,
                top_k=args.top_k,
                retrieval_n=args.retrieval_n,
            )
            response = pipeline.classify(request)
            printer(response)
        except Exception as exc:
            logger.exception("Classification failed for query %r", query)
            print(f"ERROR [{query!r}]: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


def main() -> None:
    """Entry point for the anzsic-classify console script."""
    parser = _build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    if not args.query and not args.file:
        parser.print_help()
        sys.exit(2)

    sys.exit(run(args))


if __name__ == "__main__":
    main()
