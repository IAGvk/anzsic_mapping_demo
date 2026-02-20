"""
interfaces/streamlit_app.py
──────────────────────────────────────────────────────────────────────────────
Streamlit UI for the ANZSIC classifier.

Run:
  streamlit run prod/interfaces/streamlit_app.py

Features:
  • Single query mode: text input → metrics → Cards / Table / JSON tabs
  • Batch mode: .txt file upload → progress bar → aggregated results
  • Fast mode (Stage 1 only) vs High Fidelity mode (Stage 1 + Gemini)
  • CSV download of results
  • Colour scheme: navy sidebar / light grey canvas (matches original app.py)
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────
# Allow running from the repo root with: streamlit run prod/interfaces/streamlit_app.py
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prod.domain.models import SearchMode, SearchRequest
from prod.services.container import get_pipeline

logger = logging.getLogger(__name__)

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANZSIC Classifier",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #1a3a5c 100%);
    }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #a8c4e0 !important; }

    /* Main canvas */
    .stApp { background-color: #f4f6f9; }

    /* Result cards */
    .anzsic-card {
        background: white;
        border-left: 5px solid #2563eb;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .anzsic-card.rank-1 { border-left-color: #16a34a; }
    .anzsic-card.rank-2 { border-left-color: #2563eb; }
    .anzsic-card .code { font-size: 0.78em; color: #64748b; font-family: monospace; }
    .anzsic-card .title { font-size: 1.05em; font-weight: 600; color: #1e293b; }
    .anzsic-card .reason { font-size: 0.87em; color: #475569; margin-top: 6px; }
    .anzsic-card .meta { font-size: 0.8em; color: #94a3b8; margin-top: 4px; }

    /* Metric boxes */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Backend singleton ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising ANZSIC classifier…")
def _load_pipeline():
    """Loads and caches the ClassifierPipeline for the lifetime of the app."""
    return get_pipeline()


# ── Sidebar ────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict:
    """Render sidebar controls and return selected options."""
    with st.sidebar:
        st.markdown("## ⚙️ Options")
        st.markdown("---")

        search_type = st.selectbox(
            "Search Type",
            ["Single Query", "Batch (file upload)"],
            key="search_type",
        )

        search_mode = st.selectbox(
            "Search Mode",
            ["High Fidelity (+ Gemini)", "Fast (retrieval only)"],
            key="search_mode",
        )

        st.selectbox(
            "ANZSIC Type",
            ["6-digit (unit groups)"],
            key="anzsic_type",
            disabled=True,
            help="Only 6-digit ANZSIC codes are currently supported.",
        )

        mode_enum = (
            SearchMode.HIGH_FIDELITY
            if "High" in search_mode
            else SearchMode.FAST
        )

        if mode_enum == SearchMode.HIGH_FIDELITY:
            top_k = st.number_input(
                "Max Results (auto in HF)",
                min_value=1,
                max_value=10,
                value=5,
                key="top_k",
                help="Gemini will select the best matches up to this limit.",
            )
        else:
            top_k = st.number_input(
                "Max Results",
                min_value=1,
                max_value=20,
                value=5,
                key="top_k_fast",
            )

        retrieval_n = st.slider(
            "Retrieval Pool",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="retrieval_n",
            help="Number of Stage 1 candidates before re-ranking.",
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#8facc8'>ANZSIC Classifier v1.0<br>"
            "Vertex AI · pgvector · Gemini</small>",
            unsafe_allow_html=True,
        )

    return {
        "search_type": search_type,
        "mode": mode_enum,
        "top_k": int(top_k),
        "retrieval_n": retrieval_n,
    }


# ── Result rendering ───────────────────────────────────────────────────────

def _results_to_df(results) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Rank": r.rank,
                "ANZSIC Code": r.anzsic_code,
                "ANZSIC Description": r.anzsic_desc,
                "Class": r.class_desc or "",
                "Division": r.division_desc or "",
                "Reason": r.reason or "",
            }
        )
    return pd.DataFrame(rows)


def _render_cards(results) -> None:
    for r in results:
        rank_class = "rank-1" if r.rank == 1 else "rank-2" if r.rank == 2 else ""
        st.markdown(
            f"""
            <div class="anzsic-card {rank_class}">
                <div class="code">#{r.rank} &nbsp;·&nbsp; {r.anzsic_code}</div>
                <div class="title">{r.anzsic_desc}</div>
                <div class="meta">{r.class_desc or ''} &nbsp;·&nbsp; {r.division_desc or ''}</div>
                <div class="reason">{r.reason or ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_response_tabs(response, query_label: str = "") -> None:
    """Render Cards / Table / JSON tabs for a single ClassifyResponse."""
    tab_cards, tab_table, tab_json = st.tabs(["🃏 Cards", "📋 Table", "{ } JSON"])

    with tab_cards:
        _render_cards(response.results)

    with tab_table:
        df = _results_to_df(response.results)
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode()
        label = f"results_{query_label}.csv" if query_label else "results.csv"
        st.download_button("⬇ Download CSV", csv_bytes, file_name=label, mime="text/csv")

    with tab_json:
        st.json(response.to_dict())


# ── Single query mode ──────────────────────────────────────────────────────

def _run_single(options: dict) -> None:
    st.markdown("### 🔍 Enter a query")
    query = st.text_input(
        "Occupation or business description",
        placeholder="e.g.  Mobile Mechanic, café owner, registered nurse …",
        key="query_input",
    )
    run_btn = st.button("Classify", type="primary", use_container_width=False)

    if not (run_btn and query.strip()):
        st.info("Enter a description above and press **Classify**.")
        return

    pipeline = _load_pipeline()
    request = SearchRequest(
        query=query.strip(),
        mode=options["mode"],
        top_k=options["top_k"],
        retrieval_n=options["retrieval_n"],
    )

    with st.spinner("Classifying …"):
        t0 = time.perf_counter()
        response = pipeline.classify(request)
        elapsed = time.perf_counter() - t0

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Results", len(response.results))
    c2.metric("Candidates", response.candidates_retrieved)
    c3.metric("Latency", f"{elapsed:.2f}s")
    c4.metric("Mode", response.mode.replace("_", " ").title())

    st.markdown("---")
    _render_response_tabs(response, query_label=query[:30].replace(" ", "_"))


# ── Batch mode ─────────────────────────────────────────────────────────────

def _run_batch(options: dict) -> None:
    st.markdown("### 📂 Upload a query file")
    st.caption("Plain text file, one query per line. Lines starting with # are ignored.")

    uploaded = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded is None:
        return

    raw = uploaded.read().decode("utf-8")
    queries = [l.strip() for l in raw.splitlines() if l.strip() and not l.startswith("#")]

    if not queries:
        st.warning("No queries found in the uploaded file.")
        return

    with st.expander(f"Preview ({len(queries)} queries)", expanded=False):
        for i, q in enumerate(queries[:10], 1):
            st.markdown(f"{i}. {q}")
        if len(queries) > 10:
            st.caption(f"… and {len(queries) - 10} more")

    if not st.button("Run Batch", type="primary"):
        return

    pipeline = _load_pipeline()
    all_rows: list[dict] = []
    progress = st.progress(0, text="Starting …")
    status = st.empty()

    for idx, query in enumerate(queries):
        status.markdown(f"**[{idx + 1}/{len(queries)}]** Classifying: *{query}*")
        try:
            request = SearchRequest(
                query=query,
                mode=options["mode"],
                top_k=options["top_k"],
                retrieval_n=options["retrieval_n"],
            )
            response = pipeline.classify(request)
            for r in response.results:
                all_rows.append(
                    {
                        "Query": query,
                        "Rank": r.rank,
                        "ANZSIC Code": r.anzsic_code,
                        "ANZSIC Description": r.anzsic_desc,
                        "Class": r.class_desc or "",
                        "Division": r.division_desc or "",
                        "Reason": r.reason or "",
                    }
                )
        except Exception as exc:
            logger.exception("Batch query failed: %r", query)
            all_rows.append({"Query": query, "Error": str(exc)})

        progress.progress((idx + 1) / len(queries), text=f"{idx + 1}/{len(queries)} complete")

    status.markdown(f"✅ Batch complete — {len(queries)} queries processed.")
    progress.empty()

    if not all_rows:
        st.error("No results generated.")
        return

    df = pd.DataFrame(all_rows)
    st.markdown("---")
    tab_table, tab_json = st.tabs(["📋 Table", "{ } JSON"])

    with tab_table:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False).encode(),
            file_name="batch_results.csv",
            mime="text/csv",
        )

    with tab_json:
        st.json(df.to_dict(orient="records"))


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🏭 ANZSIC Occupation Classifier")
    st.caption(
        "Hybrid vector + FTS retrieval with optional Gemini re-ranking. "
        "Powered by Vertex AI · pgvector · PostgreSQL."
    )

    options = _render_sidebar()

    if options["search_type"] == "Single Query":
        _run_single(options)
    else:
        _run_batch(options)


if __name__ == "__main__":
    main()
