# ADR 004 — Reciprocal Rank Fusion (RRF)

<span class="adr-status-accepted">Accepted</span> &nbsp; **Date:** 2025 &nbsp; **Deciders:** CEP AI Team

---

## Context

Stage 1 retrieval runs two independent search systems:

- **Vector ANN search** — excellent at colloquial queries, synonyms, semantic
  similarity. Poor at exact-match acronyms and obscure technical codes.
- **Full-text search (FTS)** — excellent at exact keyword matches. Poor at
  colloquial or informal descriptions.

We needed a fusion method to combine the results of both systems into a single
ranked list of candidates for Stage 2.

Options evaluated:

| Method | Description | Learning required? |
|---|---|---|
| **Reciprocal Rank Fusion (RRF)** | Score = Σ 1/(k + rank) for each system | ❌ None |
| Score normalisation + weighted sum | Normalise scores, tune weights | ❌ None (but needs tuning) |
| Learned re-ranker (cross-encoder) | ML model trained on query-doc pairs | ✅ Labelled data needed |
| Interleaved results | Alternate rows from each system | ❌ None |

---

## Decision

We chose **Reciprocal Rank Fusion with k = 60** (the standard published value).

The formula: $\text{score}(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}$

Reasons:

1. **No training data required** — We have no labelled query-code pairs.
   RRF requires only ranks, not calibrated scores, so it works out-of-the-box.

2. **Robust to score scale differences** — Vector cosine distances and FTS
   `ts_rank_cd` scores are on completely different scales. RRF uses only
   ordinal rank, making it immune to this mismatch.

3. **Documents in both systems get a bonus** — A code that ranks 3rd in vector
   search AND 5th in FTS will outscore a code that ranks 1st in only one system.
   This is exactly the behaviour we want — cross-system agreement is a strong
   quality signal.

4. **Pure Python, testable in isolation** — `compute_rrf()` is a standalone
   function with no I/O. It has 12 dedicated unit tests that verify exact score
   arithmetic, edge cases, and provenance flags.

---

## The k constant

$k = 60$ was established empirically in the original RRF paper (Cormack et al.,
2009) and remains the standard. A higher $k$ flattens the score distribution
(rank differences matter less); a lower $k$ amplifies top-rank advantage.

We tested $k \in \{10, 30, 60, 120\}$ on 20 representative queries from
`queries.txt` and found $k = 60$ produced the best Stage 2 input quality
(measured by Gemini selecting from the top 5 candidates ≥ 90% of the time).

---

## Consequences

**Positive:**

- Zero hyperparameter tuning needed when adding new data or changing models
- Provenance tracking: `in_vector`, `in_fts`, `vector_rank`, `fts_rank` fields
  on `Candidate` objects show which systems contributed to each result
- 12 unit tests run in < 1 ms — the most thoroughly tested component

**Negative / trade-offs:**

- RRF cannot leverage score magnitudes — a vector result with cosine similarity
  0.99 is treated the same as one with 0.62 if both rank 1st
- FTS returning zero results (common for colloquial queries) means RRF degrades
  to vector-only ranking — acceptable but suboptimal

**When to revisit:**

If we accumulate labelled query-code pairs, a learned cross-encoder re-ranker
would outperform RRF. The service interface (`HybridRetriever.retrieve()`)
returns `list[Candidate]` regardless of the fusion method — swapping RRF for
a learned model requires changing only the fusion step inside that method.
