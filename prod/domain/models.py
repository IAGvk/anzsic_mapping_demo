"""
domain/models.py
──────────────────────────────────────────────────────────────────────────────
Pure domain objects — Pydantic models with no imports from adapters or ports.

These models are the lingua franca of the entire system:
  • adapters produce and consume them
  • services orchestrate them
  • interfaces (CLI, Streamlit, future API) serialise them

Adding a FastAPI layer later is trivial because Pydantic models serialise
directly to JSON schema — no extra DTOs or marshallers required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class SearchMode(str, Enum):
    """Controls which pipeline stages are executed."""
    FAST           = "fast"           # Stage 1 only: hybrid retrieval → RRF
    HIGH_FIDELITY  = "high_fidelity"  # Stage 1 + Stage 2: retrieval + Gemini re-rank


# ── Input ──────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Validated input to the ClassifierPipeline."""

    query: str = Field(..., min_length=1, max_length=2000,
                       description="Occupation or business description to classify")
    mode: SearchMode = Field(SearchMode.HIGH_FIDELITY,
                             description="FAST = retrieval only; HIGH_FIDELITY = + LLM re-rank")
    top_k: int = Field(5, ge=1, le=20,
                       description="Maximum number of results to return")
    retrieval_n: int = Field(20, ge=5, le=50,
                             description="RRF candidate pool size per search system")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


# ── Stage 1 output ─────────────────────────────────────────────────────────────

class Candidate(BaseModel):
    """A single ANZSIC code retrieved by the hybrid search (Stage 1)."""

    anzsic_code: str
    anzsic_desc: str

    # Hierarchical classification fields
    class_code:       Optional[str] = None
    class_desc:       Optional[str] = None
    group_code:       Optional[str] = None
    group_desc:       Optional[str] = None
    subdivision_desc: Optional[str] = None
    division_desc:    Optional[str] = None
    class_exclusions: Optional[str] = None
    enriched_text:    Optional[str] = None

    # RRF fusion metadata
    rrf_score:   float         = 0.0
    in_vector:   bool          = False
    in_fts:      bool          = False
    vector_rank: Optional[int] = None
    fts_rank:    Optional[int] = None

    @property
    def source_label(self) -> str:
        """Human-readable source badge: BOTH / VEC / FTS."""
        if self.in_vector and self.in_fts:
            return "BOTH"
        if self.in_vector:
            return "VEC"
        if self.in_fts:
            return "FTS"
        return "—"


# ── Stage 2 output ─────────────────────────────────────────────────────────────

class ClassifyResult(BaseModel):
    """A single ANZSIC code after LLM re-ranking (Stage 2).

    In FAST mode the results are assembled directly from Candidate objects
    (no reason field).  In HIGH_FIDELITY mode Gemini populates 'reason'.
    """

    rank:          int
    anzsic_code:   str
    anzsic_desc:   str
    class_desc:    Optional[str] = None
    division_desc: Optional[str] = None
    reason:        Optional[str] = None

    # Carry-through from Stage 1 for display purposes
    group_desc:      Optional[str]   = None
    subdivision_desc: Optional[str]  = None
    class_exclusions: Optional[str]  = None
    rrf_score:        Optional[float] = None
    in_vector:        Optional[bool]  = None
    in_fts:           Optional[bool]  = None
    vector_rank:      Optional[int]   = None
    fts_rank:         Optional[int]   = None

    @property
    def source_label(self) -> str:
        if self.in_vector and self.in_fts:
            return "BOTH"
        if self.in_vector:
            return "VEC"
        if self.in_fts:
            return "FTS"
        return "—"


# ── Pipeline output ────────────────────────────────────────────────────────────

class ClassifyResponse(BaseModel):
    """Complete response from ClassifierPipeline.classify().

    This is the object serialised to JSON when serving via an API endpoint.
    """

    query:               str
    mode:                str
    results:             list[ClassifyResult]
    candidates_retrieved: int
    generated_at:        datetime  = Field(
                             default_factory=lambda: datetime.now(timezone.utc)
                         )
    embed_model:         str = ""
    llm_model:           str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe floats/bools)."""
        return self.model_dump(mode="json")
