"""
config/settings.py
──────────────────────────────────────────────────────────────────────────────
Single source of truth for all tuneable parameters.

All values can be overridden via environment variables or a .env file placed
at the project root.  The frozen dataclass ensures settings are never mutated
at runtime.

To swap providers, change the relevant env var — no code edits required:
  GCP_EMBED_MODEL   → swap embedding model
  GCP_GEMINI_MODEL  → swap LLM model
  DB_DSN            → swap database
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_path(key: str, default: Path) -> Path:
    return Path(os.getenv(key, str(default)))


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    # ── Provider selection ──────────────────────────────────────────────────
    # Set EMBED_PROVIDER=openai or LLM_PROVIDER=openai to switch from Vertex AI.
    # Valid values: "vertex" | "openai"
    embed_provider: str = field(
        default_factory=lambda: _env("EMBED_PROVIDER", "vertex")
    )
    llm_provider: str = field(
        default_factory=lambda: _env("LLM_PROVIDER", "vertex")
    )

    # ── OpenAI ─────────────────────────────────────────────────────────────
    openai_api_key: str = field(
        default_factory=lambda: _env("OPENAI_API_KEY", "")
    )
    # text-embedding-3-small: 1536-dim natively; set EMBED_DIM to match.
    # text-embedding-3-large: 3072-dim natively or reduced via dimensions param.
    openai_embed_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    )
    openai_llm_model: str = field(
        default_factory=lambda: _env("OPENAI_LLM_MODEL", "gpt-4o")
    )

    # ── GCP / Vertex AI ────────────────────────────────────────────────────
    gcp_project_id: str = field(
        default_factory=lambda: _env("GCP_PROJECT_ID", "top-arc-65ca")
    )
    gcp_location_id: str = field(
        default_factory=lambda: _env("GCP_LOCATION_ID", "australia-southeast1")
    )
    gcp_embed_model: str = field(
        default_factory=lambda: _env("GCP_EMBED_MODEL", "text-embedding-005")
    )
    gcp_gemini_model: str = field(
        default_factory=lambda: _env("GCP_GEMINI_MODEL", "gemini-2.5-flash")
    )
    gcloud_path: str = field(
        default_factory=lambda: _env(
            "GCLOUD_PATH",
            "/Users/s748779/gemini_local/google-cloud-sdk/bin/gcloud",
        )
    )

    # ── Network ────────────────────────────────────────────────────────────
    https_proxy: str = field(
        default_factory=lambda: _env("HTTPS_PROXY", "cloudproxy.auiag.corp:8080")
    )

    # ── Database ───────────────────────────────────────────────────────────
    db_dsn: str = field(
        default_factory=lambda: _env("DB_DSN", "dbname=anzsic_db")
    )

    # ── Retrieval pipeline ─────────────────────────────────────────────────
    rrf_k: int = field(
        default_factory=lambda: _env_int("RRF_K", 60)
    )
    retrieval_n: int = field(
        default_factory=lambda: _env_int("RETRIEVAL_N", 20)
    )
    top_k: int = field(
        default_factory=lambda: _env_int("TOP_K", 5)
    )
    embed_dim: int = field(
        default_factory=lambda: _env_int("EMBED_DIM", 768)
    )
    embed_batch_size: int = field(
        default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 50)
    )

    # ── Data paths ─────────────────────────────────────────────────────────
    master_csv_path: Path = field(
        default_factory=lambda: _env_path(
            "MASTER_CSV_PATH",
            Path(__file__).parent.parent.parent / "anzsic_master.csv",
        )
    )

    # ── HTTP timeouts (seconds) ────────────────────────────────────────────
    embed_timeout: int = field(default_factory=lambda: _env_int("EMBED_TIMEOUT", 30))
    llm_timeout: int   = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 90))
    embed_retries: int = field(default_factory=lambda: _env_int("EMBED_RETRIES", 3))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance.

    Use this everywhere instead of instantiating Settings() directly —
    it guarantees a single object is shared across the entire process.
    """
    return Settings()
