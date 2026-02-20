"""
domain/exceptions.py
──────────────────────────────────────────────────────────────────────────────
Custom exception hierarchy.

All exceptions are rooted at ANZSICError so callers can catch broadly
(except ANZSICError) or narrowly (except EmbeddingError).

When adding a FastAPI layer, map these to appropriate HTTP status codes:
  AuthenticationError → 401
  DatabaseError       → 503
  EmbeddingError      → 502
  LLMError            → 502
  ValidationError     → 422 (Pydantic handles this automatically)
"""
from __future__ import annotations


class ANZSICError(Exception):
    """Base exception for all application errors."""


class ConfigurationError(ANZSICError):
    """Raised when required configuration is missing or invalid."""


class AuthenticationError(ANZSICError):
    """Raised when GCP auth token acquisition fails."""


class EmbeddingError(ANZSICError):
    """Raised when the embedding API call fails or returns invalid output."""


class LLMError(ANZSICError):
    """Raised when the LLM API call fails or returns unparseable output."""


class DatabaseError(ANZSICError):
    """Raised when a database operation fails."""


class RetrievalError(ANZSICError):
    """Raised when Stage 1 retrieval returns no usable candidates."""


class RerankError(ANZSICError):
    """Raised when Stage 2 re-ranking fails after all retries."""
