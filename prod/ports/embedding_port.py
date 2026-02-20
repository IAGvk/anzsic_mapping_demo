"""
ports/embedding_port.py
──────────────────────────────────────────────────────────────────────────────
Abstract interface for text embedding providers.

Any class that implements these methods (structural subtyping via Protocol)
is a valid EmbeddingPort — no inheritance required.

Current implementation: VertexEmbeddingAdapter (Vertex AI text-embedding-005)
To swap: write a new adapter implementing this Protocol and change container.py
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingPort(Protocol):
    """Contract for a text embedding provider."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""
        ...

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the output vectors."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Uses asymmetric retrieval task type (RETRIEVAL_QUERY) so the vector
        is optimised for matching against document embeddings.

        Args:
            text: Natural-language query string.

        Returns:
            Dense vector of floats.

        Raises:
            EmbeddingError: On API failure or empty response.
        """
        ...

    def embed_document(self, text: str, title: str = "") -> list[float]:
        """Embed a document for storage.

        Uses RETRIEVAL_DOCUMENT task type.

        Args:
            text:  Document text.
            title: Optional document title (improves quality).

        Returns:
            Dense vector of floats.
        """
        ...

    def embed_documents_batch(
        self,
        texts: list[str],
        titles: list[str] | None = None,
    ) -> list[list[float] | None]:
        """Embed multiple documents in a single API call.

        Args:
            texts:  List of document strings.
            titles: Optional parallel list of document titles.

        Returns:
            List of vectors in the same order as input.
            Individual elements may be None if that item failed.
        """
        ...
