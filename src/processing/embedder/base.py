from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingError(RuntimeError):
    """Raised when embedding fails."""


class TextEmbedder(ABC):
    """Text-only embedding: vectors in, no storage or document types."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector size produced by this embedder."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query or document string; returns one vector."""

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings; default is one call per string."""
        if not texts:
            return []
        return [self.embed_query(t) for t in texts]
