from __future__ import annotations

from typing import Any

from .base import EmbeddingError, TextEmbedder

__all__ = [
    "EmbeddingError",
    "TextEmbedder",
    "SentenceTransformerEmbedder",
    "get_text_embedder",
]


def __getattr__(name: str) -> Any:
    if name == "SentenceTransformerEmbedder":
        from .sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder
    if name == "get_text_embedder":
        from .provider import get_text_embedder

        return get_text_embedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
