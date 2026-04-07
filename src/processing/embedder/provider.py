from __future__ import annotations

from typing import Any

from .base import TextEmbedder


def get_text_embedder(name: str = "sentence_transformers", **kwargs: Any) -> TextEmbedder:
    """
    Construct a TextEmbedder by provider name.
    kwargs are passed to the concrete class (e.g. model_name=...).
    """
    if name == "sentence_transformers":
        from .sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(
        f"Unknown embedder provider '{name}'. Available: ['sentence_transformers']"
    )
