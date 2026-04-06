from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

from .base import EmbeddingError, TextEmbedder

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(TextEmbedder):
    """Embedding via Hugging Face sentence-transformers models."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        self._model_name = model_name
        try:
            kwargs = dict(model_kwargs)
            if device is not None:
                kwargs["device"] = device
            self._model = SentenceTransformer(model_name, **kwargs)
        except Exception as e:
            raise EmbeddingError(f"Could not load embedding model '{model_name}'") from e

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def _encode_safe(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        except Exception as e:
            raise EmbeddingError("Embedding encode failed") from e

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension
        return self._encode_safe([text.strip()])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        stripped = [(t.strip() if t else "") for t in texts]
        dim = self.dimension
        out: list[list[float]] = [[0.0] * dim for _ in stripped]
        batch_indices: list[int] = []
        batch_texts: list[str] = []
        for i, s in enumerate(stripped):
            if s:
                batch_indices.append(i)
                batch_texts.append(s)
        if not batch_texts:
            return out
        encoded = self._encode_safe(batch_texts)
        for j, idx in enumerate(batch_indices):
            out[idx] = encoded[j]
        return out
