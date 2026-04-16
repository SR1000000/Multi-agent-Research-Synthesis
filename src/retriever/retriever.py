from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from src.processing.embedder.base import TextEmbedder

from .retrieval_text import preferred_text
from .types import RetrievedItem

if TYPE_CHECKING:
    from src.memory.research.database import ResearchDatabase


class Retriever:
    def __init__(self, db: ResearchDatabase, embedder: TextEmbedder) -> None:
        self._db = db
        self._embedder = embedder

    def semantic_retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        if k <= 0:
            return []
        q = (query or "").strip()
        if not q:
            return []

        dim_e = self._embedder.dimension
        dim_db = self._db.config.vec_dimensions
        if dim_e != dim_db:
            raise ValueError(
                f"Embedder dimension {dim_e} does not match database vec_dimensions {dim_db}"
            )
        vec = self._embedder.embed_query(q)
        blob = struct.pack(f"{dim_db}f", *vec)
        rows = self._db.knn_text_chunks_by_embedding(blob, k)
        out: list[RetrievedItem] = []
        for row in rows:
            body = preferred_text(row.get("text") or "", row.get("contextualized_text"))
            dist = row.get("dist")
            out.append(
                RetrievedItem(
                    kind="chunk",
                    id=row["id"],
                    document_id=row["document_id"],
                    text=body,
                    score=float(dist) if dist is not None else None,
                )
            )
        return out

    def keywords_retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        if k <= 0:
            return []
        q = (query or "").strip()
        if not q:
            return []
        return self._db.query_artifact_search(q, k)

    def rank(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        return items

    def fusion_retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        if k <= 0:
            return []
        q = (query or "").strip()
        if not q:
            return []

        semantic = self.semantic_retrieve(q, k)
        keyword = self.keywords_retrieve(q, k)

        merged: list[RetrievedItem] = []
        seen: set[tuple[str, str]] = set()
        # Interleave both retrieval streams so keyword-only artifact kinds
        # are not dropped when semantic results already fill top-k.
        combined: list[RetrievedItem] = []
        max_len = max(len(semantic), len(keyword))
        for idx in range(max_len):
            if idx < len(semantic):
                combined.append(semantic[idx])
            if idx < len(keyword):
                combined.append(keyword[idx])

        for item in combined:
            key = (item.kind, item.id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

        ranked = self.rank(merged)
        return ranked[:k]
