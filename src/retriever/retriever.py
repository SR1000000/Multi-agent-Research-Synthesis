from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from src.processing.embedder.base import TextEmbedder

if TYPE_CHECKING:
    from src.memory.research.database import ResearchDatabase


@dataclass
class RetrievedItem:
    """Identity + ranking for one hit. id is the artifact PK in the base table."""

    kind: Literal["chunk", "table", "equation", "image"]
    id: str
    document_id: str
    score: float | None = None


def _keyword_row_to_retrieved_item(row: dict[str, Any]) -> RetrievedItem:
    kind = cast(
        Literal["chunk", "table", "equation", "image"],
        str(row["kind"]),
    )
    score_raw = row.get("score")
    score = float(score_raw) if score_raw is not None else None
    return RetrievedItem(
        kind=kind,
        id=str(row["item_id"]),
        document_id=str(row["document_id"]),
        score=score,
    )


def _semantic_row_to_retrieved_item(row: dict[str, Any]) -> RetrievedItem:
    score_raw = row.get("dist")
    score = float(score_raw) if score_raw is not None else None
    return RetrievedItem(
        kind="chunk",
        id=str(row["id"]),
        document_id=str(row["document_id"]),
        score=score,
    )


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
        return [_semantic_row_to_retrieved_item(dict(r)) for r in rows]

    def keywords_retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        if k <= 0:
            return []
        q = (query or "").strip()
        if not q:
            return []
        rows = self._db.query_artifact_search(q, k)
        return [_keyword_row_to_retrieved_item(dict(r)) for r in rows]

    def rank(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        # Todo: have a re-rank model to organize this
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
