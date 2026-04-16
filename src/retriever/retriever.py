from __future__ import annotations

import struct
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from src.processing.embedder.base import TextEmbedder

from .bm25 import chunk_display_text, tokenize
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
            body = chunk_display_text(row.get("text") or "", row.get("contextualized_text"))
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

        chunk_rows = self._db.fetch_all_text_chunks_for_retrieval()
        table_rows = self._db.fetch_all_tables_for_retrieval()
        equation_rows = self._db.fetch_all_equations_for_retrieval()
        image_rows = self._db.fetch_all_images_for_retrieval()

        corpus: list[list[str]] = []
        meta: list[tuple[str, str, str, str]] = []

        def _preferred_text(row: dict, *fallback_keys: str) -> str:
            contextualized = (row.get("contextualized_text") or "").strip()
            if contextualized:
                return contextualized
            for key in fallback_keys:
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return ""

        for row in chunk_rows:
            chosen_text = _preferred_text(row, "text")
            meta.append(
                (
                    "chunk",
                    row["id"],
                    row["document_id"],
                    chunk_display_text(row.get("text") or "", row.get("contextualized_text")),
                )
            )
            corpus.append(
                tokenize(chosen_text)
            )

        for row in table_rows:
            chosen_text = _preferred_text(row, "content")
            meta.append(("table", row["id"], row["document_id"], chosen_text))
            corpus.append(tokenize(chosen_text))

        for row in equation_rows:
            chosen_text = _preferred_text(row, "text")
            meta.append(("equation", row["id"], row["document_id"], chosen_text))
            corpus.append(tokenize(chosen_text))

        for row in image_rows:
            chosen_text = _preferred_text(row, "caption", "storage_path")
            meta.append(("image", row["id"], row["document_id"], chosen_text))
            # For images, we'll use the caption or storage path for keyword retrieval. Actual image fetching will be handled separately.
            corpus.append(tokenize(chosen_text))

        if not corpus:
            return []

        q_tokens = tokenize(q)
        if not q_tokens:
            return []

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        out: list[RetrievedItem] = []
        for i in ranked:
            kind, doc_id, document_id, text = meta[i]
            item = RetrievedItem(
                kind=kind,  # This will now be "chunk", "table", "equation", or "image"
                id=doc_id,
                document_id=document_id,
                text=text,
                score=float(scores[i]),
            )
            out.append(item)
        return out

    def rank(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        return items

    def fusion_retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        if k <= 0:
            return []
        q = (query or "").strip()
        if not q:
            return []

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_sem = pool.submit(self.semantic_retrieve, q, k)
            f_kw = pool.submit(self.keywords_retrieve, q, k)
            semantic = f_sem.result()
            keyword = f_kw.result()

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
