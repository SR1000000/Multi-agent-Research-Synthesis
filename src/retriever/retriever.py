from __future__ import annotations

import struct
from concurrent.futures import ThreadPoolExecutor

from rank_bm25 import BM25Okapi

from src.memory.research.database import ResearchDatabase
from src.processing.embedder.base import TextEmbedder

from .bm25 import chunk_bm25_text, chunk_display_text, table_bm25_text, tokenize
from .types import RetrievedItem


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
        for row in chunk_rows:
            meta.append(
                (
                    "chunk",
                    row["id"],
                    row["document_id"],
                    chunk_display_text(row.get("text") or "", row.get("contextualized_text")),
                )
            )
            corpus.append(
                tokenize(chunk_bm25_text(row.get("text") or "", row.get("contextualized_text")))
            )

        for row in equation_rows:
            meta.append(("equation", row["id"], row["document_id"], row.get("text") or ""))
            corpus.append(tokenize(row.get("text") or ""))

        for row in image_rows:
            meta.append(("image", row["id"], row["document_id"], row.get("caption") or row.get("storage_path") or ""))
            # For images, we'll use the caption or storage path for keyword retrieval. Actual image fetching will be handled separately.
            corpus.append(tokenize(row.get("caption") or row.get("storage_path") or ""))

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
        for item in semantic + keyword:
            key = (item.kind, item.id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

        ranked = self.rank(merged)
        return ranked[:k]
