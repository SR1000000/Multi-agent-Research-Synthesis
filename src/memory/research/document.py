from __future__ import annotations

import json
import struct
from dataclasses import asdict
from pathlib import Path

from src.processing.document.schema import (
    ExtractionResult,
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    PaperMetadata,
)


def _load_ordered_chunk_rows(db, doc_id):
    """Return chunk rows in document order, falling back to insertion order."""
    return db.connection.execute(
        """
        SELECT *
        FROM text_chunks
        WHERE document_id = ?
        ORDER BY
            COALESCE(CAST(json_extract(meta_data, '$.chunk_index') AS INTEGER), rowid),
            rowid
        """,
        (doc_id,),
    ).fetchall()


def document_exists(db, content_hash: str) -> bool:
    """Returns True if a document with the given content hash already exists."""
    row = db._conn.execute(
        "SELECT 1 FROM documents WHERE content_hash = ? LIMIT 1", (content_hash,)
    ).fetchone()
    return row is not None


def load_document_by_hash(db, content_hash: str) -> ExtractionResult | None:
    """Loads an ExtractionResult from the database using its content hash."""
    row = db._conn.execute(
        "SELECT id FROM documents WHERE content_hash = ? LIMIT 1", (content_hash,)
    ).fetchone()
    if not row:
        return None
    return load_document(db, row["id"])


def save_document(db, result: ExtractionResult) -> None:
    """Persists an ExtractionResult to the database."""
    doc_id = result.doc_id
    content_hash = result.content_hash

    with db._conn:
        db._conn.execute(
            "DELETE FROM text_chunks_vec WHERE chunk_id IN (SELECT id FROM text_chunks WHERE document_id = ?)",
            (doc_id,),
        )

        db._conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (id, source_path, filename, markdown, page_count, content_hash, run_id, schema, paper_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                result.source_path,
                Path(result.source_path).name,
                result.markdown,
                result.page_count,
                content_hash,
                result.run_id,
                result.schema,
                json.dumps(asdict(result.paper_metadata)) if result.paper_metadata else None,
            )
        )

        for img in result.images:
            db._conn.execute(
                """
                INSERT OR REPLACE INTO images
                (id, document_id, mime_type, base64_data, storage_path, page_number, caption, contextualized_text, bbox, source_filename, confidence, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    img.id,
                    doc_id,
                    img.mime_type,
                    img.base64_data,
                    img.storage_path,
                    img.page,
                    img.caption,
                    img.contextualized_text,
                    json.dumps(img.bbox) if img.bbox else None,
                    img.source_filename,
                    img.confidence,
                    img.category,
                )
            )

        for tbl in result.tables:
            db._conn.execute(
                """
                INSERT OR REPLACE INTO tables
                (id, document_id, content, page_number, caption, contextualized_text, col_count, row_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tbl.id, doc_id, tbl.content, tbl.page, tbl.title, tbl.contextualized_text, tbl.col_count, tbl.row_count)
            )

        for eq in result.equations:
            db._conn.execute(
                """
                INSERT OR REPLACE INTO equations
                (id, document_id, text, display_mode, contextualized_text, page_number, caption)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eq.id,
                    doc_id,
                    eq.latex_or_text,
                    eq.display_mode,
                    eq.contextualized_text,
                    eq.page,
                    eq.caption,
                )
            )

        for chunk in result.source_chunks:
            db._conn.execute(
                """
                INSERT OR REPLACE INTO text_chunks
                (id, document_id, text, meta_data, contextualized_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    doc_id,
                    chunk.text,
                    json.dumps(chunk.meta_data),
                    chunk.contextualized_text
                )
            )

        embs = result.chunk_embeddings
        sources = result.chunk_embedding_sources
        if embs is not None and sources is not None:
            dim = db.config.vec_dimensions
            db._logger.log(
                f"[ResearchDatabase] Writing embeddings doc_id={doc_id} "
                f"chunks={len(result.source_chunks)} embs={len(embs)} dim_expected={dim}"
            )
            inserted = 0
            skipped_dim = 0
            for chunk, emb, src in zip(result.source_chunks, embs, sources):
                if len(emb) != dim:
                    skipped_dim += 1
                    continue
                blob = struct.pack(f"{dim}f", *emb)
                db._conn.execute(
                    """
                    INSERT OR REPLACE INTO text_chunks_vec (chunk_id, embedding, source)
                    VALUES (?, ?, ?)
                    """,
                    (chunk.id, blob, src),
                )
                inserted += 1
            db._logger.log(
                f"[ResearchDatabase] Embeddings write complete doc_id={doc_id} "
                f"inserted={inserted} skipped_dim={skipped_dim}"
            )
        else:
            db._logger.log(
                f"[ResearchDatabase] No embeddings to write doc_id={doc_id} "
                f"chunk_embeddings={'set' if result.chunk_embeddings is not None else 'None'} "
                f"chunk_embedding_sources={'set' if result.chunk_embedding_sources is not None else 'None'}"
            )


def load_document(db, doc_id: str) -> ExtractionResult | None:
    """Loads an ExtractionResult from the database."""
    doc_row = db._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not doc_row:
        return None

    img_rows = db._conn.execute("SELECT * FROM images WHERE document_id = ?", (doc_id,)).fetchall()
    images = [
        ExtractedImage(
            id=row["id"],
            mime_type=row["mime_type"],
            base64_data=row["base64_data"] or "",
            page=row["page_number"],
            caption=row["caption"] or "",
            storage_path=row["storage_path"],
            contextualized_text=row["contextualized_text"],
            bbox=json.loads(row["bbox"]) if row["bbox"] else None,
            source_filename=row["source_filename"],
            confidence=row["confidence"],
            category=row["category"],
        ) for row in img_rows
    ]

    tbl_rows = db._conn.execute("SELECT * FROM tables WHERE document_id = ?", (doc_id,)).fetchall()
    tables = [
        ExtractedTable(
            id=row["id"],
            content=row["content"],
            page=row["page_number"],
            title=row["caption"] or "",
            contextualized_text=row["contextualized_text"],
            col_count=row["col_count"],
            row_count=row["row_count"]
        ) for row in tbl_rows
    ]

    eq_rows = db._conn.execute("SELECT * FROM equations WHERE document_id = ?", (doc_id,)).fetchall()
    equations = [
        ExtractedEquation(
            id=row["id"],
            latex_or_text=row["text"],
            display_mode=row["display_mode"] or "block",
            page=row["page_number"],
            caption=row["caption"] or "",
            contextualized_text=row["contextualized_text"]
        ) for row in eq_rows
    ]

    paper_metadata = None
    if doc_row["paper_metadata"]:
        paper_metadata = PaperMetadata(**json.loads(doc_row["paper_metadata"]))

    chunk_rows = _load_ordered_chunk_rows(db, doc_id)
    source_chunks = [
        ExtractedChunk(
            id=row["id"],
            text=row["text"],
            contextualized_text=row["contextualized_text"],
            meta_data=json.loads(row["meta_data"])
        ) for row in chunk_rows
    ]

    return ExtractionResult(
        doc_id=doc_id,
        source_path=doc_row["source_path"],
        markdown=doc_row["markdown"],
        source_chunks=source_chunks,
        images=images,
        tables=tables,
        equations=equations,
        page_count=doc_row["page_count"],
        run_id=doc_row["run_id"],
        schema=doc_row["schema"],
        paper_metadata=paper_metadata,
    )


def get_image(db, image_id: str) -> ExtractedImage | None:
    """Loads a specific image from the database by its ID."""
    row = db._conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        return None
    return ExtractedImage(
        id=row["id"],
        mime_type=row["mime_type"],
        base64_data=row["base64_data"] or "",
        page=row["page_number"],
        caption=row["caption"] or "",
        storage_path=row["storage_path"],
        contextualized_text=row["contextualized_text"],
        bbox=json.loads(row["bbox"]) if row["bbox"] else None,
        source_filename=row["source_filename"],
        confidence=row["confidence"],
        category=row["category"],
    )


def get_table(db, table_id: str) -> ExtractedTable | None:
    """Loads a specific table from the database by its ID."""
    row = db._conn.execute("SELECT * FROM tables WHERE id = ?", (table_id,)).fetchone()
    if not row:
        return None
    return ExtractedTable(
        id=row["id"],
        content=row["content"],
        page=row["page_number"],
        title=row["caption"] or "",
        contextualized_text=row["contextualized_text"],
        col_count=row["col_count"],
        row_count=row["row_count"]
    )


def get_chunks_for_dispatch(db, doc_id: str) -> list[dict]:
    """
    Return all text chunks for a document in stable document order, as lightweight
    dicts with keys: id, text, contextualized_text, meta_data (parsed dict).
    """
    rows = _load_ordered_chunk_rows(db, doc_id)

    return [
        {
            "id": row["id"],
            "text": row["text"],
            "contextualized_text": row["contextualized_text"],
            "meta_data": json.loads(row["meta_data"]) if row["meta_data"] else {},
        }
        for row in rows
    ]
