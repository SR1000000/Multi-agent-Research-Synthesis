from __future__ import annotations

import json
import re
import struct
from dataclasses import asdict
from pathlib import Path

from src.memory.research.schema import ImageMetadata
from src.processing.document.schema import (
    ExtractionResult,
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    PaperMetadata,
)

_IMG_TOKEN_RE = re.compile(r"\[\[img:([^\]]+)\]\]")


def _parse_img_refs_from_text(text: str) -> list[str]:
    return _IMG_TOKEN_RE.findall(text or "")


def _image_aspect_ratio(bbox: dict | None) -> str:
    if not bbox:
        return "landscape"
    w = bbox.get("width")
    if w is None:
        w = bbox.get("x2", 0) - bbox.get("x1", bbox.get("x", 0))
    h = bbox.get("height")
    if h is None:
        h = bbox.get("y2", 0) - bbox.get("y1", bbox.get("y", 0))
    try:
        fw = float(w)
        fh = float(h)
    except (TypeError, ValueError):
        return "landscape"
    if fw <= 0 or fh <= 0:
        return "landscape"
    ratio = fw / fh
    if ratio > 1.2:
        return "landscape"
    if ratio < 0.83:
        return "portrait"
    return "square"


def _extracted_image_from_row(row) -> ExtractedImage:
    """Build ExtractedImage from sqlite Row; tolerate older DB rows missing new columns."""
    keys = row.keys()

    def _col(name: str, default):
        return row[name] if name in keys else default

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
        vlm_caption=_col("vlm_caption", "") or "",
        mermaid=_col("mermaid", None),
        figure_group_id=_col("figure_group_id", None),
        figure_label=_col("figure_label", None),
        figure_number=_col("figure_number", None),
        panel_index=_col("panel_index", None),
        panel_role=_col("panel_role", None),
        identity_signal=_col("identity_signal", None),
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



def build_search_text(contextualized_text: str | None, raw_value: str | None) -> str:
    """Build search text by concatenating contextualized and raw content with double newline separator."""
    ctx = (contextualized_text or "").strip()
    raw = (raw_value or "").strip()
    if ctx and raw:
        return f"{ctx}\n\n{raw}"
    return ctx or raw


def insert_fts_row(db, item_id: str, doc_id: str, kind: str, search_text: str) -> None:
    """Insert a new row into the FTS index and mapping table."""
    # Insert into mapping table to get an auto-assigned rowid
    cursor = db._conn.execute(
        "INSERT INTO fts_rowid_map(item_id, document_id, kind) VALUES (?, ?, ?)",
        (item_id, doc_id, kind),
    )
    rowid = cursor.lastrowid
    # Insert into FTS index using the assigned rowid
    db._conn.execute(
        """
        INSERT INTO artifact_search_fts(rowid, item_id, document_id, kind, search_text)
        VALUES (?, ?, ?, ?, ?)
        """,
        (rowid, item_id, doc_id, kind, search_text),
    )


def delete_fts_rows_for_document(db, doc_id: str) -> None:
    """Delete all FTS rows and mappings for a given document."""
    db._conn.execute(
        "DELETE FROM artifact_search_fts WHERE rowid IN (SELECT rowid FROM fts_rowid_map WHERE document_id = ?)",
        (doc_id,),
    )
    db._conn.execute(
        "DELETE FROM fts_rowid_map WHERE document_id = ?",
        (doc_id,),
    )

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
        # Delete existing FTS rows for this document first
        delete_fts_rows_for_document(db, doc_id)

        # Clear existing artifact data
        db._conn.execute(
            "DELETE FROM text_chunks_vec WHERE chunk_id IN (SELECT id FROM text_chunks WHERE document_id = ?)",
            (doc_id,),
        )
        db._conn.execute("DELETE FROM text_chunks WHERE document_id = ?", (doc_id,))
        db._conn.execute("DELETE FROM images WHERE document_id = ?", (doc_id,))
        db._conn.execute("DELETE FROM tables WHERE document_id = ?", (doc_id,))
        db._conn.execute("DELETE FROM equations WHERE document_id = ?", (doc_id,))

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
                (id, document_id, mime_type, base64_data, storage_path, page_number, caption, contextualized_text, bbox, source_filename, confidence, category,
                 vlm_caption, mermaid, figure_group_id, figure_label, figure_number, panel_index, panel_role, identity_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    img.vlm_caption or "",
                    img.mermaid,
                    img.figure_group_id,
                    img.figure_label,
                    img.figure_number,
                    img.panel_index,
                    img.panel_role,
                    img.identity_signal,
                )
            )
            # Insert FTS row for this image
            image_raw_value = img.caption or img.storage_path
            search_text = build_search_text(img.contextualized_text, image_raw_value)
            insert_fts_row(db, img.id, doc_id, 'image', search_text)

        for tbl in result.tables:
            db._conn.execute(
                """
                INSERT OR REPLACE INTO tables
                (id, document_id, content, page_number, caption, contextualized_text, col_count, row_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tbl.id, doc_id, tbl.content, tbl.page, tbl.title, tbl.contextualized_text, tbl.col_count, tbl.row_count)
            )
            # Insert FTS row for this table
            search_text = build_search_text(tbl.contextualized_text, tbl.content)
            insert_fts_row(db, tbl.id, doc_id, 'table', search_text)

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
            # Insert FTS row for this equation
            search_text = build_search_text(eq.contextualized_text, eq.latex_or_text)
            insert_fts_row(db, eq.id, doc_id, 'equation', search_text)

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
            # Insert FTS row for this text chunk
            search_text = build_search_text(chunk.contextualized_text, chunk.text)
            insert_fts_row(db, chunk.id, doc_id, 'chunk', search_text)

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
    images = [_extracted_image_from_row(row) for row in img_rows]

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


def get_images_for_chunks(db, chunk_ids: list[str]) -> list[ImageMetadata]:
    """
    Return ImageMetadata for images explicitly referenced by [[img:ID]] tokens
    in chunk text. Only rows with embeddable data (storage_path or base64) are returned.
    Order follows first-seen token order when walking chunk_ids in caller order.
    """
    if not chunk_ids:
        return []

    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = db._conn.execute(
        f"SELECT id, text, contextualized_text FROM text_chunks WHERE id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    rows_by_id = {row["id"]: row for row in rows}

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for cid in chunk_ids:
        row = rows_by_id.get(cid)
        if row is None:
            continue
        for token_source in (row["text"] or "", row["contextualized_text"] or ""):
            for img_id in _parse_img_refs_from_text(token_source):
                if img_id not in seen:
                    seen.add(img_id)
                    ordered_ids.append(img_id)

    if not ordered_ids:
        return []

    img_placeholders = ",".join(["?"] * len(ordered_ids))
    img_rows = db.connection.execute(
        f"""
        SELECT id, caption, vlm_caption, bbox
        FROM images
        WHERE id IN ({img_placeholders})
          AND (
            storage_path IS NOT NULL
            OR (base64_data IS NOT NULL AND base64_data != '')
          )
        """,
        ordered_ids,
    ).fetchall()
    by_img_id = {r["id"]: r for r in img_rows}

    out: list[ImageMetadata] = []
    for img_id in ordered_ids:
        row = by_img_id.get(img_id)
        if row is None:
            db._logger.log(
                f"[get_images_for_chunks] Skipping image id={img_id!r}: not in DB or not embeddable",
                level="warning",
            )
            continue
        bbox = json.loads(row["bbox"]) if row["bbox"] else None
        keys = row.keys()
        vlm = (row["vlm_caption"] or "") if "vlm_caption" in keys else ""
        out.append(
            ImageMetadata(
                id=row["id"],
                caption=row["caption"] or "",
                vlm_caption=vlm,
                aspect_ratio=_image_aspect_ratio(bbox),
                bbox=bbox,
            )
        )
    return out


def get_image(db, image_id: str) -> ExtractedImage | None:
    """Loads a specific image from the database by its ID."""
    row = db._conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        return None
    return _extracted_image_from_row(row)


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


def get_equation(db, equation_id: str) -> ExtractedEquation | None:
    """Loads a specific equation from the database by its ID."""
    row = db._conn.execute("SELECT * FROM equations WHERE id = ?", (equation_id,)).fetchone()
    if not row:
        return None
    return ExtractedEquation(
        id=row["id"],
        latex_or_text=row["text"],
        display_mode=row["display_mode"] or "block",
        page=row["page_number"],
        caption=row["caption"] or "",
        contextualized_text=row["contextualized_text"]
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
