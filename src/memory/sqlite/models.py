from __future__ import annotations

from dataclasses import dataclass


TABLE_NAMES = ["images", "tables", "equations", "text_chunks", "text_chunks_vec", "documents"]


@dataclass
class DocumentRecord:
    """
    Metadata for a processed document.
    Maps to the `documents` table.
    """
    id: str  # UUID4
    source_path: str  # absolute resolved path to the original file
    filename: str  # basename of the file
    markdown: str  # full document exported via DoclingDocument.export_to_markdown()
    page_count: int
    content_hash: str  # sha256 of the raw file bytes
    created_at: str  # ISO8601 timestamp
    docling_schema_version: str | None = None  # from DoclingDocument.version


@dataclass
class ImageRecord:
    """
    Metadata for an image extracted from a document.
    Maps to the `images` table.
    """
    id: str  # UUID4
    document_id: str  # FK to documents
    mime_type: str  # e.g. "image/png"
    base64_data: str  # base64-encoded image bytes
    page_number: int | None = None
    caption: str | None = None
    bbox_json: str | None = None  # JSON-serialized bounding box dict {l, t, r, b}
    annotation_json: str | None = None  # JSON-serialized list of annotation dicts


@dataclass
class TableRecord:
    """
    Metadata for a table extracted from a document.
    Maps to the `tables` table.
    """
    id: str  # UUID4
    document_id: str  # FK to documents
    html_content: str  # from TableItem.export_to_html()
    page_number: int | None = None
    caption: str | None = None
    bbox_json: str | None = None
    col_count: int | None = None
    row_count: int | None = None


@dataclass
class EquationRecord:
    """
    Metadata for an equation extracted from a document.
    Maps to the `equations` table.
    """
    id: str  # UUID4
    document_id: str  # FK to documents
    text: str  # raw formula text
    orig: str | None = None  # original representation
    page_number: int | None = None
    bbox_json: str | None = None
    caption: str | None = None


@dataclass
class TextChunkRecord:
    """
    Metadata for a text chunk produced by HybridChunker.
    Maps to the `text_chunks` table.
    """
    id: str  # UUID4
    document_id: str  # FK to documents
    text: str  # HybridChunk.text
    headings_json: str  # JSON list of headings
    captions_json: str  # JSON list of captions
    page_numbers_json: str  # JSON list of unique page numbers
    doc_item_labels_json: str  # JSON list of unique DocItemLabel string values
    chunk_index: int  # 0-based position within document
    content_hash: str  # sha256 of the `.text` field
    embedding_model: str | None = None
    embedded_at: str | None = None
