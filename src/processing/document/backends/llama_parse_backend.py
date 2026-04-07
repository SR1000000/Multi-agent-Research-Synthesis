from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from llama_cloud import LlamaCloud

from src.memory.objectstore import LocalObjectStore, ObjectStoreProvider
from src.processing.chunker import TextChunkerProvider

from ..backend_base import OCRBackend
from ..schema import (
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionResult,
    PaperMetadata,
)


# ---------------------------------------------------------------------------
# LlamaParse v2 call configuration
# ---------------------------------------------------------------------------
# Tier choice for research papers:
#   "agentic_plus" → best table accuracy, equation handling, figure captions.
#   Switch to "agentic" to halve cost with only minor quality loss.
#
# expand values (ParsingGetResponse fields to populate):
#   "markdown_full"           → single markdown string for the whole document
#   "items"                   → structured per-page items (tables, images, headings, text)
#   "metadata"                → per-page confidence + page_number list
#   "images_content_metadata" → image presigned_url + caption + category + bbox
#
# output_options.images_to_save:
#   "embedded" → only actual figure/image objects (NOT full-page screenshots)
#   DO NOT include "screenshot" or "layout" — those inflate cost and storage.
#
# agentic_options.custom_prompt instructs the LLM on equation and caption format.
# ---------------------------------------------------------------------------

_TIER = "agentic_plus"
_VERSION = "latest"

_PARSE_KWARGS: dict[str, Any] = {
    "tier": _TIER,
    "version": _VERSION,
    "agentic_options": {
        "custom_prompt": (
            "This is a scientific research paper. "
            "Render all mathematical equations in LaTeX notation. "
            "Inline equations: $equation$. Block/display equations: $$equation$$. "
            "For every figure or image, include its full caption verbatim. "
            "For every table, include its full caption or title above the table. "
            "Preserve section headings exactly as they appear."
        ),
    },
    "output_options": {
        # Only extract actual embedded figures — not page screenshots or layout crops.
        "images_to_save": ["embedded"],
        # We want HTML tables in the items response. HTML is the richest format
        # (preserves merged cells). markdown and csv are also present as fallback.
        "markdown": {
            "tables": {
                # False = emit HTML tables (not markdown pipe tables) in markdown_full.
                "output_tables_as_markdown": False,
                "merge_continued_tables": True,
            },
            "inline_images": False,
            "annotate_links": True,
        },
        "extract_printed_page_number": True,
    },
    "processing_options": {
        # Catches tables that are not clearly bordered.
        "aggressive_table_extraction": True,
        "ocr_parameters": {"languages": ["en"]},
        "ignore": {
            "ignore_diagonal_text": False,
            "ignore_hidden_text": True,
        },
    },
    # Request all the data we need in one call — avoids a second .get() round-trip.
    "expand": [
        "markdown_full",       # full document markdown
        "items",               # structured page items (tables, images, headings, text)
        "metadata",            # page list with page_number
        "images_content_metadata",  # presigned_url per extracted image
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute or dict key access with a default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _nested(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for k in keys:
        cur = _attr(cur, k)
        if cur is None:
            return default
    return cur


# ---------------------------------------------------------------------------
# Markdown parsing utilities (research paper structure)
# ---------------------------------------------------------------------------

_RE_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_RE_ABSTRACT = re.compile(
    r"(?:^#{1,3}\s*Abstract\s*\n)(.*?)(?=^#{1,3}\s|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_RE_SECTION = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_RE_REFERENCES = re.compile(
    r"^#{1,3}\s*References?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Matches bracketed refs like "[1] Author..." or "1. Author..."
_RE_CITATION_LINE = re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s+.+", re.MULTILINE)
# Author lines often appear between title and abstract (second line of first heading block)
_RE_AUTHORS = re.compile(
    r"^#{1,3}\s*(?:Authors?|By)\s*\n(.*?)(?=^#{1,3}\s|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_RE_DOI = re.compile(r"\bdoi[:\s]+([^\s,;\)]+)", re.IGNORECASE)
_RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_RE_BLOCK_EQ = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_RE_INLINE_EQ = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def _parse_paper_metadata(markdown: str) -> PaperMetadata:
    """
    Extract structured research paper metadata from markdown_full using regex.
    No LLM call — fast and free. Best-effort for standard paper layouts.
    """
    meta = PaperMetadata()

    # Title: first H1
    title_match = _RE_TITLE.search(markdown)
    if title_match:
        meta.title = title_match.group(1).strip()

    # Abstract
    abstract_match = _RE_ABSTRACT.search(markdown)
    if abstract_match:
        meta.abstract = abstract_match.group(1).strip()

    # Authors: look for an "Authors" section, or the block of text right after
    # the title and before the abstract (heuristic for standard paper format).
    authors_match = _RE_AUTHORS.search(markdown)
    if authors_match:
        raw = authors_match.group(1).strip()
        # Split on commas, semicolons, newlines, or " and "
        parts = re.split(r"[,;\n]| and ", raw)
        meta.authors = [p.strip() for p in parts if p.strip()]
    else:
        # Heuristic: lines between title and abstract that look like names
        if title_match and abstract_match:
            between = markdown[title_match.end():abstract_match.start()]
            candidates = [
                ln.strip().lstrip("#").strip()
                for ln in between.splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            if candidates:
                meta.authors = candidates

    # Sections: all headings except abstract/references, collect body text
    sections: dict[str, str] = {}
    heading_positions = [(m.start(), m.group(2).strip()) for m in _RE_SECTION.finditer(markdown)]
    for i, (pos, heading) in enumerate(heading_positions):
        end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(markdown)
        body = markdown[pos:end]
        # Strip the heading line itself
        body = body[body.index("\n") + 1:].strip() if "\n" in body else ""
        norm = heading.lower()
        if any(skip in norm for skip in ("abstract", "reference", "bibliography")):
            continue
        sections[heading] = body
    meta.sections = sections

    # Citations: everything after the References heading
    refs_match = _RE_REFERENCES.search(markdown)
    if refs_match:
        refs_block = markdown[refs_match.end():]
        meta.citations = [
            m.group(0).strip() for m in _RE_CITATION_LINE.finditer(refs_block)
        ]

    # DOI
    doi_match = _RE_DOI.search(markdown)
    if doi_match:
        meta.doi = doi_match.group(1).strip().rstrip(".")

    # Year: first 4-digit year in the markdown (title/author block tends to have it)
    year_match = _RE_YEAR.search(markdown[:2000])
    if year_match:
        meta.year = year_match.group(0)

    return meta


def _extract_equations_from_markdown(doc_id: str, markdown: str) -> list[ExtractedEquation]:
    """
    Scan markdown_full for LaTeX equation blocks and inline equations.

    LlamaParse does not emit a dedicated equation item type (as of v2).
    Equations appear in the markdown rendered by the agentic model. With the
    custom_prompt asking for LaTeX, block equations come out as $$...$$
    and inline as $...$.

    We deduplicate by expression content.
    """
    equations: list[ExtractedEquation] = []
    seen: set[str] = set()
    counter = 1

    # Block equations first (higher value)
    for m in _RE_BLOCK_EQ.finditer(markdown):
        expr = m.group(1).strip()
        if not expr or expr in seen:
            continue
        seen.add(expr)
        equations.append(ExtractedEquation(
            id=f"{doc_id}_eq_{counter:03d}",
            latex_or_text=expr,
            display_mode="block",
        ))
        counter += 1

    # Inline equations
    for m in _RE_INLINE_EQ.finditer(markdown):
        expr = m.group(1).strip()
        if not expr or expr in seen or len(expr) < 2:
            continue
        seen.add(expr)
        equations.append(ExtractedEquation(
            id=f"{doc_id}_eq_{counter:03d}",
            latex_or_text=expr,
            display_mode="inline",
        ))
        counter += 1

    return equations


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class LlamaParseBackend(OCRBackend):
    """OCR backend powered by LlamaParse v2 (llama-cloud SDK)."""

    def __init__(
        self,
        object_store: ObjectStoreProvider | None = None,
        text_chunker: TextChunkerProvider | None = None,
    ) -> None:
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        self._client = LlamaCloud(api_key=api_key)
        self._object_store = object_store or LocalObjectStore()
        self._text_chunker = text_chunker

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        source = Path(source_pdf_path)
        doc_id = source.stem

        # Single blocking call — parse + poll + expand all in one.
        parse_result = self._client.parsing.parse(
            upload_file=str(source),
            **_PARSE_KWARGS,
        )

        markdown_full: str = _attr(parse_result, "markdown_full") or ""
        metadata_block = _attr(parse_result, "metadata")
        pages = _attr(metadata_block, "pages") or []
        page_count = len(pages)

        chunks = self._extract_chunks(doc_id, markdown_full)
        tables = self._extract_tables(doc_id, parse_result)
        images = self._extract_images(doc_id, parse_result)
        equations = _extract_equations_from_markdown(doc_id, markdown_full)
        paper_metadata = _parse_paper_metadata(markdown_full) if markdown_full else None

        return ExtractionResult(
            doc_id=doc_id,
            source_path=str(source),
            markdown=markdown_full,
            source_chunks=chunks,
            images=images,
            tables=tables,
            equations=equations,
            page_count=page_count,
            paper_metadata=paper_metadata,
            schema=f"llamaparse/{_TIER}/{_VERSION}",
        )

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def _extract_chunks(self, doc_id: str, markdown_full: str) -> list[ExtractedChunk]:
        text = markdown_full.strip()
        if not text:
            return []

        if self._text_chunker is None:
            return [ExtractedChunk(
                id=f"{doc_id}_chunk_0000",
                text=text,
                meta_data={"chunk_index": 0, "splitter": "none"},
            )]

        # Use MarkdownSplitter when available for structure-aware splitting.
        # Falls back to plain text splitter.
        try:
            raw_chunks = self._text_chunker.chunk_markdown(text)
            splitter_name = type(self._text_chunker).__name__ + ".markdown"
        except AttributeError:
            raw_chunks = self._text_chunker.chunk_text(text)
            splitter_name = type(self._text_chunker).__name__

        result: list[ExtractedChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            result.append(ExtractedChunk(
                id=f"{doc_id}_chunk_{idx:04d}",
                text=chunk_text,
                meta_data={"chunk_index": idx, "splitter": splitter_name},
            ))
        return result

    # ------------------------------------------------------------------
    # Tables — sourced from structured items (most reliable)
    # ------------------------------------------------------------------

    def _extract_tables(self, doc_id: str, parse_result: Any) -> list[ExtractedTable]:
        """
        Walk items.pages[*].items and collect TableItem entries.

        TableItem fields (all required by schema):
            html  str  — HTML <table> string  ← preferred
            md    str  — markdown pipe table  ← fallback
            csv   str  — CSV string           ← last resort
            rows  list[list]                  ← used for row/col count

        Optional:
            type              str   = "table"
            merged_from_pages list[int]
            merged_into_page  int
            bbox              list[BBox]
        """
        tables: list[ExtractedTable] = []
        counter = 1

        items_block = _attr(parse_result, "items")
        item_pages = _attr(items_block, "pages") or []

        for page in item_pages:
            page_number: int | None = _attr(page, "page_number")
            page_items = _attr(page, "items") or []
            last_heading: str = ""

            for item in page_items:
                item_type = str(_attr(item, "type", "")).lower()

                # Track the nearest heading so we can use it as a table title.
                if item_type == "heading":
                    last_heading = str(_attr(item, "value", "") or "").strip()
                    continue

                if item_type != "table":
                    continue

                html: str = str(_attr(item, "html") or "").strip()
                md: str = str(_attr(item, "md") or "").strip()
                csv: str = str(_attr(item, "csv") or "").strip()
                content = html or md or csv
                if not content:
                    continue

                rows = _attr(item, "rows") or []
                row_count = len(rows) if rows else None
                col_count = len(rows[0]) if (rows and isinstance(rows[0], list)) else None

                # Title: use the heading above the table, or a generic fallback.
                title = last_heading or f"Table {counter}"

                tables.append(ExtractedTable(
                    id=f"{doc_id}_tbl_{counter:03d}",
                    content=content,
                    page=page_number,
                    title=title,
                    caption="",       # LlamaParse v2 items don't carry a caption field
                    col_count=col_count,
                    row_count=row_count,
                ))
                counter += 1

        return tables

    # ------------------------------------------------------------------
    # Images — sourced from images_content_metadata (has presigned_url)
    # ------------------------------------------------------------------

    def _extract_images(self, doc_id: str, parse_result: Any) -> list[ExtractedImage]:
        """
        Download embedded figures from presigned URLs in images_content_metadata.

        ImagesContentMetadataImage fields:
            filename      str            ← for extension detection
            index         int            ← 0-based order
            presigned_url str | None     ← download URL (expires ~15 min)
            content_type  str | None     ← MIME type e.g. "image/png"
            category      str | None     ← "embedded" | "screenshot" | "layout"
            bbox          BBox | None    ← position on page
            size_bytes    int | None

        Caption is NOT on ImagesContentMetadataImage — it lives on the
        corresponding ImageItem in items.pages[*].items. We build a lookup
        map from index → caption using the items tree first.
        """
        # --- Step 1: build index → caption map from structured items ---
        caption_by_index: dict[int, str] = {}
        items_block = _attr(parse_result, "items")
        item_pages = _attr(items_block, "pages") or []
        img_item_counter = 0  # items are 0-indexed globally in order
        for page in item_pages:
            for item in (_attr(page, "items") or []):
                if str(_attr(item, "type", "")).lower() == "image":
                    caption = str(_attr(item, "caption") or "").strip()
                    if caption:
                        caption_by_index[img_item_counter] = caption
                    img_item_counter += 1

        # --- Step 2: download and store images ---
        images_meta = _attr(parse_result, "images_content_metadata")
        image_entries = _attr(images_meta, "images") or []

        extracted: list[ExtractedImage] = []
        for entry in image_entries:
            category = str(_attr(entry, "category") or "").lower()
            # Only keep actual embedded figures — skip screenshots and layout crops.
            if category and category != "embedded":
                continue

            presigned_url: str | None = _attr(entry, "presigned_url")
            if not presigned_url:
                continue

            index: int = _attr(entry, "index", 0)
            filename: str = str(_attr(entry, "filename") or f"img_{index}.bin")
            mime_type: str = str(_attr(entry, "content_type") or "application/octet-stream")
            caption: str = caption_by_index.get(index, "")

            try:
                image_bytes = _download_bytes(presigned_url)
            except Exception as exc:
                print(f"[LlamaParseBackend] Failed to download image {filename}: {exc}")
                continue

            img_id = f"{doc_id}_img_{index + 1:03d}"
            ext = _extension(filename, mime_type)
            storage_key = f"{doc_id}/images/{img_id}.{ext}"
            self._object_store.write(storage_key, image_bytes)

            extracted.append(ExtractedImage(
                id=img_id,
                mime_type=mime_type,
                base64_data=base64.b64encode(image_bytes).decode("utf-8"),
                page=None,  # page not present on ImagesContentMetadataImage
                caption=caption,
                local_path=storage_key,
            ))

        return extracted


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _download_bytes(url: str, timeout: int = 30) -> bytes:
    with urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _extension(filename: str, mime_type: str) -> str:
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].strip().lower()
        if ext and len(ext) <= 5:
            return ext
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/tiff": "tiff",
    }.get(mime_type, "bin")