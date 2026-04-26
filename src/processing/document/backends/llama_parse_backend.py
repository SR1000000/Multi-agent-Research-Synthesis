from __future__ import annotations

import base64
import os
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from llama_cloud import LlamaCloud

from src.logging.logger import AgentLogger
from src.memory.objectstore import LocalObjectStore, ObjectStoreProvider
from src.processing.chunker import TextChunkerProvider

from ..backend_base import OCRBackend
from ..schema import (
    ExtractedChunk,
    ExtractedEquation,
    ExtractedTable,
    ExtractionResult,
    PaperMetadata,
)
from .llama_parse_figures import (
    assign_entries_to_anchors,
    build_extracted_images,
    collect_figure_anchors,
    iter_layout_entries,
    rescue_orphan_figure_entries,
    rewrite_markdown_images_and_tables,
)

_TIER = "agentic"  # agentic_plus is the highest tier of LlamaParse, agentic is 4x cheaper (and roughly 4-6x faster)
# Prefix embedded in every artifact ID produced by this backend.
# Other backends should define their own prefix (e.g. "dl" for Docling)
# so IDs remain self-describing and collision-free across providers.
_LP_ID_PREFIX = "lp"
_VERSION = "latest"
_PARSE_CREATE_MAX_ATTEMPTS = 3
_PARSE_WAIT_MAX_ATTEMPTS = 3
_PARSE_WAIT_BACKOFF_SECONDS = 2
_PARSE_WAIT_HEARTBEAT_SECONDS = 15.0
_IMAGE_DOWNLOAD_MAX_ATTEMPTS = 3
_IMAGE_DOWNLOAD_BACKOFF_SECONDS = 2

_PARSE_KWARGS: dict[str, Any] = {
    "tier": _TIER,
    "version": _VERSION,
    "agentic_options": {
        "custom_prompt": (
            "This is a scientific research paper. "
            "Render all mathematical equations in LaTeX notation. "
            "Inline equations: $equation$. Block/display equations: $$equation$$. "
            "For every figure or image, include its full caption verbatim. "
            "CRITICAL: For every figure or diagram, you MUST include the original "
            "Markdown image reference tag (e.g., ![caption](image)) immediately "
            "before any Mermaid or HTML transcription you provide. Never omit the image tag."
            "Preserve section headings exactly as they appear."
        ),
    },
    "output_options": {
        # "layout"   = LlamaParse's bbox-cropped layout detections — covers every page,
        #              including those where figures are not embedded as PDF objects.
        # "embedded" = actual embedded PDF objects — only present on a subset of pages
        #              and misses figures that are part of the rendered page layout.
        "images_to_save": ["layout"],
        # We want HTML tables in the items response. HTML is the richest format
        # (preserves merged cells). markdown and csv are also present as fallback.
        "markdown": {
            "tables": {
                # False = emit HTML tables (not markdown pipe tables) in markdown_full.
                "output_tables_as_markdown": False,
                "merge_continued_tables": True,
            },
            #"inline_images": True,
            "annotate_links": True,
        },
        "extract_printed_page_number": True,
    },
    "processing_options": {
        "cost_optimizer": {"enable": True},
        # Catches tables that are not clearly bordered.
        "aggressive_table_extraction": True,
        "ocr_parameters": {"languages": ["en"]},
        "ignore": {
            "ignore_diagonal_text": False,
            "ignore_hidden_text": True,
        },
    },
}
_PARSE_EXPAND: list[str] = [
    "markdown_full",       # full document markdown
    "items",               # structured page items (tables, images, headings, text)
    "metadata",            # page list with page_number
    "images_content_metadata",  # presigned_url per extracted image
]

def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute or dict key access with a default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _serialize_parse_payload(parse_result: Any) -> Any:
    if hasattr(parse_result, "model_dump"):
        return parse_result.model_dump()
    if hasattr(parse_result, "dict"):
        return parse_result.dict()
    if isinstance(parse_result, dict):
        return parse_result
    return {"repr": repr(parse_result)}


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

# Extracts the page number from a LlamaParse layout filename, e.g. 'page_4_chart_1_v2.jpg' → 4
_PAGE_IN_FILENAME_RE = re.compile(r"\bpage_(\d+)_")


def _page_from_filename(filename: str) -> int | None:
    """Return the page number encoded in a LlamaParse layout filename, or None."""
    m = _PAGE_IN_FILENAME_RE.search(filename)
    return int(m.group(1)) if m else None


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
    Scan markdown_full for LaTeX block equations only.

    LlamaParse (agentic tier) does not emit a dedicated equation item type;
    equations appear inline in the rendered markdown. With the custom_prompt
    requesting LaTeX notation, block equations come out as $$...$$.

    Equations are deduplicated by expression content.
    IDs are prefixed with the LlamaParse provider prefix (_LP_ID_PREFIX).
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
            id=f"{doc_id}_{_LP_ID_PREFIX}_eq_{counter:03d}",
            latex_or_text=expr,
            display_mode="block",
        ))
        counter += 1

    return equations


def _is_garbage_table(html: str, rows: list) -> bool:
    """
    Detect tables that are actually misread figures/heatmaps.
    Signals:
    - Content is a figure caption rather than data
    - Rows contain mostly empty cells with occasional tokens
    """
    if not rows:
        return False
    col_count = len(rows[0]) if isinstance(rows[0], list) else 1
    row_count = len(rows)
    # Check if cells are overwhelmingly single words (token-soup pattern)
    flat_cells = [
        str(cell).strip()
        for row in rows if isinstance(row, list)
        for cell in row
    ]
    if not flat_cells:
        return False
    single_word_ratio = sum(1 for c in flat_cells if c and len(c.split()) <= 1) / len(flat_cells)
    if col_count >= 4 and single_word_ratio > 0.85 and row_count > 10:
        return True
    return False

class LlamaParseBackend(OCRBackend):
    """OCR backend powered by LlamaParse v2 (llama-cloud SDK)."""

    def __init__(
        self,
        object_store: ObjectStoreProvider | None = None,
        text_chunker: TextChunkerProvider | None = None,
        logger: AgentLogger | None = None,
    ) -> None:
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        self._client = LlamaCloud(api_key=api_key)
        self._object_store = object_store or LocalObjectStore()
        self._text_chunker = text_chunker
        self._logger = logger or AgentLogger()

    def _wait_parse_job_with_status(self, job_id: str) -> None:
        """Call LlamaParse wait_for_completion while emitting periodic status (SDK has no progress callback)."""
        stop = threading.Event()

        def _heartbeat() -> None:
            t0 = time.monotonic()
            while not stop.wait(_PARSE_WAIT_HEARTBEAT_SECONDS):
                elapsed = int(time.monotonic() - t0)
                msg = (
                    f"[LlamaParseBackend] Parse still in progress job_id={job_id} "
                    f"elapsed={elapsed}s (waiting on LlamaCloud)…"
                )
                self._logger.log(msg, level="info")

        hb = threading.Thread(target=_heartbeat, name=f"llamaparse-wait-{job_id[:8]}", daemon=True)
        hb.start()
        try:
            self._client.parsing.wait_for_completion(job_id)
        finally:
            stop.set()
            hb.join(timeout=2.0)

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        source = Path(source_pdf_path)
        doc_id = source.stem

        self._logger.log(f"[LlamaParseBackend] Starting extract path={source}", level="info")
        parse_result, run_id = self._parse_with_retry(str(source))
        dump_path = self._logger.dump_json_artifact(
            file_name=f"llamaparse_raw_{doc_id}.json",
            payload=_serialize_parse_payload(parse_result),
            run_id=run_id,
        )
        if dump_path:
            self._logger.log(f"[LlamaParseBackend] Wrote raw parse result to {dump_path}")
        else:
            self._logger.log("[LlamaParseBackend] Failed to write raw parse result", level="warning")

        markdown_full: str = _attr(parse_result, "markdown_full") or ""
        metadata_block = _attr(parse_result, "metadata")
        pages = _attr(metadata_block, "pages") or []
        page_count = len(pages)

        # ── Extract tables & figure-anchored images (geometry + captions) ────
        tables = self._extract_tables(doc_id, parse_result)
        equations = _extract_equations_from_markdown(doc_id, markdown_full)
        paper_metadata = _parse_paper_metadata(markdown_full) if markdown_full else None

        anchors = collect_figure_anchors(parse_result)
        layout_entries = iter_layout_entries(parse_result)
        assignments = assign_entries_to_anchors(anchors, layout_entries)
        anchors, assignments = rescue_orphan_figure_entries(
            parse_result, anchors, assignments, layout_entries
        )

        def _dl(url: str) -> bytes:
            return _download_bytes_with_retry(
                url,
                attempts=_IMAGE_DOWNLOAD_MAX_ATTEMPTS,
                backoff_seconds=_IMAGE_DOWNLOAD_BACKOFF_SECONDS,
            )

        def _store(key: str, data: bytes) -> str:
            return self._object_store.write(key, data)

        images, image_repl_map, loose_url_map, mermaid_fences = build_extracted_images(
            doc_id,
            anchors,
            assignments,
            layout_entries,
            download_bytes=_dl,
            write_to_store=_store,
            id_prefix=_LP_ID_PREFIX,
        )

        table_ids = [tbl.id for tbl in tables]
        rewritten_markdown = rewrite_markdown_images_and_tables(
            markdown_full,
            image_repl_map,
            loose_url_map,
            mermaid_fences,
            table_ids,
        )

        self._logger.log(
            f"[LlamaParseBackend] Figure anchors={len(anchors)} layout_entries={len(layout_entries)} "
            f"images_extracted={len(images)} image_replacements={len(image_repl_map)} "
            f"mermaid_strips={len(mermaid_fences)} table_tokens={len(table_ids)}"
        )

        # ── Chunk the rewritten markdown ───────────────────────────────────────
        chunks = self._extract_chunks(doc_id, rewritten_markdown)

        return ExtractionResult(
            doc_id=doc_id,
            source_path=str(source),
            markdown=rewritten_markdown,
            source_chunks=chunks,
            images=images,
            tables=tables,
            equations=equations,
            page_count=page_count,
            paper_metadata=paper_metadata,
            schema=f"llamaparse/{_TIER}/{_VERSION}",
            run_id=run_id,
        )

    def _parse_with_retry(self, source_path: str) -> tuple[Any, str | None]:
        last_exc: Exception | None = None
        for create_attempt in range(1, _PARSE_CREATE_MAX_ATTEMPTS + 1):
            job_id: str | None = None
            try:
                job = self._client.parsing.create(
                    upload_file=source_path,
                    **_PARSE_KWARGS,
                )
                job_id = str(_attr(job, "id") or "")
                if job_id:
                    print(f"[LlamaParseBackend] Parse job created id={job_id}")

                for wait_attempt in range(1, _PARSE_WAIT_MAX_ATTEMPTS + 1):
                    try:
                        if not job_id:
                            raise RuntimeError("LlamaParse create did not return job id.")
                        self._wait_parse_job_with_status(job_id)
                        done_msg = f"[LlamaParseBackend] Parse job finished job_id={job_id}, fetching result…"
                        self._logger.log(done_msg, level="info")
                        result = self._client.parsing.get(job_id, expand=_PARSE_EXPAND)
                        return result, job_id
                    except Exception as exc:
                        last_exc = exc
                        if wait_attempt >= _PARSE_WAIT_MAX_ATTEMPTS:
                            raise
                        sleep_s = _PARSE_WAIT_BACKOFF_SECONDS * wait_attempt
                        print(
                            f"[LlamaParseBackend] wait_for_completion failed "
                            f"(job_id={job_id}, attempt={wait_attempt}/{_PARSE_WAIT_MAX_ATTEMPTS}): {exc}. "
                            f"Retrying in {sleep_s}s..."
                        )
                        time.sleep(sleep_s)
            except Exception as exc:
                last_exc = exc
                if create_attempt >= _PARSE_CREATE_MAX_ATTEMPTS:
                    break
                sleep_s = _PARSE_WAIT_BACKOFF_SECONDS * create_attempt
                print(
                    f"[LlamaParseBackend] parse create/wait failed "
                    f"(attempt={create_attempt}/{_PARSE_CREATE_MAX_ATTEMPTS}): {exc}. "
                    f"Retrying in {sleep_s}s..."
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"LlamaParse parse failed after retries: {last_exc}") from last_exc

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
                if _is_garbage_table(html, rows):
                    continue
                row_count = len(rows) if rows else None
                col_count = len(rows[0]) if (rows and isinstance(rows[0], list)) else None

                # Title: use the heading above the table, or a generic fallback.
                title = last_heading or f"Table {counter}"

                tables.append(ExtractedTable(
                    id=f"{doc_id}_{_LP_ID_PREFIX}_tbl_{counter:03d}",
                    content=content,
                    page=page_number,
                    title=title,
                    caption="",       # LlamaParse v2 items don't carry a caption field
                    col_count=col_count,
                    row_count=row_count,
                ))
                counter += 1

        return tables


def _download_bytes(url: str, timeout: int = 30) -> bytes:
    with urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _download_bytes_with_retry(
    url: str,
    timeout: int = 30,
    attempts: int = 3,
    backoff_seconds: int = 2,
) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _download_bytes(url, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_s = backoff_seconds * attempt
            print(
                f"[LlamaParseBackend] download retry {attempt}/{attempts} failed: {exc}. "
                f"Retrying in {sleep_s}s..."
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Download failed after retries: {last_exc}") from last_exc

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