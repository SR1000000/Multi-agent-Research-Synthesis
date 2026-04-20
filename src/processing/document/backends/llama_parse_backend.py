from __future__ import annotations

import base64
import os
import re
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
    ExtractedImage,
    ExtractedTable,
    ExtractionResult,
    PaperMetadata,
)

_TIER = "agentic"
# Prefix embedded in every artifact ID produced by this backend.
# Other backends should define their own prefix (e.g. "dl" for Docling)
# so IDs remain self-describing and collision-free across providers.
_LP_ID_PREFIX = "lp"
_VERSION = "latest"
_PARSE_CREATE_MAX_ATTEMPTS = 3
_PARSE_WAIT_MAX_ATTEMPTS = 3
_PARSE_WAIT_BACKOFF_SECONDS = 2
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
            "For every table, include its full caption or title above the table. "
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

# Matches Markdown image syntax used by LlamaParse: ![alt](url-or-filename)
_IMAGE_REF_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
# Matches inline HTML tables emitted by LlamaParse (output_tables_as_markdown=False)
_HTML_TABLE_PATTERN = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE | re.DOTALL)
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
def _rewrite_markdown_refs(
    markdown: str,
    image_url_to_id: dict[str, str],
    image_caption_to_id: dict[str, str],
    table_ids: list[str],
    page_to_img_ids: dict[int, list[str]] | None = None,
    img_id_to_page: dict[str, int] | None = None,
) -> str:
    """
    Replace LlamaParse's temporary presigned image URLs and inline HTML tables
    with stable cross-reference tokens so chunks stay linked to DB records.

    Image refs:  ``![caption](https://presigned-url...)``  →  ``[[img:{id}]]``
    Image refs:  ``![caption](image)``  (LLM placeholder) →  ``[[img:{id}]]``  (matched by caption)
    Table refs:  ``<table>...</table>``                    →  ``[[tbl:{id}]]``

    Multi-panel figures: when a primary image is placed its same-page siblings
    are emitted as additional tokens immediately adjacent, preventing them from
    being stolen by the next caption match further in the document.
    """
    _page_to_imgs = page_to_img_ids or {}
    _img_to_page = img_id_to_page or {}
    used_ids: set[str] = set()

    def _image_repl(match: re.Match) -> str:
        alt = match.group(1).strip()
        url = match.group(2).strip()

        # Primary: match on URL, basename, or filename
        img_id = (
            image_url_to_id.get(url)
            or image_url_to_id.get(Path(url).name)
        )

        # Fallback: LlamaParse agentic tier uses url='image' as a placeholder;
        # in that case match on the alt text, which equals the figure caption.
        if not img_id and url == "image" and alt:
            img_id = image_caption_to_id.get(alt)

        if not img_id or img_id in used_ids:
            return match.group(0)  # leave unrecognised or already-placed refs intact

        tokens: list[str] = [f"[[img:{img_id}]]"]
        used_ids.add(img_id)

        # Sibling tokens: other layout images sharing the same page that haven't
        # been placed yet. These are sub-panels of the same multi-part figure.
        page = _img_to_page.get(img_id)
        if page is not None:
            for sibling_id in _page_to_imgs.get(page, []):
                if sibling_id != img_id and sibling_id not in used_ids:
                    tokens.append(f"[[img:{sibling_id}]]")
                    used_ids.add(sibling_id)

        return " ".join(tokens)

    markdown = _IMAGE_REF_PATTERN.sub(_image_repl, markdown)

    # Replace the N-th <table>…</table> block with the N-th table token.
    tbl_index = [0]  # mutable cell for closure

    def _table_repl(match: re.Match) -> str:
        i = tbl_index[0]
        tbl_index[0] += 1
        if i < len(table_ids):
            return f"[[tbl:{table_ids[i]}]]"
        return match.group(0)  # more HTML tables than extracted → leave as-is

    markdown = _HTML_TABLE_PATTERN.sub(_table_repl, markdown)
    return markdown


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

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        source = Path(source_pdf_path)
        doc_id = source.stem

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

        # ── Extract tables & images first so we can build reference maps ──────
        tables = self._extract_tables(doc_id, parse_result)
        images = self._extract_images(doc_id, parse_result)
        equations = _extract_equations_from_markdown(doc_id, markdown_full)
        paper_metadata = _parse_paper_metadata(markdown_full) if markdown_full else None

        # ── Rewrite markdown: replace presigned URLs / HTML tables with tokens ─
        image_url_to_id, image_caption_to_id, page_to_img_ids, img_id_to_page = \
            self._build_image_url_map(doc_id, parse_result)
        table_ids = [tbl.id for tbl in tables]
        rewritten_markdown = _rewrite_markdown_refs(
            markdown_full,
            image_url_to_id,
            image_caption_to_id,
            table_ids,
            page_to_img_ids=page_to_img_ids,
            img_id_to_page=img_id_to_page,
        )

        if image_url_to_id or image_caption_to_id or table_ids:
            self._logger.log(
                f"[LlamaParseBackend] Rewrote markdown refs: "
                f"url_mapped={len(image_url_to_id)} caption_mapped={len(image_caption_to_id)} "
                f"pages_with_images={len(page_to_img_ids)} table_tokens={len(table_ids)}"
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
                        self._client.parsing.wait_for_completion(job_id)
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

    def _build_image_url_map(
        self, doc_id: str, parse_result: Any
    ) -> tuple[dict[str, str], dict[str, str], dict[int, list[str]], dict[str, int]]:
        """
        Build lookup dicts for rewriting image refs in markdown_full.

        Returns:
            url_to_id:       presigned URL / filename / basename → img_id
            caption_to_id:   figure caption text → img_id
                             (only the FIRST layout image per page is caption-eligible;
                             subsequent images on the same page are sub-panels that get
                             emitted as siblings rather than matched by caption)
            page_to_img_ids: page number → ordered list of img_ids on that page
            img_id_to_page:  img_id → page number (reverse of page_to_img_ids)
        """
        url_to_id: dict[str, str] = {}
        caption_to_id: dict[str, str] = {}
        page_to_img_ids: dict[int, list[str]] = {}
        img_id_to_page: dict[str, int] = {}

        # ─ Step 1: caption → ordinal position, using the items walk ──────────────
        items_block = _attr(parse_result, "items")
        item_pages = _attr(items_block, "pages") or []
        img_item_counter = 0
        meta_index_to_caption: dict[int, str] = {}
        for page in item_pages:
            for item in (_attr(page, "items") or []):
                if str(_attr(item, "type", "")).lower() == "image":
                    caption = str(_attr(item, "caption") or "").strip()
                    if caption:
                        meta_index_to_caption[img_item_counter] = caption
                    img_item_counter += 1

        # ─ Step 2: walk images_content_metadata ──────────────────────────
        images_meta = _attr(parse_result, "images_content_metadata")
        image_entries = _attr(images_meta, "images") or []

        caption_counter = 0            # only advances for the first image on each new page
        last_caption_page: int | None = None

        for entry in image_entries:
            category = str(_attr(entry, "category") or "").lower()
            if category in ("screenshot", "embedded"):
                continue  # mirrors the filter in _extract_images

            presigned_url: str | None = _attr(entry, "presigned_url")
            if not presigned_url:
                continue

            index: int = _attr(entry, "index", 0)
            filename: str = str(_attr(entry, "filename") or f"img_{index}.bin")
            img_id = f"{doc_id}_{_LP_ID_PREFIX}_img_{index + 1:03d}"

            # URL/filename map
            url_to_id[presigned_url] = img_id
            url_to_id[filename] = img_id
            url_to_id[Path(filename).name] = img_id

            # Page grouping
            page_num = _page_from_filename(filename)
            if page_num is not None:
                page_to_img_ids.setdefault(page_num, []).append(img_id)
                img_id_to_page[img_id] = page_num

            # Caption map: only the FIRST layout image on each page gets a caption.
            # Subsequent images on the same page are sub-panels; they are emitted
            # as siblings of the primary token rather than having their own entry.
            if page_num != last_caption_page:
                caption = meta_index_to_caption.get(caption_counter)
                if caption:
                    caption_to_id[caption] = img_id
                caption_counter += 1
                last_caption_page = page_num

        return url_to_id, caption_to_id, page_to_img_ids, img_id_to_page

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

        Caption and Confidence are NOT on ImagesContentMetadataImage — they live on the
        corresponding ImageItem in items.pages[*].items. We build a lookup
        map from index → (caption, confidence) using the items tree first.
        """
        
        meta_by_index: dict[int, dict[str, Any]] = {}
        items_block = _attr(parse_result, "items")
        item_pages = _attr(items_block, "pages") or []
        img_item_counter = 0  # items are 0-indexed globally in order
        for page in item_pages:
            for item in (_attr(page, "items") or []):
                if str(_attr(item, "type", "")).lower() == "image":
                    caption = str(_attr(item, "caption") or "").strip()
                    # Item bbox is a list of dicts with confidence
                    item_bboxes = _attr(item, "bbox") or []
                    confidence = _attr(item_bboxes[0], "confidence") if item_bboxes else None
                    
                    meta_by_index[img_item_counter] = {
                        "caption": caption,
                        "confidence": confidence
                    }
                    img_item_counter += 1

        images_meta = _attr(parse_result, "images_content_metadata")
        image_entries = _attr(images_meta, "images") or []

        # Caption assignment: only the FIRST layout image on each page gets a
        # caption from the items tree.  Additional images on the same page are
        # sub-panels of the same figure; they are stored without a caption and
        # referenced via the sibling-injection mechanism in _rewrite_markdown_refs.
        caption_counter: int = 0
        last_caption_page: int | None = None

        extracted: list[ExtractedImage] = []
        for entry in image_entries:
            category = str(_attr(entry, "category") or "").lower()
            # Only keep layout detections — these cover every page, including
            # pages where figures are not embedded PDF objects.
            # Skip screenshots and embedded objects.
            if category in ("screenshot", "embedded"):
                continue

            presigned_url: str | None = _attr(entry, "presigned_url")
            if not presigned_url:
                continue

            index: int = _attr(entry, "index", 0)
            filename: str = str(_attr(entry, "filename") or f"img_{index}.bin")
            mime_type: str = str(_attr(entry, "content_type") or "application/octet-stream")
            raw_bbox = _attr(entry, "bbox")
            bbox = _serialize_parse_payload(raw_bbox) if raw_bbox else None

            # Advance caption only when we move to a new page.
            page_num = _page_from_filename(filename)
            confidence = None
            if page_num != last_caption_page:
                item_meta = meta_by_index.get(caption_counter, {})
                caption = item_meta.get("caption", "")
                confidence = item_meta.get("confidence")
                caption_counter += 1
                last_caption_page = page_num
            else:
                caption = ""  # sub-panel, no separate caption

            try:
                image_bytes = _download_bytes_with_retry(
                    presigned_url,
                    attempts=_IMAGE_DOWNLOAD_MAX_ATTEMPTS,
                    backoff_seconds=_IMAGE_DOWNLOAD_BACKOFF_SECONDS,
                )
            except Exception as exc:
                print(f"[LlamaParseBackend] Failed to download image {filename}: {exc}")
                continue

            img_id = f"{doc_id}_{_LP_ID_PREFIX}_img_{index + 1:03d}"
            ext = _extension(filename, mime_type)
            storage_key = f"{doc_id}/images/{img_id}.{ext}"
            storage_path: str | None = None
            base64_data = ""
            try:
                storage_path = self._object_store.write(storage_key, image_bytes)
            except Exception as exc:
                print(f"[LlamaParseBackend] Failed to store image {filename} in object store: {exc}")
                # Fallback to storing as base64 data
                base64_data = base64.b64encode(image_bytes).decode("utf-8")

            extracted.append(ExtractedImage(
                id=img_id,
                mime_type=mime_type,
                base64_data=base64_data,
                page=page_num,
                caption=caption,
                storage_path=storage_path,
                bbox=bbox,
                source_filename=filename,
                confidence=confidence,
                category=category,
            ))

        return extracted


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