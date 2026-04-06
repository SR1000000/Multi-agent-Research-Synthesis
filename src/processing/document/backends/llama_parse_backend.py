from __future__ import annotations

import base64
import os
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
)


def _llama_parse_parse_kwargs() -> dict[str, Any]:
    """Keyword arguments for `LlamaCloud.parsing.parse(...)`.

    Content vs metadata `expand` values:
    https://developers.llamaindex.ai/python/cloud/llamaparse/basics/retrieving-results/
    """
    return {
        "tier": "cost_effective",
        "version": "latest",
        "output_options": {
            "images_to_save": ["embedded", "layout"],
        },
        "expand": [
            "items",
            "metadata",
            "markdown_full",
            "text_full",
            "images_content_metadata",
        ],
    }


def _read_attr(data: Any, key: str, default: Any = None) -> Any:
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _read_nested(data: Any, *keys: str, default: Any = None) -> Any:
    current = data
    for key in keys:
        current = _read_attr(current, key, None)
        if current is None:
            return default
    return current


class LlamaParseBackend(OCRBackend):
    """OCR backend powered by LlamaParse (Llama Cloud)."""

    def __init__(
        self,
        object_store: ObjectStoreProvider | None = None,
        text_chunker: TextChunkerProvider | None = None,
    ) -> None:
        api_key = self._resolve_llama_api_key()

        self._client = LlamaCloud(api_key=api_key)
        self._object_store = object_store or LocalObjectStore()
        self._text_chunker = text_chunker

    def _resolve_llama_api_key(self) -> str | None:
        return os.environ.get("LLAMA_API_KEY")

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        source = Path(source_pdf_path)
        doc_id = source.stem
        parse_result = self._parse_document(source)

        markdown_full = _read_attr(parse_result, "markdown_full", "") or ""
        metadata_block = _read_attr(parse_result, "metadata", {}) or {}
        pages = _read_attr(metadata_block, "pages", []) or []
        page_count = len(pages)

        chunks = self._extract_chunks(doc_id, markdown_full, metadata_block)
        tables, equations = self._extract_structured_items(doc_id, parse_result)
        images = self._extract_images(doc_id, parse_result)

        return ExtractionResult(
            doc_id=doc_id,
            source_path=str(source),
            markdown=markdown_full,
            source_chunks=chunks,
            images=images,
            tables=tables,
            equations=equations,
            page_count=page_count,
            schema="llamaparse/cost_effective/latest",
        )

    def _parse_document(self, source: Path) -> Any:
        return self._client.parsing.parse(
            upload_file=str(source),
            **_llama_parse_parse_kwargs(),
        )

    def _extract_chunks(self, doc_id: str, markdown_full: str, metadata: Any) -> list[ExtractedChunk]:
        if self._text_chunker is None:
            text = (markdown_full or "").strip()
            if not text:
                text = str(_read_nested(metadata, "document", "text_full", default="") or "").strip()
            if not text:
                return []
            return [
                ExtractedChunk(
                    id=f"{doc_id}_chunk_0000",
                    text=text,
                    meta_data={"chunk_index": 0, "splitter": "none"},
                )
            ]

        text = (markdown_full or "").strip()
        if text:
            raw_chunks = self._text_chunker.chunk_markdown(text)
            splitter_name = type(self._text_chunker).__name__
        else:
            fallback_text = str(_read_nested(metadata, "document", "text_full", default="") or "")
            raw_chunks = self._text_chunker.chunk_text(fallback_text)
            splitter_name = type(self._text_chunker).__name__

        extracted: list[ExtractedChunk] = []
        for index, value in enumerate(raw_chunks):
            chunk_text = value.strip()
            if not chunk_text:
                continue
            extracted.append(
                ExtractedChunk(
                    id=f"{doc_id}_chunk_{index:04d}",
                    text=chunk_text,
                    meta_data={
                        "chunk_index": index,
                        "splitter": splitter_name,
                    },
                )
            )
        return extracted

    def _extract_structured_items(
        self, doc_id: str, parse_result: Any
    ) -> tuple[list[ExtractedTable], list[ExtractedEquation]]:
        tables: list[ExtractedTable] = []
        equations: list[ExtractedEquation] = []
        equation_seen: set[str] = set()
        table_counter = 1
        equation_counter = 1

        pages = _read_nested(parse_result, "items", "pages", default=[]) or []
        for page in pages:
            page_number = _read_attr(page, "page_number")
            page_items = _read_attr(page, "items", []) or []
            for item in page_items:
                item_type = str(_read_attr(item, "type", "")).lower()
                if item_type == "table":
                    html = _read_attr(item, "html", None)
                    md = _read_attr(item, "md", None)
                    csv = _read_attr(item, "csv", None)
                    content = html or md or csv or ""
                    if not content:
                        continue

                    title = _read_attr(item, "value", None) or f"Table {table_counter}"
                    rows = _read_attr(item, "rows", None)
                    col_count = None
                    row_count = None
                    if isinstance(rows, list) and rows:
                        row_count = len(rows)
                        if isinstance(rows[0], list):
                            col_count = len(rows[0])

                    tables.append(
                        ExtractedTable(
                            id=f"{doc_id}_tbl_{table_counter:03d}",
                            content=content,
                            page=page_number if isinstance(page_number, int) else None,
                            title=str(title),
                            col_count=col_count,
                            row_count=row_count,
                        )
                    )
                    table_counter += 1
                    continue

                if item_type in ("equation", "formula", "math", "math_formula"):
                    raw_value = _read_attr(item, "value", None) or _read_attr(item, "md", None)
                    expression = str(raw_value or "").strip()
                    if not expression or expression in equation_seen:
                        continue
                    equation_seen.add(expression)
                    equations.append(
                        ExtractedEquation(
                            id=f"{doc_id}_eq_{equation_counter:03d}",
                            latex_or_text=expression,
                            display_mode="block",
                            page=page_number if isinstance(page_number, int) else None,
                        )
                    )
                    equation_counter += 1

        return tables, equations

    def _extract_images(self, doc_id: str, parse_result: Any) -> list[ExtractedImage]:
        images_data = _read_attr(parse_result, "images_content_metadata", None)
        image_entries = _read_attr(images_data, "images", []) if images_data else []
        if not image_entries:
            return []

        extracted_images: list[ExtractedImage] = []
        for index, image_info in enumerate(image_entries, start=1):
            image_url = _read_attr(image_info, "presigned_url")
            if not image_url:
                continue
            filename = str(_read_attr(image_info, "filename", f"image_{index}.bin"))
            mime_type = str(_read_attr(image_info, "content_type", "application/octet-stream"))
            page_number = _read_attr(image_info, "page_number")
            caption = str(_read_attr(image_info, "caption", "") or "")

            image_bytes = self._download_image(image_url)
            extension = self._extension_from_filename_or_mime(filename, mime_type)
            image_id = f"{doc_id}_img_{index:03d}"
            storage_key = f"{doc_id}/images/{image_id}.{extension}"
            self._object_store.write(storage_key, image_bytes)

            extracted_images.append(
                ExtractedImage(
                    id=image_id,
                    mime_type=mime_type,
                    base64_data=base64.b64encode(image_bytes).decode("utf-8"),
                    page=page_number if isinstance(page_number, int) else None,
                    caption=caption,
                )
            )

        return extracted_images

    def _download_image(self, url: str) -> bytes:
        with urlopen(url, timeout=30) as response:
            return response.read()

    def _extension_from_filename_or_mime(self, filename: str, mime_type: str) -> str:
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1].strip().lower()
            if ext:
                return ext
        if mime_type == "image/png":
            return "png"
        if mime_type == "image/jpeg":
            return "jpg"
        if mime_type == "image/webp":
            return "webp"
        return "bin"
