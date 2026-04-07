import base64
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import psutil
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, FormulaItem, ImageRefMode

from ..backend_base import OCRBackend
from ..schema import (
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionResult,
)

_INLINE_EQUATION_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_BLOCK_EQUATION_PATTERN = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s")

def _disable_hf_symlink_usage_on_windows() -> None:
    """
    Force huggingface_hub to copy files instead of symlinking on Windows.
    This prevents PermissionError on Windows systems where symlinks require admin rights.
    """
    if os.name != "nt":
        return

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import huggingface_hub.file_download as hf_file_download
    except ImportError:
        return

    hf_file_download._are_symlinks_supported_in_dir.clear()

    def _always_false(_cache_dir: str | Path | None = None) -> bool:
        return False

    hf_file_download.are_symlinks_supported = _always_false

class DoclingBackend(OCRBackend):
    """
    OCR Backend powered by Docling.
    Handles PDF conversion, hybrid chunking, and multimodal artifact extraction.
    """
    def __init__(self) -> None:
        _disable_hf_symlink_usage_on_windows()
        self._converter: DocumentConverter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self._build_pdf_pipeline_options()
                )
            }
        )

    def _build_pdf_pipeline_options(self) -> PdfPipelineOptions:
        """
        Builds the default pipeline options for PDF processing.
        Enables image generation, formula enrichment, and CPU acceleration.
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_picture_images = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=psutil.cpu_count(logical=False), device="cpu"
        )
        return pipeline_options

    def _infer_heading_depth_from_numbering(self, document: DoclingDocument) -> None:
        """
        Infers heading levels based on section numbering (e.g., 1.1 -> level 2).
        Updates the 'level' attribute of section header items in the document.
        """
        headers = [
            item for item in document.texts
            if "section_header" in str(getattr(item, "label", ""))
        ]

        has_numbered = False
        for item in headers:
            text = (getattr(item, "text", "") or "").strip()
            m = _SECTION_NUMBER_RE.match(text)
            if m:
                item.level = m.group(1).count(".") + 1
                has_numbered = True

        if not has_numbered:
            return

        last_numbered_level = 0
        for item in headers:
            text = (getattr(item, "text", "") or "").strip()
            if _SECTION_NUMBER_RE.match(text):
                last_numbered_level = item.level
            elif last_numbered_level > 1:
                item.level = last_numbered_level + 1

    def _extract_page_no(self, item: Any) -> int | None:
        """Extracts the 1-indexed page number from a Docling document item's provenance."""
        provs = getattr(item, "prov", None)
        if not provs:
            return None
        first = provs[0]
        page_no = getattr(first, "page_no", None)
        return int(page_no) if (page_no is not None and isinstance(page_no, int)) else None

    def _extract_caption(self, item: Any, document: DoclingDocument) -> str:
        """Resolves the caption string for a picture or table item via caption_text()."""
        try:
            text = item.caption_text(doc=document)
            return text.strip() if text else ""
        except Exception:
            return ""

    def extract(self, source_path: str) -> ExtractionResult:
        """
        Orchestrates the full extraction pipeline for a given PDF source.
        Parses the doc, chunks text, and extracts multimodal artifacts.
        """
        source = Path(source_path)
        doc_id = source.stem

        conv_res = self._converter.convert(source)
        document = conv_res.document
        self._infer_heading_depth_from_numbering(document)

        # Export to markdown with referenced images for visual context in LLM
        markdown = document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)

        chunks = self._extract_text_chunks(document, doc_id)
        images, tables, equations = self._extract_multimodal_artifacts(document, doc_id, markdown)

        return ExtractionResult(
            doc_id=doc_id,
            source_path=str(source),
            markdown=markdown,
            source_chunks=chunks,
            images=images,
            tables=tables,
            equations=equations,
            page_count=len(document.pages) if hasattr(document, "pages") else 0,
            schema=getattr(document, "version", None),
        )

    def _extract_text_chunks(self, document: DoclingDocument, doc_id: str) -> list[ExtractedChunk]:
        """
        Performs semantic chunking using HybridChunker.
        Collects page numbers, docling item labels, and captions per chunk.
        """
        chunker = HybridChunker()
        extracted_chunks: list[ExtractedChunk] = []
        for idx, chunk in enumerate(chunker.chunk(dl_doc=document)):
            page_numbers: list[int] = []
            labels: list[str] = []
            for doc_item in (chunk.meta.doc_items or []):
                label = getattr(doc_item, "label", None)
                if label:
                    labels.append(str(label))
                for prov in (getattr(doc_item, "prov", None) or []):
                    page_no = getattr(prov, "page_no", None)
                    if page_no is not None:
                        page_numbers.append(page_no)

            extracted_chunks.append(
                ExtractedChunk(
                    id=f"{doc_id}_chunk_{idx:04d}",
                    text=chunk.text,
                    meta_data={
                        "headings": list(chunk.meta.headings) if chunk.meta.headings else [],
                        "captions": list(chunk.meta.captions) if chunk.meta.captions else [],
                        "page_numbers": sorted(set(page_numbers)),
                        "chunk_index": idx
                    }
                )
            )
        return extracted_chunks

    def _extract_multimodal_artifacts(
        self, document: DoclingDocument, doc_id: str, markdown: str
    ) -> tuple[list[ExtractedImage], list[ExtractedTable], list[ExtractedEquation]]:
        """
        Extracts images, tables, and equations from the document.
        """
        # Images
        images: list[ExtractedImage] = []
        for idx, item in enumerate(document.pictures, start=1):
            res = self._extract_image(item, f"{doc_id}_img_{idx:03d}", document)
            if res is not None:
                images.append(res)

        # Tables
        tables: list[ExtractedTable] = []
        for idx, item in enumerate(document.tables, start=1):
            res = self._extract_table(item, f"{doc_id}_tbl_{idx:03d}", document)
            if res is not None:
                tables.append(res)

        equations = self._extract_equations(document, doc_id, markdown)

        return images, tables, equations


    def _extract_equations(
        self, document: DoclingDocument, doc_id: str, markdown: str
    ) -> list[ExtractedEquation]:
        """
        Extracts equations via two complementary passes:
        1. Structured: FormulaItem instances from doc.iterate_items() — carry page provenance.
        2. Regex: block $$...$$ then inline $...$ over the markdown export — catches anything missed.
        A shared seen_expressions set deduplicates across both passes.
        """
        equations: list[ExtractedEquation] = []
        seen_expressions: set[str] = set()
        counter = 1

        for item, _ in document.iterate_items():
            if isinstance(item, FormulaItem):
                expr = (item.text or "").strip()
                if expr and expr not in seen_expressions:
                    seen_expressions.add(expr)
                    equations.append(
                        ExtractedEquation(
                            id=f"{doc_id}_eq_{counter:03d}",
                            latex_or_text=expr,
                            display_mode="block",
                            page=self._extract_page_no(item),
                        )
                    )
                    counter += 1

        for match in _BLOCK_EQUATION_PATTERN.finditer(markdown):
            expr = match.group(1).strip()
            if expr and expr not in seen_expressions:
                seen_expressions.add(expr)
                equations.append(
                    ExtractedEquation(
                        id=f"{doc_id}_eq_{counter:03d}",
                        latex_or_text=expr,
                        display_mode="block",
                    )
                )
                counter += 1

        for match in _INLINE_EQUATION_PATTERN.finditer(markdown):
            expr = match.group(1).strip()
            if expr and expr not in seen_expressions:
                seen_expressions.add(expr)
                equations.append(
                    ExtractedEquation(
                        id=f"{doc_id}_eq_{counter:03d}",
                        latex_or_text=expr,
                        display_mode="inline",
                    )
                )
                counter += 1

        return equations

    def _extract_image(
        self, item: Any, item_id: str, document: DoclingDocument
    ) -> ExtractedImage | None:
        image = item.get_image(document)
        if image is None:
            return None
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return ExtractedImage(
            id=item_id,
            mime_type="image/png",
            base64_data=img_str,
            page=self._extract_page_no(item),
            caption=self._extract_caption(item, document),
        )

    def _extract_table(
        self, item: Any, item_id: str, document: DoclingDocument
    ) -> ExtractedTable | None:
        html = item.export_to_html(doc=document)
        return ExtractedTable(
            id=item_id,
            content=html,
            page=self._extract_page_no(item),
            title=self._extract_caption(item, document) or f"Table {item_id}",
            col_count=item.data.num_cols if hasattr(item, "data") and hasattr(item.data, "num_cols") else None,
            row_count=item.data.num_rows if hasattr(item, "data") and hasattr(item.data, "num_rows") else None,
        )
