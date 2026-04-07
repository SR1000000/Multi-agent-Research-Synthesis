import base64
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, TypeVar

import psutil
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode

from .._common import _slugify, _verify_references_in_markdown, build_artifact_references
from ..backend_base import OCRBackend
from ..schema import (
    ArtifactReference,
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionManifest,
    ExtractionResult,
    T,
)

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

_EQUATION_TOKEN_PATTERN = re.compile(r"\[\[eq:eq_\d{3}\]\]")
_INLINE_EQUATION_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_BLOCK_EQUATION_PATTERN = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s")


# ---------------------------------------------------------------------------
# Environment / setup helpers
# ---------------------------------------------------------------------------


def _disable_hf_symlink_usage_on_windows() -> None:
    """Force huggingface_hub to copy files instead of symlinking on Windows."""
    if os.name != "nt":
        return

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import huggingface_hub.file_download as hf_file_download
    except ImportError:
        # Consider logging a warning here if huggingface_hub is an expected dependency.
        return

    hf_file_download._are_symlinks_supported_in_dir.clear()

    def _always_false(_cache_dir: str | Path | None = None) -> bool:
        return False

    hf_file_download.are_symlinks_supported = _always_false


def _build_pdf_pipeline_options() -> PdfPipelineOptions:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=psutil.cpu_count(logical=False), device="cpu"
    )
    return pipeline_options


# ---------------------------------------------------------------------------
# Document-level helpers (operate on DoclingDocument)
# ---------------------------------------------------------------------------


def _infer_heading_depth_from_numbering(document: DoclingDocument) -> None:
    """Infer proper heading depth from section numbering."""
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


def _extract_page_no(item: Any) -> int | None:
    provs = getattr(item, "prov", None)
    if not provs:
        return None
    first = provs[0]
    page_no = getattr(first, "page_no", None)
    return int(page_no) if isinstance(page_no, int) else None


def _extract_caption(item: Any) -> str:
    captions = getattr(item, "captions", None)
    if not captions:
        return ""
    pieces: list[str] = []
    for cap in captions:
        text = getattr(cap, "text", "")
        if text:
            pieces.append(str(text).strip())
    return " ".join(pieces).strip()


def _get_artifact_bbox(item: Any) -> tuple[float, float, float, float] | None:
    """Return the (l, t, r, b) bounding box from an item's first provenance entry, or None."""
    provs = getattr(item, "prov", None)
    if not provs:
        return None
    bbox = getattr(provs[0], "bbox", None)
    if bbox is None:
        return None
    return (bbox.l, bbox.t, bbox.r, bbox.b)


def _find_best_chunk_for_artifact(
    artifact_page: int | None,
    artifact_bbox: tuple[float, float, float, float] | None,
    source_chunks: list[ExtractedChunk],
) -> int:
    """Return the index of the best chunk to attach an artifact token to.

    Primary: chunks on the same page as the artifact, ranked by minimum Euclidean
    distance from the artifact centroid to any of the chunk's bboxes.
    Fallback: when no chunk shares the page, or when bbox data is absent, pick the
    chunk on the nearest page. Never blindly falls back to source_chunks[-1].
    """
    if not source_chunks:
        return 0

    # Compute artifact centroid when bbox is available
    cx: float | None = None
    cy: float | None = None
    if artifact_bbox is not None:
        l, t, r, b = artifact_bbox
        cx = (l + r) / 2.0
        cy = (t + b) / 2.0

    def _bbox_distance(bboxes: list[tuple[float, float, float, float, int]], page: int) -> float:
        """Minimum Euclidean distance from (cx, cy) to any bbox on the given page."""
        if cx is None or cy is None:
            return 0.0
        best = float("inf")
        for bl, bt, br, bb, bp in bboxes:
            if bp != page:
                continue
            # Clamp the centroid to the bbox and measure distance
            nearest_x = max(bl, min(cx, br))
            nearest_y = max(bt, min(cy, bb))
            dist = ((cx - nearest_x) ** 2 + (cy - nearest_y) ** 2) ** 0.5
            if dist < best:
                best = dist
        return best if best < float("inf") else float("inf")

    # Try to find chunks that share the artifact's page
    if artifact_page is not None:
        same_page = [
            (i, chunk) for i, chunk in enumerate(source_chunks)
            if artifact_page in chunk.page_numbers
        ]
        if same_page:
            if cx is not None and cy is not None:
                best_idx, _ = min(
                    same_page,
                    key=lambda t: _bbox_distance(t[1].bboxes, artifact_page),
                )
            else:
                # No bbox — pick the last chunk on the page (most likely after inline content)
                best_idx = same_page[-1][0]
            return best_idx

    # Fallback: pick the chunk on the nearest page
    def _page_distance(chunk: ExtractedChunk) -> float:
        if not chunk.page_numbers:
            return float("inf")
        if artifact_page is None:
            return float("inf")
        return float(min(abs(p - artifact_page) for p in chunk.page_numbers))

    return min(range(len(source_chunks)), key=lambda i: _page_distance(source_chunks[i]))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _extract_chunks(document: DoclingDocument) -> list[ExtractedChunk]:
    chunker = HybridChunker()
    source_chunks: list[ExtractedChunk] = []
    for chunk in chunker.chunk(dl_doc=document):
        page_numbers: list[int] = []
        bboxes: list[tuple[float, float, float, float, int]] = []
        for doc_item in (chunk.meta.doc_items or []):
            for prov in (getattr(doc_item, "prov", None) or []):
                page_no = getattr(prov, "page_no", None)
                bbox = getattr(prov, "bbox", None)
                if page_no is not None:
                    page_numbers.append(page_no)
                if bbox is not None and page_no is not None:
                    bboxes.append((bbox.l, bbox.t, bbox.r, bbox.b, page_no))
        source_chunks.append(
            ExtractedChunk(
                text=chunk.text,
                contextualized_text=chunker.contextualize(chunk),
                headings=list(chunk.meta.headings) if chunk.meta.headings else [],
                captions=list(chunk.meta.captions) if chunk.meta.captions else [],
                page_numbers=sorted(set(page_numbers)),
                bboxes=bboxes,
            )
        )
    return source_chunks


# ---------------------------------------------------------------------------
# Artifact extraction
# ---------------------------------------------------------------------------


def _extract_artifact(
    item: Any,
    item_id: str,
    kind: Literal["image", "table"],
    document: DoclingDocument,
) -> ExtractedImage | ExtractedTable | None:
    if kind == "image":
        image = item.get_image(document)
        if image is None:
            return None
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return ExtractedImage(
            id=item_id,
            mime_type="image/png",
            base64_data=base64_str,
            page=_extract_page_no(item),
            caption=_extract_caption(item),
        )
    elif kind == "table":
        html = item.export_to_html(doc=document)
        idx = int(item_id.split("_")[-1])
        return ExtractedTable(
            id=item_id,
            html_content=html,
            page=_extract_page_no(item),
            title=f"Table {idx}",
        )
    return None


def _extract_artifacts(
    document: DoclingDocument,
    attr_name: str,
    prefix: str,
    kind: Literal["image", "table"],
    expected_type: type[T],
) -> list[tuple[Any, T]]:
    """Extract artifacts and return (source_doc_item, extracted) pairs.

    Retaining the source DocItem allows callers to access its prov bbox
    directly during artifact token injection.
    """
    pairs: list[tuple[Any, T]] = []
    items = getattr(document, attr_name, [])
    for idx, item in enumerate(items, start=1):
        item_id = f"{prefix}_{idx:03d}"
        res = _extract_artifact(item, item_id, kind, document)
        if res and isinstance(res, expected_type):
            pairs.append((item, res))
    return pairs


def _build_image_metadata(document: DoclingDocument) -> list[tuple[Any, ExtractedImage]]:
    return _extract_artifacts(
        document=document,
        attr_name="pictures",
        prefix="img",
        kind="image",
        expected_type=ExtractedImage,
    )


# ---------------------------------------------------------------------------
# Equation annotation
# ---------------------------------------------------------------------------


def _annotate_equations(markdown_text: str) -> tuple[str, list[ExtractedEquation]]:
    equations: list[ExtractedEquation] = []
    counter = 1

    def block_repl(match: re.Match[str]) -> str:
        nonlocal counter
        expression = match.group(1).strip()
        eq_id = f"eq_{counter:03d}"
        token = f"[[eq:{eq_id}]]"
        equations.append(
            ExtractedEquation(
                id=eq_id,
                latex_or_text=expression,
                display_mode="block",
                page=None,
                markdown_anchor=token,
            )
        )
        counter += 1
        return f"$${expression}$$\n<!-- {token} -->"

    markdown_text = _BLOCK_EQUATION_PATTERN.sub(block_repl, markdown_text)

    def inline_repl(match: re.Match[str]) -> str:
        nonlocal counter
        expression = match.group(1).strip()
        eq_id = f"eq_{counter:03d}"
        token = f"[[eq:{eq_id}]]"
        equations.append(
            ExtractedEquation(
                id=eq_id,
                latex_or_text=expression,
                display_mode="inline",
                page=None,
                markdown_anchor=token,
            )
        )
        counter += 1
        return f"${expression}$<!-- {token} -->"

    markdown_text = _INLINE_EQUATION_PATTERN.sub(inline_repl, markdown_text)
    return markdown_text, equations


def _annotate_markdown(
    markdown_text: str,
    images: list[ExtractedImage],
    tables: list[ExtractedTable],
) -> tuple[str, list[ExtractedEquation]]:
    markdown_text, equations = _annotate_equations(markdown_text)
    lines = ["", "## Artifact References", ""]
    for item in images:
        lines.append(f"- [[img:{item.id}]] -> Attached Image {item.id}")
    for item in tables:
        lines.append(f"- [[tbl:{item.id}]] -> Attached Table {item.id}")
    lines.append("")
    markdown_text = markdown_text.rstrip() + "\n" + "\n".join(lines)
    return markdown_text, equations


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class DoclingBackend(OCRBackend):
    """OCR backend powered by Docling.

    Owns the full extraction pipeline: PDF parsing via DocumentConverter,
    hybrid chunking, image/table/equation extraction, markdown annotation,
    and manifest generation.
    """

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        _disable_hf_symlink_usage_on_windows()
        source = Path(source_pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")

        start_time = time.time()
        doc_id = _slugify(source.stem) or "document"
        print(f"Starting extraction for {source.name} (ID: {doc_id})")

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=_build_pdf_pipeline_options()
                ),
            }
        )

        print("Parsing and converting PDF into Docling Document...")
        conv_res = converter.convert(source)
        document = conv_res.document
        _infer_heading_depth_from_numbering(document)

        print("Chunking document in memory...")
        source_chunks = _extract_chunks(document)

        print("Extracting markdown with referenced images...")
        markdown_text = document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED
        )

        print("Extracting image metadata and tables from memory...")
        image_pairs = _build_image_metadata(document)
        table_pairs = _extract_artifacts(
            document=document,
            attr_name="tables",
            prefix="tbl",
            kind="table",
            expected_type=ExtractedTable,
        )
        images = [img for _, img in image_pairs]
        tables = [tbl for _, tbl in table_pairs]

        print("Annotating equations...")
        markdown_text, equations = _annotate_markdown(
            markdown_text=markdown_text,
            images=images,
            tables=tables,
        )

        print("Building references...")
        references = build_artifact_references(
            ("image", "img", images),
            ("table", "tbl", tables),
            ("equation", "eq", equations),
        )

        print("Injecting artifact tokens into chunks...")
        for eq in equations:
            for chunk in source_chunks:
                if eq.latex_or_text and eq.latex_or_text in chunk.text:
                    chunk.text = chunk.text.replace(
                        eq.latex_or_text,
                        f"{eq.latex_or_text} {eq.markdown_anchor}",
                    )
                    chunk.contextualized_text = chunk.contextualized_text.replace(
                        eq.latex_or_text,
                        f"{eq.latex_or_text} {eq.markdown_anchor}",
                    )

        for doc_item, img in image_pairs:
            token = f"[[img:{img.id}]]"
            bbox = _get_artifact_bbox(doc_item)
            idx = _find_best_chunk_for_artifact(img.page, bbox, source_chunks)
            source_chunks[idx].text += f"\n\n{token}"
            source_chunks[idx].contextualized_text += f"\n\n{token}"

        for doc_item, tbl in table_pairs:
            token = f"[[tbl:{tbl.id}]]"
            bbox = _get_artifact_bbox(doc_item)
            idx = _find_best_chunk_for_artifact(tbl.page, bbox, source_chunks)
            source_chunks[idx].text += f"\n\n{token}"
            source_chunks[idx].contextualized_text += f"\n\n{token}"

        print("Generating manifest...")
        manifest = ExtractionManifest(
            doc_id=doc_id,
            source_pdf_path=str(source),
            markdown_path="",
            images=images,
            tables=tables,
            equations=equations,
            references=references,
        )

        print("Validating references...")
        _verify_references_in_markdown(
            markdown_text=markdown_text, manifest=manifest
        )

        print(f"[{time.time() - start_time:.2f}s] Extraction complete.")
        return ExtractionResult(
            source_chunks=source_chunks,
            manifest_json=manifest,
            image_count=len(images),
            table_count=len(tables),
            equation_count=len(equations),
            chunk_count=len(source_chunks),
        )
