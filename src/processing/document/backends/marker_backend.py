import base64
import os
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

try:
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser as MarkerConfigParser
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
except ImportError:
    PdfConverter = None  # type: ignore
    MarkerConfigParser = None  # type: ignore
    create_model_dict = None  # type: ignore
    text_from_rendered = None  # type: ignore

from .._common import _slugify, _verify_references_in_markdown, build_artifact_references
from ..backend_base import OCRBackend
from ..chunks import MarkdownChunker
from ..schema import (
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionManifest,
    ExtractionResult,
)

_BLOCK_EQUATION_PATTERN = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_INLINE_EQUATION_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

# Matches Marker's native image syntax: ![optional caption](filename)
# Handles both plain filenames and relative paths like _page_0_Figure_1.jpeg
_MARKER_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


@dataclass
class MarkerConfig:
    """Tunable settings for the Marker PDF extraction backend.

    Attributes:
        force_ocr: Force Surya OCR on every page, even those with a text layer.
            Recommended for scanned PDFs or documents with garbled/invisible text.
        strip_existing_ocr: Discard any embedded OCR text in the PDF before
            running fresh recognition. Useful when the existing text layer is
            incorrect or low-quality.
        use_llm: Enable LLM post-processing (requires a Gemini API key via the
            ``GOOGLE_API_KEY`` environment variable). Significantly improves
            accuracy for complex layouts, multi-column tables, and math equations.
        redo_inline_math: Re-process inline math expressions via the LLM.
            Only meaningful when ``use_llm=True``. Produces the highest-quality
            LaTeX conversion at the cost of extra API calls.
        ocr_engine: OCR engine to use.  ``"surya"`` (default) is the bundled
            neural engine — slower but more accurate.  ``"ocrmypdf"`` is faster
            but requires ``ocrmypdf`` to be installed separately.
        output_format: Output format Marker produces internally. Must be
            ``"markdown"`` for this backend to function correctly.
        extra: Any additional key/value pairs forwarded verbatim to Marker's
            ``ConfigParser``.  Use this to pass undocumented or future flags.

    Quick-start recipes
    -------------------
    * **Scanned PDF** → set ``force_ocr=True``
    * **Bad text layer** → set ``strip_existing_ocr=True``
    * **Complex layout / equations** → set ``use_llm=True`` (+ Gemini API key)
    * **Highest quality math** → set ``use_llm=True, redo_inline_math=True``
    * **Fast batch processing** → set ``ocr_engine="ocrmypdf"``
    """

    force_ocr: bool = False
    strip_existing_ocr: bool = False
    use_llm: bool = True
    redo_inline_math: bool = True
    ocr_engine: str = "surya"
    output_format: str = "markdown"
    extra: dict = field(default_factory=dict)

    def to_marker_dict(self) -> dict:
        """Serialize to a dict suitable for Marker's ``ConfigParser``."""
        cfg = {
            "output_format": self.output_format,
            "force_ocr": self.force_ocr,
            "strip_existing_ocr": self.strip_existing_ocr,
            "use_llm": self.use_llm,
            "redo_inline_math": self.redo_inline_math,
            "ocr_engine": self.ocr_engine,
        }
        
        # Marker expects "gemini_api_key" configuration, we support reading from GOOGLE_AI_STUDIO_API_KEY
        if self.use_llm:
            dt_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
            if dt_key:
                cfg["gemini_api_key"] = dt_key
                
        cfg.update(self.extra)
        return cfg


def _annotate_equations(markdown_text: str) -> tuple[str, list[ExtractedEquation]]:
    """Extract and annotate LaTeX equations from markdown, injecting reference tokens."""
    equations: list[ExtractedEquation] = []
    counter = 1

    def block_repl(match: re.Match) -> str:
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

    def inline_repl(match: re.Match) -> str:
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


def _replace_marker_image_refs(
    markdown_text: str,
    filename_to_id: dict[str, str],
) -> str:
    """Replace Marker's native ``![caption](filename)`` image tags with
    ``[[img:img_NNN]]`` reference tokens so that the markdown lines up
    with the IDs stored in the SQLite ``images`` table.

    Marker keys its rendered-image dict by the *basename* of the filename
    (e.g. ``"_page_0_Figure_1.jpeg"``).  The markdown it emits contains
    those same strings as the URL part of standard markdown image syntax.
    We match on basename so that both relative paths and plain names work.
    """

    def repl(match: re.Match) -> str:
        raw_path = match.group(2)
        basename = Path(raw_path).name
        img_id = filename_to_id.get(basename) or filename_to_id.get(raw_path)
        if img_id:
            return f"[[img:{img_id}]]"
        # Unknown image — leave the original tag intact so we don't silently
        # destroy content we don't recognise.
        return match.group(0)

    return _MARKER_IMAGE_PATTERN.sub(repl, markdown_text)


class MarkerBackend(OCRBackend):
    """OCR backend powered by Marker (marker-pdf).

    Uses the PdfConverter pipeline to convert PDFs to high-quality markdown
    with table, image, and equation extraction. Works on CPU, MPS, or GPU.
    Models are downloaded from HuggingFace on first use (~2–4 GB total).

    Parameters
    ----------
    config:
        A :class:`MarkerConfig` instance controlling OCR quality, LLM
        post-processing, and other Marker settings.  Pass ``MarkerConfig()``
        for sensible defaults or customise individual fields:

        .. code-block:: python

            from marker_backend import MarkerBackend, MarkerConfig

            backend = MarkerBackend(
                config=MarkerConfig(
                    force_ocr=True,          # scanned PDF
                    use_llm=True,            # better tables/math (needs Gemini key)
                    redo_inline_math=True,   # highest-quality LaTeX
                )
            )
    """

    def __init__(self, config: MarkerConfig | None = None) -> None:
        self.config = config or MarkerConfig()

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        if PdfConverter is None:
            raise ImportError(
                "marker-pdf is not installed. Run: pip install marker-pdf"
            )

        source = Path(source_pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")

        start_time = time.time()
        doc_id = _slugify(source.stem) or "document"
        print(f"Starting extraction with Marker for {source.name} (ID: {doc_id})")

        # ------------------------------------------------------------------
        # Build Marker converter with user-supplied config
        # ------------------------------------------------------------------
        print("Loading Marker models (may download on first run)...")
        marker_cfg_dict = self.config.to_marker_dict()
        if MarkerConfigParser is not None:
            marker_config = MarkerConfigParser(marker_cfg_dict).generate_config_dict()
        else:
            marker_config = marker_cfg_dict  # fallback: pass raw dict

        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config=marker_config,
        )

        print("Converting PDF with Marker...")
        rendered = converter(str(source))
        markdown_text, _, rendered_images = text_from_rendered(rendered)

        # ------------------------------------------------------------------
        # Images — build filename→id map FIRST so we can rewrite the markdown
        # ------------------------------------------------------------------
        print("Extracting images...")
        images: list[ExtractedImage] = []
        # Map both the raw key Marker uses and its basename to our img_NNN id.
        # Marker's markdown uses the basename as the image URL.
        filename_to_id: dict[str, str] = {}
        img_counter = 1
        for img_name, pil_image in (rendered_images or {}).items():
            img_id = f"img_{img_counter:03d}"
            filename_to_id[img_name] = img_id
            filename_to_id[Path(img_name).name] = img_id  # basename alias

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images.append(
                ExtractedImage(
                    id=img_id,
                    mime_type="image/png",
                    base64_data=base64_str,
                    page=None,  # Marker does not expose per-image page numbers
                    caption=img_name,
                )
            )
            img_counter += 1

        # ------------------------------------------------------------------
        # Replace Marker's native ![caption](filename) tags with [[img:NNN]]
        # tokens so the markdown is consistent with the SQLite images table.
        # This MUST happen before chunking and before equation annotation so
        # that the tokens end up in the right chunks.
        # ------------------------------------------------------------------
        if filename_to_id:
            markdown_text = _replace_marker_image_refs(markdown_text, filename_to_id)

        # ------------------------------------------------------------------
        # Tables — parse HTML tables embedded in the markdown output
        # ------------------------------------------------------------------
        print("Extracting tables...")
        tables: list[ExtractedTable] = []
        html_table_pattern = re.compile(
            r"(<table[\s\S]*?</table>)", re.IGNORECASE | re.DOTALL
        )
        tbl_counter = 1
        for match in html_table_pattern.finditer(markdown_text):
            tbl_id = f"tbl_{tbl_counter:03d}"
            tables.append(
                ExtractedTable(
                    id=tbl_id,
                    html_content=match.group(1),
                    page=None,
                    title=f"Table {tbl_counter}",
                )
            )
            tbl_counter += 1

        # ------------------------------------------------------------------
        # Equations — annotate LaTeX tokens in the markdown
        # ------------------------------------------------------------------
        print("Annotating equations...")
        markdown_text, equations = _annotate_equations(markdown_text)

        # ------------------------------------------------------------------
        # Append artifact reference section (summary at end of document)
        # ------------------------------------------------------------------
        ref_lines = ["", "## Artifact References", ""]
        for img in images:
            ref_lines.append(f"- [[img:{img.id}]] -> Attached Image {img.id}")
        for tbl in tables:
            ref_lines.append(f"- [[tbl:{tbl.id}]] -> Attached Table {tbl.id}")
        ref_lines.append("")
        markdown_text = markdown_text.rstrip() + "\n" + "\n".join(ref_lines)

        # ------------------------------------------------------------------
        # Chunking — image/table tokens are already inline in the markdown,
        # so chunker will naturally place them in the right chunks.
        # ------------------------------------------------------------------
        print("Chunking document...")
        chunker = MarkdownChunker()
        source_chunks = chunker.chunk(markdown_text)

        # ------------------------------------------------------------------
        # References & manifest
        # ------------------------------------------------------------------
        print("Building references...")
        references = build_artifact_references(
            ("image", "img", images),
            ("table", "tbl", tables),
            ("equation", "eq", equations),
        )

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
        _verify_references_in_markdown(markdown_text=markdown_text, manifest=manifest)

        print(f"[{time.time() - start_time:.2f}s] Extraction complete.")
        return ExtractionResult(
            source_chunks=source_chunks,
            manifest_json=manifest,
            image_count=len(images),
            table_count=len(tables),
            equation_count=len(equations),
            chunk_count=len(source_chunks),
        )
