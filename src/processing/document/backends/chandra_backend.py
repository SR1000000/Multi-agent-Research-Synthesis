import time
import base64
from io import BytesIO
from pathlib import Path
from typing import List

from ..backend_base import OCRBackend
from ..schema import (
    ExtractedImage,
    ExtractionManifest,
    ExtractionResult,
)
from .._common import _slugify, build_artifact_references
from ..chunks import MarkdownChunker


class ChandraOCRBackend(OCRBackend):
    """OCR backend powered by Chandra OCR 2 (local HuggingFace inference).

    Uses ``chandra-ocr[hf]`` (``pip install chandra-ocr[hf]``) with the
    ``datalab-to/chandra-ocr-2`` model checkpoint.

    Features
    --------
    - PDF → PIL image conversion is delegated to Chandra's own
      ``chandra.input.load_file`` helper (backed by ``pypdfium2``).
    - Per-page images are processed through ``InferenceManager(method="hf")``
      which returns a ``BatchOutputItem`` per page.
    - Images/figures are extracted from ``BatchOutputItem.images`` (a
      ``dict[str, PIL.Image]`` of bbox-cropped regions) and recorded as
      ``ExtractedImage`` objects with ``[[img:NNN]]`` annotation tokens.
    - Tables and equations are preserved as formatted markdown text; no
      separate ``ExtractedTable`` / ``ExtractedEquation`` objects are produced.
    """

    def __init__(self, model_checkpoint: str = "datalab-to/chandra-ocr-2"):
        try:
            from chandra.model import InferenceManager
        except ImportError as exc:
            raise ImportError(
                "ChandraOCRBackend requires 'chandra-ocr[hf]'. "
                "Install with: pip install chandra-ocr[hf]"
            ) from exc

        self.model_checkpoint = model_checkpoint
        print(f"Loading Chandra OCR model '{model_checkpoint}' (HuggingFace local)...")
        self._manager = InferenceManager(method="hf")
        self._chunker = MarkdownChunker()

    # ------------------------------------------------------------------
    # OCRBackend interface
    # ------------------------------------------------------------------

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        from chandra.input import load_file
        from chandra.model.schema import BatchInputItem

        source = Path(source_pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")

        start_time = time.time()
        doc_id = _slugify(source.stem) or "document"
        print(f"Starting Chandra extraction for {source.name} (ID: {doc_id})")

        # 1. PDF → list[PIL.Image] via Chandra's own helper
        print("Loading PDF pages via Chandra input loader...")
        page_images = load_file(str(source), config={})
        num_pages = len(page_images)
        print(f"Loaded {num_pages} page(s).")

        all_markdown = ""
        extracted_images: List[ExtractedImage] = []
        img_counter = 1

        # 2. Per-page inference
        for page_idx, pil_image in enumerate(page_images, start=1):
            print(f"Processing page {page_idx}/{num_pages}...")

            batch_item = BatchInputItem(image=pil_image, prompt_type="ocr_layout")
            result = self._manager.generate([batch_item])[0]

            # 3. Collect extracted images for this page
            page_img_ids: List[str] = []
            for _fname, crop in result.images.items():
                img_id = f"img_{img_counter:03d}"
                img_counter += 1

                buf = BytesIO()
                crop.save(buf, format="PNG")
                base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

                extracted_images.append(
                    ExtractedImage(
                        id=img_id,
                        mime_type="image/png",
                        base64_data=base64_str,
                        page=page_idx,
                        caption=f"Image {img_id} from page {page_idx}",
                    )
                )
                page_img_ids.append(img_id)

            # 4. Build annotation tokens and append to page markdown
            page_markdown = result.markdown
            if page_img_ids:
                tokens = "\n\n".join(f"[[img:{iid}]]" for iid in page_img_ids)
                page_markdown = page_markdown.rstrip() + f"\n\n{tokens}"

            all_markdown += f"\n\n<!-- Page {page_idx} -->\n" + page_markdown

        # 5. Chunk combined markdown
        print("Chunking combined markdown...")
        source_chunks = self._chunker.chunk(all_markdown)

        # 6. Build manifest
        references = build_artifact_references(
            ("image", "img", extracted_images),
        )
        manifest = ExtractionManifest(
            doc_id=doc_id,
            source_pdf_path=str(source),
            markdown_path="",
            images=extracted_images,
            tables=[],
            equations=[],
            references=references,
        )

        elapsed = time.time() - start_time
        print(f"[{elapsed:.2f}s] Chandra extraction complete.")
        return ExtractionResult(
            source_chunks=source_chunks,
            manifest_json=manifest,
            image_count=len(extracted_images),
            table_count=0,
            equation_count=0,
            chunk_count=len(source_chunks),
        )
