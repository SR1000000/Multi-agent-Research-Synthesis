import time
import re
import os
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import pypdfium2 as pdfium
import psutil

from ..backend_base import OCRBackend
from ..schema import (
    ExtractedChunk,
    ExtractedImage,
    ExtractedTable,
    ExtractedEquation,
    ExtractionManifest,
    ExtractionResult,
    ArtifactReference,
)
from .._common import _slugify, _verify_references_in_markdown, build_artifact_references
from ..chunks import MarkdownChunker

class LightOnOCRBackend(OCRBackend):
    """
    OCR backend powered by LightOnOCR-2-1B-bbox.
    """

    def __init__(self, model_id: str = "lightonai/LightOnOCR-2-1B-bbox"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        print(f"Loading LightOnOCR model {model_id} on {self.device}...")
        self.processor = LightOnOcrProcessor.from_pretrained(model_id)
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=self.dtype
        ).to(self.device)

        if self.device == "cpu":
            num_threads = psutil.cpu_count(logical=False)
            print(f"Optimizing for CPU: setting torch.set_num_threads({num_threads})")
            torch.set_num_threads(num_threads)
            
        self.chunker = MarkdownChunker()

    def extract(self, source_pdf_path: str, max_pages: int | None = None) -> ExtractionResult:
        source = Path(source_pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")

        start_time = time.time()
        doc_id = _slugify(source.stem) or "document"
        print(f"Starting LightOnOCR extraction for {source.name} (ID: {doc_id})")

        # 1. Render PDF pages to images
        page_images: List[Image.Image] = []
        with pdfium.PdfDocument(source_pdf_path) as pdf:
            for i in range(len(pdf)):
                if max_pages is not None:
                    if i >= max_pages:
                        break
                
                page = pdf[i]
                # Render at 200 DPI (scale ~2.77)
                bitmap = page.render(scale=2.77)
                pil_image = bitmap.to_pil()
                page_images.append(pil_image)
                
                bitmap.close()
                page.close()
        
        num_pages = len(page_images)
        print(f"Rendered {num_pages} pages.")

        all_markdown = ""
        extracted_images: List[ExtractedImage] = []
        img_counter = 1

        # 2. Process each page
        for page_idx, pil_image in enumerate(page_images, start=1):
            print(f"Processing page {page_idx}/{num_pages}...")
            
            # Simple prompt for the model
            conversation = [{"role": "user", "content": [{"type": "image", "image": pil_image}]}]
            inputs = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=4096)
                generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
                output_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            # 3. Parse BBoxes and extract images
            page_markdown, page_images_metadata = self._process_bboxes(
                output_text, 
                pil_image, 
                page_idx, 
                img_counter
            )
            extracted_images.extend(page_images_metadata)
            img_counter += len(page_images_metadata)
            
            all_markdown += f"\n\n<!-- Page {page_idx} -->\n" + page_markdown

        # 4. Chunking
        print("Chunking combined markdown...")
        source_chunks = self.chunker.chunk(all_markdown)
        
        # 5. Build Reference Manifest
        references = build_artifact_references(
            ("image", "img", extracted_images),
            # Tables and Equations are not handled as separate artifacts by LightOnOCR
            # but are part of the markdown text.
        )

        manifest = ExtractionManifest(
            doc_id=doc_id,
            source_pdf_path=str(source),
            markdown_path="",
            images=extracted_images,
            tables=[],
            equations=[],
            references=references
        )

        print(f"[{time.time() - start_time:.2f}s] LightOnOCR extraction complete.")
        return ExtractionResult(
            source_chunks=source_chunks,
            manifest_json=manifest,
            image_count=len(extracted_images),
            table_count=0,
            equation_count=0,
            chunk_count=len(source_chunks)
        )

    def _process_bboxes(self, text: str, page_image: Image.Image, page_no: int, start_id: int) -> Tuple[str, List[ExtractedImage]]:
        """
        Parses ![image](...)x1,y1,x2,y2 and crops the image.
        Returns (modified_markdown, list_of_ExtractedImage).
        """
        metadata: List[ExtractedImage] = []
        modified_text = text
        
        # Pattern for ![image](...)x1,y1,x2,y2
        # Normalization is usually [0, 1000]
        pattern = re.compile(r"!\[image\]\(.*?\)\[(\d+),(\d+),(\d+),(\d+)\]")
        
        matches = list(pattern.finditer(text))
        offset = 0
        
        w, h = page_image.size
        
        for i, match in enumerate(matches):
            x1_norm, y1_norm, x2_norm, y2_norm = map(int, match.groups())
            
            # Map back to image pixels
            x1 = x1_norm * w / 1000
            y1 = y1_norm * h / 1000
            x2 = x2_norm * w / 1000
            y2 = y2_norm * h / 1000
            
            # Crop
            crop = page_image.crop((x1, y1, x2, y2))
            
            # Save to bytes
            buf = BytesIO()
            crop.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            base64_str = base64.b64encode(img_bytes).decode("utf-8")
            
            img_id = f"img_{start_id + i:03d}"
            token = f"[[img:{img_id}]]"
            
            metadata.append(ExtractedImage(
                id=img_id,
                mime_type="image/png",
                base64_data=base64_str,
                page=page_no,
                caption=f"Image {img_id} from page {page_no}"
            ))
            
            # Replace tag in text with token
            tag_start = match.start() + offset
            tag_end = match.end() + offset
            token_str = f"\n\n{token}\n\n"
            modified_text = modified_text[:tag_start] + token_str + modified_text[tag_end:]
            offset += len(token_str) - (tag_end - tag_start)
            
        return modified_text, metadata
