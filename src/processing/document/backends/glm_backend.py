import time
from pathlib import Path
from typing import List
import io

from ..backend_base import OCRBackend
from ..schema import (
    ExtractedImage,
    ExtractionManifest,
    ExtractionResult,
)
from .._common import _slugify, build_artifact_references
from ..chunks import MarkdownChunker

class GLMOCRBackend(OCRBackend):
    """OCR backend powered by GLM-OCR (local HuggingFace inference).
    
    Uses `transformers` to load `zai-org/GLM-OCR` for purely local
    optical character recognition without an API key. 
    Requires: pip install transformers>=5.0.0 accelerate pypdfium2 pillow torchvision
    """
    
    def __init__(self, model_checkpoint: str = "zai-org/GLM-OCR", device: str = "cuda"):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as exc:
            raise ImportError(
                "GLMOCRBackend requires HuggingFace libraries. "
                "Ensure local environment has `transformers`, `torch`, `accelerate`."
            ) from exc
            
        self.chunker = MarkdownChunker()
        self.model_checkpoint = model_checkpoint
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"Loading GLM-OCR model '{model_checkpoint}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint, 
            trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_checkpoint,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        ).eval()
        
        print("Initialized GLM-OCR local backend.")

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        import torch
        import pypdfium2 as pdfium
        
        source = Path(source_pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")
            
        start_time = time.time()
        doc_id = _slugify(source.stem) or "document"
        print(f"Starting GLM-OCR local extraction for {source.name} (ID: {doc_id})")
        
        # 1. PDF -> list[PIL.Image] via pypdfium2
        print("Loading PDF pages via pypdfium2...")
        
        all_markdown = ""
        extracted_images: List[ExtractedImage] = []
        
        with pdfium.PdfDocument(str(source)) as pdf:
            num_pages = len(pdf)
            print(f"Loaded {num_pages} page(s).")
            
            # 2. Per-page inference
            for page_idx in range(num_pages):
                print(f"Processing page {page_idx + 1}/{num_pages}...")
                
                # Render page to PIL image
                page = pdf[page_idx]
                # using scale=2.0 for better OCR resolution
                bitmap = page.render(scale=2.0)
                pil_image = bitmap.to_pil()
                
                # We can close pypdfium2 objects right after we get the PIL image
                bitmap.close()
                page.close()
                
                # 3. Model Inference
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Recognize the document and output it in markdown format."}]}]
                query = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )
                inputs = self.processor(
                    text=query,
                    images=[pil_image],
                    return_tensors="pt"
                ).to(self.model.device)
                inputs.pop("token_type_ids", None)
                
                gen_kwargs = {"max_new_tokens": 4096, "do_sample": False}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    page_markdown = self.processor.decode(outputs[0], skip_special_tokens=True)
                    
                all_markdown += f"\n\n<!-- Page {page_idx + 1} -->\n" + page_markdown
                
        print("Chunking combined markdown...")
        source_chunks = self.chunker.chunk(all_markdown)
        
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
        print(f"[{elapsed:.2f}s] GLM-OCR local extraction complete.")
        return ExtractionResult(
            source_chunks=source_chunks,
            manifest_json=manifest,
            image_count=len(extracted_images),
            table_count=0,
            equation_count=0,
            chunk_count=len(source_chunks),
        )
