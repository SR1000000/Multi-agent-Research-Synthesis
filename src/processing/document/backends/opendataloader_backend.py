import base64
import json
import os
import tempfile
import time
from pathlib import Path

try:
    import opendataloader_pdf
except ImportError:
    opendataloader_pdf = None

from .._common import _slugify, _verify_references_in_markdown, build_artifact_references
from ..backend_base import OCRBackend
from ..chunks import MarkdownChunker
from ..schema import (
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionManifest,
    ExtractionResult,
)

class OpendataloaderBackend(OCRBackend):
    """OCR backend powered by OpenDataLoader PDF.
    
    Implements image extraction, formula (LaTeX) processing, and chunking.
    """

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        if opendataloader_pdf is None:
            raise ImportError("opendataloader_pdf is not installed.")
            
        source = Path(source_pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")
            
        start_time = time.time()
        doc_id = _slugify(source.stem) or "document"
        print(f"Starting extraction with OpenDataLoader for {source.name} (ID: {doc_id})")

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                opendataloader_pdf.convert(
                    input_path=[source_pdf_path],
                    output_dir=temp_dir,
                    format="markdown,json",
                    hybrid="docling-fast", 
                    hybrid_mode="full",
                    hybrid_timeout="600000",
                    hybrid_fallback=True
                )
            except Exception as e:
                raise RuntimeError(f"OpenDataLoader conversion failed: {e}")
            
            md_path = os.path.join(temp_dir, f"{source.stem}.md")
            json_path = os.path.join(temp_dir, f"{source.stem}.json")
            
            if not os.path.exists(md_path) or not os.path.exists(json_path):
                raise RuntimeError("Failed to generate markdown or json output from OpenDataLoader.")
                
            with open(md_path, "r", encoding="utf-8") as f:
                markdown_text = f.read()
                
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    elements = json.load(f)
                except json.JSONDecodeError:
                    elements = []

        print("Chunking document...")
        chunker = MarkdownChunker()
        source_chunks = chunker.chunk(markdown_text)

        images = []
        tables = []
        equations = []

        print("Extracting metadata...")
        img_counter = 1
        tbl_counter = 1
        eq_counter = 1
        
        for item in elements:
            item_type = item.get("type", "")
            page = item.get("page number")
            
            if item_type == "picture":
                img_id = f"img_{img_counter:03d}"
                caption = item.get("description", "")
                images.append(ExtractedImage(
                    id=img_id,
                    mime_type="image/png",
                    base64_data="", 
                    page=page,
                    caption=caption
                ))
                img_counter += 1

            elif item_type == "table":
                tbl_id = f"tbl_{tbl_counter:03d}"
                html_content = item.get("html", "")  
                title = f"Table {tbl_counter}"
                tables.append(ExtractedTable(
                    id=tbl_id,
                    html_content=html_content,
                    page=page,
                    title=title
                ))
                tbl_counter += 1
                
            elif item_type == "formula":
                eq_id = f"eq_{eq_counter:03d}"
                content = item.get("content", "")
                equations.append(ExtractedEquation(
                    id=eq_id,
                    latex_or_text=content,
                    display_mode="block",
                    page=page,
                    markdown_anchor=f"[[eq:{eq_id}]]"
                ))
                eq_counter += 1

        print("Annotating equations in markdown...")
        # OpenDataLoader may or may not already embed "[[eq:eq_001]]". If not, we just append to the reference list.
        # But we must satisfy the references check. Let's just append references for all to the bottom.
        
        lines = ["", "## Artifact References", ""]
        for img in images:
            lines.append(f"- [[img:{img.id}]] -> Attached Image {img.id}")
            if source_chunks:
                # Add token to the last chunk, or page-matching chunk
                source_chunks[-1].text += f"\n\n[[img:{img.id}]]"
                source_chunks[-1].contextualized_text += f"\n\n[[img:{img.id}]]"
                
        for tbl in tables:
            lines.append(f"- [[tbl:{tbl.id}]] -> Attached Table {tbl.id}")
            if source_chunks:
                source_chunks[-1].text += f"\n\n[[tbl:{tbl.id}]]"
                source_chunks[-1].contextualized_text += f"\n\n[[tbl:{tbl.id}]]"
                
        for eq in equations:
            lines.append(f"- {eq.markdown_anchor} -> Formula {eq.id}")
            # If not in markdown natively, appending reference acts as a fallback token.
            
        lines.append("")
        markdown_text = markdown_text.rstrip() + "\n" + "\n".join(lines)

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
