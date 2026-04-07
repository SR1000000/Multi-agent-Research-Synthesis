from dataclasses import dataclass, field
from typing import Literal, TypeVar

@dataclass
class ExtractedImage:
    """
    Metadata for an image extracted from the PDF.
    
    Attributes:
        id: Unique artifact identifier (e.g., img_001)
        mime_type: The MIME type of the image (e.g., image/png)
        base64_data: The raw base64 encoded string of the image
        page: 1-indexed page number where the image was found
        caption: Extracted caption text associated with the picture
    """
    id: str
    mime_type: str
    base64_data: str
    page: int | None
    caption: str

@dataclass
class ExtractedTable:
    """
    Metadata for a table extracted from the PDF.
    
    Attributes:
        id: Unique artifact identifier (e.g., tbl_001)
        html_content: Raw HTML string representing the table
        page: 1-indexed page number where the table was found
        title: Extracted table title or inferred name
    """
    id: str
    html_content: str
    page: int | None
    title: str

@dataclass
class ExtractedEquation:
    """
    Metadata for a mathematical equation enriched by Docling.
    
    Attributes:
        id: Unique artifact identifier (e.g., eq_001)
        latex_or_text: LaTeX source or raw text of the equation
        display_mode: 'block' for standalone equations, 'inline' for text-embedded ones
        page: 1-indexed page number
        markdown_anchor: The unique token (e.g. [[eq:eq_001]]) used in the source markdown
    """
    id: str
    latex_or_text: str
    display_mode: Literal["block", "inline"]
    page: int | None
    markdown_anchor: str

@dataclass
class ExtractedChunk:
    """
    A semantically coherent text chunk produced by the HybridChunker.
    
    Attributes:
        text: Raw text content of the chunk
        contextualized_text: Text with heading breadcrumbs prepended for better LLM grounding
        headings: List of all ancestor headers from the document root
        captions: List of captions from figures or tables semantically linked to this chunk
        page_numbers: Sorted list of all page numbers spanned by this chunk's doc_items
        bboxes: List of (l, t, r, b, page_no) tuples, one per provenance entry across all doc_items
    """
    text: str
    contextualized_text: str
    headings: list[str]
    captions: list[str]
    page_numbers: list[int] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float, int]] = field(default_factory=list)

@dataclass
class ArtifactReference:
    """
    A mapping between a markdown token and a concrete extracted artifact.
    """
    token: str
    item_id: str
    kind: Literal["image", "table", "equation"]

@dataclass
class ExtractionManifest:
    """
    The master index for a processed document's extracted assets.
    """
    doc_id: str
    source_pdf_path: str
    markdown_path: str
    images: list[ExtractedImage]
    tables: list[ExtractedTable]
    equations: list[ExtractedEquation]
    references: list[ArtifactReference]

@dataclass
class ExtractionResult:
    """
    The final output of the document processing pipeline.
    """
    source_chunks: list[ExtractedChunk]
    manifest_json: ExtractionManifest
    image_count: int
    table_count: int
    equation_count: int
    chunk_count: int

T = TypeVar("T", ExtractedImage, ExtractedTable)
