from dataclasses import dataclass, field
from typing import Literal, TypeVar, Any

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
        contextualized_text: Succinct context situated within the document
    """
    id: str
    mime_type: str
    base64_data: str
    page: int | None = None
    caption: str = ""
    contextualized_text: str | None = None

@dataclass
class ExtractedTable:
    """
    Metadata for a table extracted from the PDF.

    Attributes:
        id: Unique artifact identifier (e.g., tbl_001)
        html_content: Raw HTML string representing the table
        page: 1-indexed page number where the table was found
        title: Extracted table title or inferred name
        contextualized_text: Succinct context situated within the document
        col_count: Number of columns
        row_count: Number of rows
    """
    id: str
    content: str
    page: int | None = None
    title: str = ""
    contextualized_text: str | None = None
    col_count: int | None = None
    row_count: int | None = None

@dataclass
class ExtractedEquation:
    """
    Metadata for a mathematical equation enriched by Docling.

    Attributes:
        id: Unique artifact identifier (e.g., eq_001)
        latex_or_text: LaTeX source or raw text of the equation
        display_mode: 'block' for standalone equations, 'inline' for text-embedded ones
        page: 1-indexed page number
        contextualized_text: Succinct context situated within the document
    """
    id: str
    latex_or_text: str
    display_mode: Literal["inline", "block"]
    page: int | None = None
    caption: str = ""
    contextualized_text: str | None = None

@dataclass
class ExtractedChunk:
    """
    A semantically coherent text chunk produced by the HybridChunker.

    Attributes:
        id: Unique artifact identifier (e.g., chunk_0001)
        text: Raw text content of the chunk
        headings: List of all ancestor headers from the document root
        captions: Captions from figures or tables semantically linked to this chunk
        page_numbers: Sorted list of all page numbers spanned by this chunk's doc_items
        contextualized_text: Succinct context situated within the document (reserved for LLM stage)
    """
    id: str
    text: str
    meta_data: dict[str, Any] = field(default_factory=dict)
    contextualized_text: str | None = None

@dataclass
class ExtractionResult:
    """The final output of the document processing pipeline."""
    doc_id: str
    source_path: str
    source_chunks: list[ExtractedChunk]
    images: list[ExtractedImage]
    tables: list[ExtractedTable]
    equations: list[ExtractedEquation]
    markdown: str | None = None
    page_count: int = 0
    schema: str | None = None
    content_hash: str = ""
    chunk_embeddings: list[list[float]] | None = None
    chunk_embedding_sources: list[str] | None = None

    @property
    def chunk_count(self) -> int:
        return len(self.source_chunks)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def equation_count(self) -> int:
        return len(self.equations)

T = TypeVar("T", ExtractedImage, ExtractedTable)

class Contextualizer:
    """Placeholder for an LLM-based contextualizer that grounds chunks with document context."""
    def contextualize(self, result: ExtractionResult) -> ExtractionResult:
        # TODO: Implement chunk contextualization
        return result