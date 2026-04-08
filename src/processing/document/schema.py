from dataclasses import dataclass, field
from typing import Literal, TypeVar, Any


@dataclass
class ExtractedImage:
    """
    Metadata for an image/figure extracted from the PDF.

    Attributes:
        id: Unique artifact identifier (e.g., img_001)
        mime_type: The MIME type of the image (e.g., image/png)
        base64_data: The raw base64 encoded string of the image
        page: 1-indexed page number where the image was found
        caption: Extracted caption text associated with the figure (from ImageItem.caption)
        local_path: Path where the image file is stored (local now, cloud key later)
        contextualized_text: Succinct context situated within the document
    """
    id: str
    mime_type: str
    base64_data: str
    page: int | None = None
    caption: str = ""
    local_path: str | None = None
    contextualized_text: str | None = None


@dataclass
class ExtractedTable:
    """
    Metadata for a table extracted from the PDF.

    Attributes:
        id: Unique artifact identifier (e.g., tbl_001)
        content: Raw HTML string representing the table (preferred), fallback to markdown
        page: 1-indexed page number where the table was found
        caption: Extracted caption text from surrounding context
        title: Inferred table name (from heading above or generic fallback)
        col_count: Number of columns (derived from rows[0])
        row_count: Number of rows (derived from rows)
        contextualized_text: Succinct context situated within the document
    """
    id: str
    content: str
    page: int | None = None
    caption: str = ""
    title: str = ""
    col_count: int | None = None
    row_count: int | None = None
    contextualized_text: str | None = None


@dataclass
class ExtractedEquation:
    """
    A mathematical equation extracted from the document.

    LlamaParse does not produce a dedicated equation item type — equations appear
    inline in markdown as $...$ or block as $$...$$. This dataclass holds
    equations found by regex scanning the markdown_full string.

    Attributes:
        id: Unique artifact identifier (e.g., eq_001)
        latex_or_text: LaTeX source of the equation (content between $ delimiters)
        display_mode: 'block' for $$...$$, 'inline' for $...$
        page: 1-indexed page number (best-effort from surrounding context; often None)
        caption: Surrounding sentence/label if detectable
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
    A semantically coherent text chunk produced by the MarkdownSplitter.

    Attributes:
        id: Unique artifact identifier (e.g., chunk_0001)
        text: Raw text/markdown content of the chunk (may contain inline $equations$)
        meta_data: Arbitrary key-value metadata (chunk_index, splitter name, heading, etc.)
        contextualized_text: Succinct context situated within the document (LLM stage)
    """
    id: str
    text: str
    meta_data: dict[str, Any] = field(default_factory=dict)
    contextualized_text: str | None = None


@dataclass
class PaperMetadata:
    """
    Structured metadata parsed from the markdown of a research paper.

    All fields are best-effort and may be empty/None if the paper layout
    does not follow standard conventions. Parsing is done with regex against
    markdown_full — no LLM call needed for well-formatted papers.

    Attributes:
        title: Paper title (first H1 heading, or largest heading on page 1)
        authors: List of author name strings
        abstract: Full abstract text block
        keywords: List of keyword strings if present
        sections: Ordered mapping of section heading → section body text
        citations: Raw reference strings from the References section
        doi: DOI string if found in metadata block
        venue: Journal/conference name if found
        year: Publication year if found
    """
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    doi: str | None = None
    venue: str | None = None
    year: str | None = None


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
    run_id: str | None = None
    content_hash: str = ""
    paper_metadata: PaperMetadata | None = None
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

@dataclass
class ArtifactReference:
    type: str
    id: str
    markdown_token: str

@dataclass
class ExtractionManifest:   # For backends that emit ExtractionManifest (Not LlamaParse)
    doc_id: str
    source_pdf_path: str
    markdown_path: str
    images: list[ExtractedImage]
    tables: list[ExtractedTable]
    equations: list[ExtractedEquation]
    references: list[ArtifactReference]
