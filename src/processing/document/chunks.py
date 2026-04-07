from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .schema import ExtractedChunk

# Headers to split on, in order of precedence (H1 → H6)
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]

# Separators used when a section still exceeds max_chunk_size after header splitting
_MARKDOWN_SEPARATORS = [
    "\n\n",   # paragraph boundary
    "\n",     # line boundary
    " ",      # word boundary
    "",       # character boundary (last resort)
]


class MarkdownChunker:
    """
    A semantic chunker for Markdown text that uses LangChain's
    MarkdownHeaderTextSplitter for header-aware splitting, with a
    RecursiveCharacterTextSplitter fallback for oversized sections.
    """

    def __init__(self, max_chunk_size: int = 4000, chunk_overlap: int = 0):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
            strip_headers=False,  # keep headers in chunk text for context
        )
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_MARKDOWN_SEPARATORS,
        )

    def chunk(self, text: str) -> List[ExtractedChunk]:
        """
        Splits markdown text into ExtractedChunk objects.

        1. MarkdownHeaderTextSplitter divides the document at header boundaries
           and attaches header breadcrumbs as metadata.
        2. Any section that still exceeds max_chunk_size is further split
           by RecursiveCharacterTextSplitter.
        """
        # Step 1: Header-based split — returns list of LangChain Documents
        header_docs = self._header_splitter.split_text(text)

        # Step 2: Size-based fallback for oversized sections
        final_docs = self._char_splitter.split_documents(header_docs)

        chunks: List[ExtractedChunk] = []
        for doc in final_docs:
            header_stack = self._extract_header_stack(doc.metadata)
            contextualized = self._contextualize(header_stack, doc.page_content)
            chunks.append(
                ExtractedChunk(
                    text=doc.page_content,
                    contextualized_text=contextualized,
                    headings=header_stack,
                    captions=[],
                    page_numbers=[],  # filled downstream by the OCR backend
                    bboxes=[],
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_header_stack(self, metadata: dict) -> List[str]:
        """
        Builds an ordered breadcrumb list from LangChain header metadata.
        MarkdownHeaderTextSplitter stores headers as {"h1": "Title", "h2": "Sub", ...}.
        """
        keys = ["h1", "h2", "h3", "h4", "h5", "h6"]
        return [metadata[k] for k in keys if k in metadata]

    def _contextualize(self, headers: List[str], text: str) -> str:
        """Prepends a breadcrumb trail to the chunk text for better LLM grounding."""
        if not headers:
            return text
        prefix = " > ".join(headers) + "\n\n"
        return prefix + text
