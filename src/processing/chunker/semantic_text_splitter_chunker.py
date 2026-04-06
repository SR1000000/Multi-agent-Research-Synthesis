from __future__ import annotations

from semantic_text_splitter import MarkdownSplitter, TextSplitter

from .config import SemanticTextSplitterConfig
from .provider import TextChunkerProvider


class SemanticTextSplitterChunker(TextChunkerProvider):
    """Chunker backed by semantic-text-splitter."""

    def __init__(self, config: SemanticTextSplitterConfig | None = None) -> None:
        self._config = config or SemanticTextSplitterConfig()
        self._markdown_splitter = MarkdownSplitter(self._config.markdown_capacity)
        self._text_splitter = TextSplitter(self._config.text_capacity)

    def chunk_markdown(self, text: str) -> list[str]:
        return self._markdown_splitter.chunks(text)

    def chunk_text(self, text: str) -> list[str]:
        return self._text_splitter.chunks(text)
