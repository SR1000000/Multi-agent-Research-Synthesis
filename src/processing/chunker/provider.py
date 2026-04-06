from __future__ import annotations

from abc import ABC, abstractmethod


class TextChunkerProvider(ABC):
    """Interface for text chunking providers."""

    @abstractmethod
    def chunk_markdown(self, text: str) -> list[str]:
        """Split markdown text into semantic chunks."""
        pass

    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        """Split plain text into semantic chunks."""
        pass
