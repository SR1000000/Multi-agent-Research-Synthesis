from .config import SemanticTextSplitterConfig
from .provider import TextChunkerProvider
from .semantic_text_splitter_chunker import SemanticTextSplitterChunker


def get_text_chunker(name: str) -> TextChunkerProvider:
    if name == "semantic":
        return SemanticTextSplitterChunker()
    raise ValueError(f"Unknown text chunker '{name}'. Available: ['semantic']")


__all__ = [
    "TextChunkerProvider",
    "SemanticTextSplitterConfig",
    "SemanticTextSplitterChunker",
    "get_text_chunker",
]
