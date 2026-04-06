import hashlib
from pathlib import Path
from typing import Any

from src.processing.chunker import TextChunkerProvider

from .backend_base import OCRBackend
from .backends import DoclingBackend, LightOnOCRBackend, LlamaParseBackend
from .schema import ExtractionResult, Contextualizer, Embedder

BACKEND_REGISTRY: dict[str, type[OCRBackend]] = {
    "llama_parse": LlamaParseBackend,
    "docling": DoclingBackend,
    "lighton": LightOnOCRBackend,
}


def get_ocr_backend(
    name: str = "llama_parse",
    text_chunker: TextChunkerProvider | None = None,
) -> OCRBackend:
    """Instantiate an OCR backend by name."""
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown OCR backend '{name}'. "
            f"Available: {list(BACKEND_REGISTRY.keys())}"
        )
    if cls is LlamaParseBackend:
        return cls(text_chunker=text_chunker)
    return cls()


class DocProcessor:
    def __init__(
        self,
        backend: str | OCRBackend = "llama_parse",
        text_chunker: TextChunkerProvider | None = None,
        db: Any = None,
        contextualizer: Contextualizer | None = None,
        embedder: Embedder | None = None
    ):
        """
        Create a document processor with the given OCR backend and optional pipeline stages.
        """
        if isinstance(backend, str):
            self.backend = get_ocr_backend(backend, text_chunker=text_chunker)
        else:
            self.backend = backend
        self._db = db
        self._contextualizer = contextualizer
        self._embedder = embedder

    def process_document(self, source_path: str) -> ExtractionResult | None:
        """
        Full pipeline: check cache → extract → contextualize → embed → store.
        Returns ExtractionResult on success, None on failure.
        """
        try:
            content_hash = ""
            
            with open(source_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()

            # Skip re-ingestion if document already exists
            if self._db and self._db.document_exists(content_hash):
                cached = self._db.load_document_by_hash(content_hash)
                if cached:
                    print(f"[DocProcessor] Cache hit for {source_path}")
                    return cached

            # Parse document
            print(f"[DocProcessor] Extracting {source_path}...")
            result = self.backend.extract(source_path)
            result.content_hash = content_hash

            # Contextualize each chunk
            if self._contextualizer:
                print(f"[DocProcessor] Contextualizing chunks...")
                result = self._contextualizer.contextualize(result)

            # Create embeddings on document chunks for retrieval
            embeddings = None
            if self._embedder:
                print(f"[DocProcessor] Embedding chunks...")
                embeddings = self._embedder.embed_extraction_result(result)

            # Persist to database
            if self._db:
                print(f"[DocProcessor] Storing to database...")
                self._db.save_document(result)

            return result

        except Exception as e:
            print(f"[DocProcessor] Failed to ingest {source_path}: {e}")
            return None