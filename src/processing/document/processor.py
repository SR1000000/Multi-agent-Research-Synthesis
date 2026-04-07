import hashlib
from dataclasses import asdict
from typing import Any

from src.logging.logger import AgentLogger
from src.processing.chunker import TextChunkerProvider

from .backend_base import OCRBackend
from .backends import DoclingBackend, LightOnOCRBackend, LlamaParseBackend, MarkerBackend, ChandraOCRBackend, GLMOCRBackend
from src.processing.embedder.base import TextEmbedder

from .schema import ExtractedChunk, ExtractionResult, Contextualizer

BACKEND_REGISTRY: dict[str, type[OCRBackend]] = {
    "llama_parse": LlamaParseBackend,
    "docling": DoclingBackend,
    "lighton": LightOnOCRBackend,
}
if DoclingBackend:
    BACKEND_REGISTRY["docling"] = DoclingBackend
if ChandraOCRBackend:
    BACKEND_REGISTRY["chandra"] = ChandraOCRBackend
if GLMOCRBackend:
    BACKEND_REGISTRY["glm"] = GLMOCRBackend
if MarkerBackend:
    BACKEND_REGISTRY["marker"] = MarkerBackend


def get_ocr_backend(
    name: str = "llama_parse",
    text_chunker: TextChunkerProvider | None = None,
    logger: AgentLogger | None = None,
) -> OCRBackend:
    """Instantiate an OCR backend by name."""
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown OCR backend '{name}'. "
            f"Available: {list(BACKEND_REGISTRY.keys())}"
        )
    if cls is LlamaParseBackend:
        return cls(text_chunker=text_chunker, logger=logger)
    return cls()


def _chunk_text_for_embedding(chunk: ExtractedChunk) -> str:
    if chunk.contextualized_text and chunk.contextualized_text.strip():
        return chunk.contextualized_text.strip()
    return chunk.text


class DocProcessor:
    def __init__(
        self,
        backend: str | OCRBackend = "llama_parse",
        text_chunker: TextChunkerProvider | None = None,
        db: Any = None,
        contextualizer: Contextualizer | None = None,
        embedder: TextEmbedder | None = None,
        logger: AgentLogger | None = None,
    ):
        """
        Create a document processor with the given OCR backend and optional pipeline stages.
        Pass a non-None embedder to run the embedding step on chunks.
        """
        self._logger = logger or AgentLogger()
        if isinstance(backend, str):
            self.backend = get_ocr_backend(backend, text_chunker=text_chunker, logger=self._logger)
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

            if self._embedder is not None:
                print(f"[DocProcessor] Embedding chunks...")
                try:
                    texts = [_chunk_text_for_embedding(c) for c in result.source_chunks]
                    print(f"[DocProcessor] Embedding input chunks={len(texts)}")
                    result.chunk_embeddings = self._embedder.embed_queries(texts)
                    result.chunk_embedding_sources = texts
                    emb_count = len(result.chunk_embeddings) if result.chunk_embeddings is not None else 0
                    emb_dim = len(result.chunk_embeddings[0]) if emb_count else 0
                    print(f"[DocProcessor] Embeddings generated count={emb_count} dim={emb_dim}")
                except Exception:
                    result.chunk_embeddings = None
                    result.chunk_embedding_sources = None
                    import traceback
                    print("[DocProcessor] Embedding failed; embeddings set to None")
                    traceback.print_exc()
            else:
                print("[DocProcessor] Embedder is None; skipping embedding step")

            # Persist to database
            if self._db:
                dump_path = self._logger.dump_json_artifact(
                    file_name="last_extraction_result.json",
                    payload=asdict(result),
                    run_id=result.run_id,
                )
                if dump_path:
                    self._logger.log(f"[DocProcessor] Wrote debug ExtractionResult to {dump_path}")
                else:
                    self._logger.log("[DocProcessor] Failed to write debug ExtractionResult JSON", level="warning")

                print(f"[DocProcessor] Storing to database...")
                self._db.save_document(result)

            return result

        except Exception as e:
            print(f"[DocProcessor] Failed to ingest {source_path}: {e}")
            return None
