import hashlib
from dataclasses import asdict
from typing import Any

from src.logging.logger import AgentLogger
from src.processing.chunker import TextChunkerProvider

from .backend_base import OCRBackend
from .backends import DoclingBackend, LightOnOCRBackend, LlamaParseBackend, MarkerBackend, ChandraOCRBackend, GLMOCRBackend
from src.processing.embedder.base import TextEmbedder

from .schema import ExtractedChunk, ExtractionResult
from ..context.contextualizer import Contextualizer

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
    object_store: Any = None,
) -> OCRBackend:
    """Instantiate an OCR backend by name."""
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown OCR backend '{name}'. "
            f"Available: {list(BACKEND_REGISTRY.keys())}"
        )
    if cls is LlamaParseBackend:
        return cls(text_chunker=text_chunker, logger=logger, object_store=object_store)
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
        object_store: Any = None,
    ):
        """
        Create a document processor with the given OCR backend and optional pipeline stages.
        Pass a non-None embedder to run the embedding step on chunks.
        """
        self._logger = logger or AgentLogger()
        if isinstance(backend, str):
            self.backend = get_ocr_backend(backend, text_chunker=text_chunker, logger=self._logger, object_store=object_store)
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

            self._logger.log(f"[DocProcessor] Start ingest path={source_path}", level="info")
            with open(source_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()
            self._logger.log(f"[DocProcessor] Content hash computed prefix={content_hash[:16]}…", level="info")

            # Skip re-ingestion if document already exists
            if self._db and self._db.document_exists(content_hash):
                self._logger.log("[DocProcessor] Document exists in DB; loading cached record…", level="info")
                cached = self._db.load_document_by_hash(content_hash)
                if cached:
                    self._logger.log(f"[DocProcessor] Cache hit for {source_path} doc_id={cached.doc_id}", level="info")
                    # Check if re-contextualization is needed. There could be docs processed before without contextualize run
                    chunks_todo, artifacts_todo = self._get_uncontextualized(cached)
                    self._logger.log(
                        f"[DocProcessor] Contextualization backlog chunks={len(chunks_todo)} "
                        f"artifacts={len(artifacts_todo)} contextualizer={'on' if self._contextualizer else 'off'}",
                        level="info",
                    )
                    if self._contextualizer and (chunks_todo or artifacts_todo):
                        self._logger.log("[DocProcessor] Re-contextualizing cached document…", level="info")
                        cached = self._contextualizer.contextualize(cached)
                        self._logger.log("[DocProcessor] Re-contextualization finished", level="info")
                        # Re-embed with updated contexts. Docs embedded without contextualized run uses their text content for embedding. 
                        # We want to embed the contextualized text content now
                        if self._embedder:
                            self._logger.log("[DocProcessor] Re-embedding cached document…", level="info")
                            cached = self._reembed(cached)
                            self._logger.log("[DocProcessor] Re-embed finished", level="info")
                        # Update stored document
                        self._logger.log("[DocProcessor] Saving updated cached document to DB…", level="info")
                        self._db.save_document(cached)
                    else:
                        self._logger.log("[DocProcessor] Cache hit — no contextualization work needed", level="info")
                    return cached

            # Parse document
            self._logger.log(
                f"[DocProcessor] Extracting with backend={type(self.backend).__name__} path={source_path}…",
                level="info",
            )
            result = self.backend.extract(source_path)
            result.content_hash = content_hash
            self._logger.log(
                f"[DocProcessor] Extract done doc_id={result.doc_id} chunks={result.chunk_count} "
                f"images={result.image_count} tables={result.table_count} equations={result.equation_count} "
                f"pages={result.page_count}",
                level="info",
            )

            # Contextualize each chunk
            if self._contextualizer:
                chunks_todo, artifacts_todo = self._get_uncontextualized(result)
                self._logger.log(
                    f"[DocProcessor] Contextualizing backlog chunks={len(chunks_todo)} artifacts={len(artifacts_todo)}…",
                    level="info",
                )
                result = self._contextualizer.contextualize(result)
                self._logger.log("[DocProcessor] Contextualization finished", level="info")

            if self._embedder is not None:
                self._logger.log("[DocProcessor] Embedding chunks…", level="info")
                try:
                    texts = [_chunk_text_for_embedding(c) for c in result.source_chunks]
                    self._logger.log(f"[DocProcessor] Embedding input chunks={len(texts)}", level="info")
                    result.chunk_embeddings = self._embedder.embed_queries(texts)
                    result.chunk_embedding_sources = texts
                    emb_count = len(result.chunk_embeddings) if result.chunk_embeddings is not None else 0
                    emb_dim = len(result.chunk_embeddings[0]) if emb_count else 0
                    self._logger.log(f"[DocProcessor] Embeddings generated count={emb_count} dim={emb_dim}", level="info")
                except Exception:
                    result.chunk_embeddings = None
                    result.chunk_embedding_sources = None
                    import traceback
                    self._logger.log("[DocProcessor] Embedding failed; embeddings set to None", level="warning")
                    traceback.print_exc()
            else:
                self._logger.log("[DocProcessor] Embedder is None; skipping embedding step", level="info")

            # Final sanity check to ensure chunks align with media databases
            from ._common import verify_extraction_result
            self._logger.log("[DocProcessor] Verifying extraction result integrity…", level="info")
            verify_extraction_result(result, logger=self._logger)

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

                self._logger.log("[DocProcessor] Storing to database…", level="info")
                self._db.save_document(result)
                self._logger.log("[DocProcessor] Persist complete", level="info")

            self._logger.log(f"[DocProcessor] Ingest complete doc_id={result.doc_id}", level="info")
            return result

        except Exception as e:
            self._logger.log(f"[DocProcessor] Failed to ingest {source_path}: {e}", level="info")
            return None

    def _get_uncontextualized(self, result: ExtractionResult) -> tuple[list, list]:
        """Returns (chunks_todo, artifacts_todo) that still need context."""
        chunks_todo = [c for c in result.source_chunks if not c.contextualized_text]
        artifacts_todo = [
            a for a in list(result.images) + list(result.tables) + list(result.equations)
            if not a.contextualized_text
        ]
        return chunks_todo, artifacts_todo

    def _reembed(self, result: ExtractionResult) -> ExtractionResult:
        """Re-embed chunks with updated contextualized_text."""
        texts = [_chunk_text_for_embedding(c) for c in result.source_chunks]
        result.chunk_embeddings = self._embedder.embed_queries(texts)
        result.chunk_embedding_sources = texts
        return result