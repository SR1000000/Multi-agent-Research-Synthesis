# Document Processing Design

Technical description of PDF ingestion, chunking, artifact extraction, and how results feed the database and graph.

## OCR backend architecture

The pipeline uses the **Strategy** pattern: each backend subclasses `OCRBackend` in `src/processing/document/backend_base.py` and implements `extract(source_pdf_path: str) -> ExtractionResult`.

`DocProcessor` (`src/processing/document/processor.py`) resolves a string backend name through `get_ocr_backend`, which reads `BACKEND_REGISTRY`.

### Backend Registry

`processor.py` maintains a `BACKEND_REGISTRY` mapping string keys to concrete backend classes:

| Key | Class | Status | Requirements | Local Model Size | Processing Time (local, cpu-only)|
|------|------|--------|--------------|------------------|------------------------|
| `"lighton"` | `LightOnOCRBackend` | ❌ Inactive (Too slow CPU inference (~4 mins per page)) | transformers>=5.0.0, pillow, pypdfium2| ~1 GB | after 10 minutes, only 2 pages |
| `"docling"` | `DoclingBackend` | ❌ Inactive (has issues with subscripts in text)| docling>=2.70,<3.0 | 1.5-2 GB | 3 minutes |
| `"chandra"` | `ChandraOCRBackend` | ❌ Inactive (Too slow CPU inference (~7 mins per page))| chandra-ocr[hf] | ~7GB | after 10 minutes, only 1 page |
| `"glm"` | `GLMOCRBackend` | ❌ Inactive (Slow CPU inference (~2.5 mins per page)) | transformers>=5.0.0, pillow, pypdfium2 | ~1 GB | after 10 minutes, only 4 pages |
| `"marker"` | `MarkerBackend` | ❌ Inactive (high-accuracy PDF→markdown via surya OCR; extracts images, tables, math) | marker-pdf | ~8 GB | 14 minutes |
| `"llama_parse"` | `LlamaParseBackend` | ✅ Active (Cloud parser, high-accuracy tables & LaTeX, extracts layout images) | llama-cloud | N/A (Cloud) | ~25 seconds |

`processor.ARCHIVAL_BACKENDS` lists `docling`, `lighton`, `chandra`, `glm`, `marker`. Choosing one of these raises a clear error: older modules may still exist under `src/processing/document/backends/`, but they are **not** wired into `BACKEND_REGISTRY` in the current build.

### `DocProcessor` pipeline

1. **Cache** — SHA-256 of the PDF; if `db.document_exists(hash)`, return `load_document_by_hash` (skips parse and embedding).
2. **Extract** — `backend.extract(path)`; result gets `content_hash` and `run_id` as set by the backend / caller.
3. **Contextualize** — Two distinct contextualizers are injected by `main.py` into `DocProcessor`:
   - **`Contextualizer`** (`src/processing/context/contextualizer.py`): Runs batch LLM calls to produce localized `contextualized_text` for each chunk, image, table, and equation. Supports `--no-cache-control` and `--no-context-batching` flags. Output is used for the embedder and FTS index.
   - **`DocumentContextualizer`** (`src/processing/context/document.py`): Runs a separate LLM pass to produce a document-level structured summary (section outline, paper metadata) stored in `documents.document_context`. This is a distinct second stage with its own config.
4. **Embed** — if an embedder is provided, each chunk is embedded using `contextualized_text` when non-empty, else `text`; vectors are stored on the `ExtractionResult` for DB persistence.
5. **Verify** — `_common.verify_extraction_result` checks consistency.
6. **Persist** — optional debug JSON dump via the logger; then `db.save_document` when a database is configured.

Default backend is **`llama_parse`**.

### Adding a new backend

1. Implement a module under `src/processing/document/backends/`.
2. Subclass `OCRBackend` and implement `extract()`.
3. Add the class to `BACKEND_REGISTRY` in `processor.py` (and extend `get_ocr_backend` if the constructor needs extra deps, as with `LlamaParseBackend`).
4. Expose a `--processor` choice in `main.py` if it should be user-selectable.

### File layout (current)

```
src/processing/document/
├── __init__.py
├── _common.py              # Shared helpers; verify_extraction_result
├── backend_base.py         # OCRBackend
├── backends/
│   ├── __init__.py         # Imports legacy backends if deps exist (not all registered)
│   ├── llama_parse_backend.py
│   ├── llama_parse_figures.py
│   └── …                   # Other backend modules (archival / optional installs)
├── processor.py            # DocProcessor, BACKEND_REGISTRY, get_ocr_backend
└── schema.py               # ExtractedChunk, ExtractedImage, …, ExtractionResult

src/processing/chunker/
├── __init__.py             # get_text_chunker
├── provider.py             # TextChunkerProvider
├── semantic_text_splitter_chunker.py  # MarkdownSplitter / TextSplitter
└── config.py
```

## LlamaParse behavior (summary)

`LlamaParseBackend` calls LlamaCloud with structured options (tier, markdown/table settings, layout images, items + metadata expansions). It builds `PaperMetadata` from the markdown, rewrites image/table placeholders, runs figure grouping in `llama_parse_figures.py`, and fills `ExtractionResult` with markdown, images, tables, equations, and chunks.



## Text chunking

After full-document markdown is produced:

- **`--text-splitter semantic` (default)** — `SemanticTextSplitterChunker` uses the `semantic-text-splitter` library (`MarkdownSplitter` for markdown).
- **`--text-splitter none`** — `text_chunker` is `None`; the backend emits a **single** chunk (`splitter: none` in `meta_data`).

`main.py` maps `--text-splitter` via `_TEXT_SPLITTER_ALIASES` and passes the chunker into `DocProcessor`.

## Multimodal artifacts and storage

- **SQLite (`data/research.db`)** — See `database.md`: documents, chunks, vec table, images, tables, equations.
- **Object store** — `LlamaParseBackend` can upload images through an `ObjectStoreProvider` (`LocalObjectStore` / `R2ObjectStore` from `main.py` `--object-store`), with paths recorded on `ExtractedImage.storage_path`.

## Graph integration

Document processing runs before the graph. Its outputs become the evidence base used by planning, slide drafting, critic review, and final export.

### How graph stages use chunks

- **Planning** loads all chunks per document, groups them into sections using heading heuristics on chunk text, and builds a presentation plan whose slide blueprints reference the source chunks.
- **Drafting and review** load the referenced source rows and use ordered text, preferring contextualized text where useful. Slide drafting can also use retrieval over the stored artifacts during initial writing and rewriting.

### Embeddings and retrieval

Chunks are embedded at ingestion for storage in `text_chunks_vec`. The current planning path is section-based over full chunk lists; retrieval over embeddings and stored artifacts is available to later graph stages that need more targeted evidence.
