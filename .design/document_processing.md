# Document Processing Design

Detailed technical implementation of document ingestion, chunking, and artifact extraction.

## OCR Backend Architecture

The pipeline uses the **Strategy pattern** to decouple document extraction from any specific OCR library. All backends implement a single interface:

```python
class OCRBackend(ABC):
    @abstractmethod
    def extract(self, source_pdf_path: str) -> ExtractionResult: ...
```

### Backend Registry

`processor.py` maintains a `BACKEND_REGISTRY` mapping string keys to concrete backend classes:

| Key | Class | Status | Requirements | Local Model Size | Processing Time (local)|
|------|------|--------|--------------|------------------|------------------------|
| `"lighton"` | `LightOnOCRBackend` | ✅ Active (Too slow CPU inference (~4 mins per page)) | transformers>=5.0.0, pillow, pypdfium2| ~1 GB | after 10 minutes, only 2 pages |
| `"docling"` | `DoclingBackend` | ✅ Active (has issues with subscripts in text)| docling>=2.70,<3.0 | 1.5-2 GB | 3 minutes |
| `"chandra"` | `ChandraOCRBackend` | ✅ Active (Too slow CPU inference (~7 mins per page))| chandra-ocr[hf] | ~7GB | after 10 minutes, only 1 page |
| `"glm"` | `GLMOCRBackend` | ✅ Active (Slow CPU inference (~2.5 mins per page)) | transformers>=5.0.0, pillow, pypdfium2 | ~1 GB | after 10 minutes, only 4 pages |
| `"marker"` | `MarkerBackend` | ✅ Active (Default, high-accuracy PDF→markdown via surya OCR; extracts images, tables, math) | marker-pdf | ~8 GB | 14 minutes |
| `"llama_parse"` | `LlamaParseBackend` | ✅ Active (Cloud parser, high-accuracy tables & LaTeX, extracts layout images) | llama-cloud | N/A (Cloud) | ~25 seconds |

`DocProcessor` accepts an optional `backend` parameter (string key or `OCRBackend` instance). It defaults to `"marker"`.  Marker is currently the best performing, Docling is the fastest, but has issues with subscripts in text.

Note: some of the requirements are mutually exclusive (e.g. `docling` requires a specific `transformers` version less than 5, while LightOnOCR-2 is only implemented in `transformers` version 5 or later)


### Adding a new backend

1. Create a new module in `src/processing/document/backends/`.
2. Subclass `OCRBackend` and implement `extract()` to return an `ExtractionResult`.
3. Register the class in `processor.BACKEND_REGISTRY`.
4. (Optional) Add a CLI flag in `main.py` (e.g. `--ocr-backend`) and pass the key to `DocProcessor(backend=...)`.

### File layout

```
src/processing/document/
├── __init__.py              # Public re-exports for the document processing package
├── _common.py               # Shared helpers: _slugify, build_artifact_references,
│                            #   _verify_references_in_markdown (no OCR dependency)
├── backend_base.py          # OCRBackend ABC — defines the extract() interface
├── backends/
│   ├── __init__.py          # Guarded imports of all backends; defines __all__
│   ├── chandra_backend.py   # ChandraOCRBackend — local HuggingFace inference
│   │                        #   via InferenceManager(method="hf"); image extraction
│   │                        #   from BatchOutputItem.images; PDF→PIL via load_file()
│   ├── docling_backend.py   # Full Docling pipeline: parse → HybridChunk → images /
│   │                        #   tables / equations → markdown annotation → manifest
│   ├── lighton_backend.py   # LightOnOCR-2-1B-bbox: PDF→PIL via pypdfium2, per-page
│   │                        #   inference, bbox-based image cropping, MarkdownChunker
│   ├── marker_backend.py    # MarkerBackend — default. Uses marker-pdf PdfConverter;
│   │                        #   extracts images as PIL→PNG→base64; parses HTML table
│   │                        #   blocks from markdown; annotates LaTeX equations;
│   │                        #   works on CPU, MPS, or CUDA (auto-detected).
│   └── glm_backend.py       # GLMOCRBackend — local HuggingFace inference via AutoModelForCausalLM;
│                            #   uses pypdfium2 for PDF→PIL conversion
├── chunks.py                # MarkdownChunker — LangChain header-splitter + recursive
│                            #   char fallback; produces list[ExtractedChunk]
├── processor.py             # DocProcessor + BACKEND_REGISTRY factory; default backend
│                            #   is "marker"; accepts string key or OCRBackend instance
└── schema.py                # Shared dataclasses: ExtractedChunk, ExtractedImage,
                             #   ExtractedTable, ExtractedEquation, ExtractionManifest,
                             #   ExtractionResult, ArtifactReference
```

## Multimodal Artifacts

Each run ingests the PDF and writes artifacts to `artifacts/<doc>`, which includes:

- `research.db`: SQLite database (located in `data/`) containing:
    - `images`: Binary BLOBs of extracted pictures.
    - `tables`: HTML representations of tables.
    - `equations`: LaTeX or raw text formulas.
    - `text_chunks`: Semantic markdown chunks with metadata.
    - `manifest`: Index of document-artifact associations.

### Artifact Content

- `document.md`: Human-readable audit artifact. Not loaded into agent state. Any agent that needs the full annotated document can retrieve its path from the manifest and read it from disk.
- `chunks.jsonl`: One JSON object per line, produced by Docling's `HybridChunker`. Each record contains raw text, contextualized text with heading breadcrumbs, ancestor headings, and associated captions.
- `equations.jsonl`: Separate equation records for deterministic downstream processing.
- `manifest.json`: Index of all extracted assets (images, tables, equations) and references mapping markdown tokens to concrete artifacts.

## Document Chunking

Documents are split into semantically coherent, token-aware chunks using Docling's `HybridChunker`, which operates directly on the `DoclingDocument` object before any Markdown export. It respects document structure — chunks never break mid-sentence or mid-paragraph, and each chunk carries the full heading breadcrumb of the section it belongs to.

> [!NOTE]
> "Token indices sequence length is longer than the specified maximum sequence length for this model" is a documented false alarm in Docling's `HybridChunker`.

## Agent Integration

### How agents use chunks

The lead researcher receives a **chunk directory** (heading breadcrumbs only) to understand document structure. It selects relevant chunk indices based on keyword overlap with the query. The editor and critic nodes then receive only the `contextualized_text` of those selected chunks.

### Extending to full RAG

The keyword scorer can be replaced with an embedding similarity search. By embedding the `contextualized_text` at ingestion time, retrieval can switch to vector similarity without altering the downstream agent logic.
