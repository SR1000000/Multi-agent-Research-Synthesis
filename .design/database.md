# Database Persistence Architecture

This document describes the design of the database layer used to store extraction artifacts and intermediate synthesis state. The project utilizes a dual-database architecture to separate persistent research data from volatile session data.

## Overview

The system uses two distinct SQLite databases located in the `data/` directory:

1.  **`research.db`**: A long-term storage for processed documents. It acts as a cache to avoid re-parsing expensive PDFs and provides vector search capabilities for the research agents.
2.  **`proto_slides` in `research.db`**: A temporary workspace used during the slide synthesis pipeline. It stores "proto-slides" which are intermediate representations of presentation content before they are exported to PowerPoint.

## Research Database (`research.db`)

The `research.db` uses `sqlite-vec` to provide native vector search within SQLite. It stores the full output of the `DocProcessor` pipeline.

### Schema Details

The tables in `research.db` map to the models defined in `src.processing.document.schema`.

-   **`documents`**: Tracks ingested files by `doc_id`. Stores the raw `markdown` representation, `page_count`, and `paper_metadata` (title, authors, etc.).
-   **`text_chunks`**: Stores semantic text blocks extracted from the document. Each chunk includes its original `text` and often a `contextualized_text` (LLM-augmented for better retrieval).
-   **`text_chunks_vec`**: A virtual table (using `sqlite-vec`) storing embeddings for each text chunk to support semantic search.
-   **`images`**: Stores image metadata, `caption`, and either `base64_data` or a `storage_path` (pointing to the Object Store).
-   **`tables`**: Stores extracted tables as HTML `content` along with rows/column counts and captions.
-   **`equations`**: Stores LaTeX or text representations of math equations found in the PDF.

### Automatic Caching

Caching is handled automatically by `DocProcessor`. When a PDF is processed:
1.  The file's SHA-256 `content_hash` is calculated.
2.  The database is queried to see if a document with that hash already exists.
3.  If found, the system loads the results directly from the database, skipping OCR and embedding steps.

## Proto-Slides Workspace (`research.db`)

The `proto_slides` table in `research.db` is reset at the start of every execution of `main.py`. It serves as a shared workspace between the synthesis agents and the presentation builder while leaving the ingested document cache intact.

### Schema Details

-   **`proto_slides`**: 
    -   `slide_number` (ID): The sequence position of the slide.
    -   `content` (JSON): The structured slide content (title, bullet points, image descriptors).
    -   `chunk_references` (JSON): A list of IDs linking back to the `text_chunks` in `research.db` that provided the evidence for this slide.

## Inspecting the Databases

The databases are standard SQLite files and can be inspected with any SQLite client.

**Recommendation:** Use [DB Browser for SQLite](https://sqlitebrowser.org/) or SQLite Viewer extension for VS Code.

1.  Open `data/research.db` to see indexed papers and their extracted media.
2.  Open `data/research.db` and inspect the `proto_slides` table (while the agent is running or after the program finishes) to see the intermediate slide data being generated.

## Technical Notes

-   **Vector Search**: The project requires the `sqlite-vec` extension. This is handled automatically via the `sqlite-vec` Python package.
-   **WAL Mode**: Both databases use Write-Ahead Logging (`PRAGMA journal_mode=WAL`) for better performance and concurrency.
-   **Object Storage**: While metadata is in SQLite, actual image bytes may reside in a local directory or R2 bucket depending on the `--object-store` CLI flag, with the path stored in the `images` table.
