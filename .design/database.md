# Database Persistence Architecture

This document describes the SQLite layer used to cache ingested documents, store chunk embeddings for sqlite-vec, and hold intermediate slide-synthesis state.

## Overview

The project uses a single SQLite database file, default path **`data/research.db`** (`StorageConfig` in `src/memory/research/config.py`). The same file holds:

1. **Durable document cache** — parsed PDFs keyed by SHA-256 content hash so re-runs skip LlamaParse and re-embedding when the file is unchanged.
2. **Synthesis workspace** — proto-slides and review events for a single LangGraph run, cleared when `main.py` starts.

Connection setup loads the **sqlite-vec** extension (`load_sqlite_vec_extension` in `src/memory/research/database.py`) so the `text_chunks_vec` virtual table works. Journal mode defaults to **WAL** (`PRAGMA journal_mode=WAL`).

## Research database (`research.db`)

Schema DDL and Pydantic models for slide JSON share `src/memory/research/schema.py`. Row shapes align with `src/processing/document/schema.py` (`ExtractionResult` and artifact dataclasses) when saving via `document.save_document`.

### `documents`

| Role |
|------|
| One row per ingested PDF. Primary key `id` is the `doc_id` used everywhere else. |

Notable columns: `source_path`, `filename`, full-document `markdown`, `page_count`, `content_hash` (SHA-256 of file bytes), optional `run_id` / `schema` (extraction metadata), `paper_metadata` (JSON), `created_at`.

### `text_chunks`

Serialized chunks after splitting. Columns: `text`, `meta_data` (JSON, e.g. `chunk_index`, splitter name), optional `contextualized_text`. Optional `embedding_model` / `embedded_at` exist in DDL for future use; the current save path in `document.save_document` writes `id`, `document_id`, `text`, `meta_data`, `contextualized_text`.

### `text_chunks_vec` (sqlite-vec `vec0`)

Virtual table: `chunk_id` (matches `text_chunks.id`), `embedding` (float vector, width `vec_dimensions`, default **768** from `VEC_DIMENSIONS` in `src/util.py`), and `source` (the string that was embedded — preferring `contextualized_text` when present, else `text`, consistent with `DocProcessor`).

Populated when ingestion supplies `chunk_embeddings` / `chunk_embedding_sources` on the `ExtractionResult`. Vectors with wrong dimension are skipped.

### `images`

Figure and layout images linked by `document_id`. Stores `mime_type`, inline `base64_data` and/or `storage_path` (object store), `page_number`, captions, optional `contextualized_text`, JSON `bbox`, `source_filename`, `confidence`, `category`, `vlm_caption`, `mermaid`, and multi-panel fields (`figure_group_id`, `figure_label`, `figure_number`, `panel_index`, `panel_role`, `identity_signal`).

### `tables`

HTML (or rich) table `content`, dimensions, `page_number`, and contextual fields. The `caption` column is used when loading as the table’s display title (`ExtractedTable.title`).

### `equations`

`text` (LaTeX or plain), `display_mode`, `page_number`, `caption`, `contextualized_text`.

### Automatic caching

`DocProcessor.process_document` hashes the PDF, calls `document_exists` / `load_document_by_hash` on a cache hit, and otherwise runs extraction, optional embedding, verification, then `save_document`.

## Proto-slides workspace (`proto_slides`)

Cleared at the start of each `main.py` run via `clear_proto_slides()` while the document tables above are left intact.

| Column | Purpose |
|--------|---------|
| `slide_number` | Primary key; content slide index. |
| `content` | JSON of structured slide body (`SlideContent` / `ProtoSlide`). |
| `chunk_references` | JSON list of `text_chunks.id` values grounding the slide. |
| `created_at`, `updated_at` | Timestamps. |
| `previous_content`, `previous_chunk_references`, `previous_updated_at` | Prior revision snapshot for rewrites. |

## Review audit (`slide_review_events`)

Cleared with proto-slides at run start (`clear_slide_review_events()`). Records supervisor/critic decisions: `session_id`, `cycle_number`, `scope_type` / `scope_id`, `check_type`, optional `assignment_id`, `issue_code`, `severity`, `fingerprint`, `rewrite_instruction_summary`, `affected_slide_numbers`, `decision`, `created_at`.

## Inspecting the database

Use any SQLite client; load **sqlite-vec** if you query `text_chunks_vec` (same as the app). [DB Browser for SQLite](https://sqlitebrowser.org/) is a common choice.

## Technical notes

- **Vector dimension** must match `StorageConfig.vec_dimensions` and the embedder output (default 768 for sentence-transformers / MPNet-class models).
- **Object storage**: Large or offloadable image bytes may live under `--object-store` (`local` or `r2`); `images.storage_path` holds the reference. Default `main.py` tries R2 then falls back to local.
- **Migrations**: `ResearchDatabase.setup()` applies incremental `ALTER TABLE` for older DB files when new columns are added.
