from __future__ import annotations


# Documents table stores the master record for each processed file
CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    markdown TEXT,
    page_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    run_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    schema TEXT,
    paper_metadata TEXT
);
"""

# Images table stores extracted image data as base64 TEXT
CREATE_IMAGES_TABLE = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    base64_data TEXT NOT NULL,
    page_number INTEGER,
    caption TEXT,
    local_path TEXT,
    contextualized_text TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
"""

# Tables table stores HTML representation of tables
CREATE_TABLES_TABLE = """
CREATE TABLE IF NOT EXISTS tables (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    caption TEXT,
    col_count INTEGER,
    row_count INTEGER,
    contextualized_text TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
"""

# Equations table stores extracted formulas
CREATE_EQUATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS equations (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    display_mode TEXT,
    page_number INTEGER,
    caption TEXT,
    contextualized_text TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
"""

# Text chunks table stores serialized content for embedding
CREATE_TEXT_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS text_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    meta_data TEXT,
    contextualized_text TEXT,
    embedding_model TEXT,
    embedded_at TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
"""

# sqlite-vec virtual table for vector similarity search
CREATE_TEXT_CHUNKS_VEC_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS text_chunks_vec USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[{vec_dimensions}],
    source TEXT
);
"""

# Indexes for performance and filtering
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_images_document_id ON images(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_images_page_number ON images(page_number);",
    "CREATE INDEX IF NOT EXISTS idx_tables_document_id ON tables(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_tables_page_number ON tables(page_number);",
    "CREATE INDEX IF NOT EXISTS idx_equations_document_id ON equations(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_equations_page_number ON equations(page_number);",
    "CREATE INDEX IF NOT EXISTS idx_text_chunks_document_id ON text_chunks(document_id);",
]
