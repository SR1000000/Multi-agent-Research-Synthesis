from __future__ import annotations
import json
from typing import List, Literal, Optional
from pydantic import AliasChoices, BaseModel, Field, create_model, field_validator


class BulletPoint(BaseModel):
    text: str = Field(
        description=(
            'The bullet point text. Supports Markdown formatting and LaTeX math '
            '(e.g., inline `$E=mc^2$` or display `$$\\frac{a}{b}$$`). '
            'Field name is "text" — do NOT use "content".'
        ),
        validation_alias=AliasChoices("text", "content"),
    )
    sub_bullets: List[str] = Field(
        default_factory=list,
        description='Optional sub-bullet points as plain strings (NOT objects). Example: ["Detail A", "Detail B"]',
    )
    content_type: Literal["insight", "evidence", "statistic", "example", "caveat"] = Field(default="insight", description="Semantic type of this bullet point")

    @field_validator("sub_bullets", mode="before")
    @classmethod
    def coerce_sub_bullets(cls, v: list) -> list[str]:
        if not isinstance(v, list):
            return v
        return [
            (item.get("text") or item.get("content") or str(item))
            if isinstance(item, dict)
            else str(item)
            for item in v
        ]


class SlideContent(BaseModel):
    title: str = Field(description="Punchy, active heading for the slide (e.g. 'Accuracy Jumps 40%' not 'Accuracy Results')")
    subtitle: Optional[str] = Field(default=None, description="Optional subtitle, primarily for title slides and major section openers")
    key_message: str = Field(description="One sentence capturing what the audience should understand after this slide")
    bullets: List[BulletPoint] = Field(description="3-5 structured bullet points for the slide body")
    speaker_notes: str = Field(description="Speaker notes in a professional, conversational tone — include context and nuance too detailed for the slide itself")
    media_id: Optional[str] = Field(default=None, description="Optional ID of an image or chart from research.db")
    layout: Literal["title_slide", "title_and_body", "two_column", "big_number", "quote", "media_left", "media_right"] = Field(
        default="title_and_body",
        description="Slide layout: 'title_slide' for openers/dividers, 'title_and_body' for standard bullet slides, 'two_column' for comparisons, 'big_number' for stat callouts, 'quote' for direct quotations, 'media_left'/'media_right' when a figure or chart is the focus"
    )
    narrative_role: Literal["hook", "problem", "evidence", "insight", "transition", "call_to_action", "conclusion"] = Field(
        default="evidence",
        description="Role this slide plays in the deck's narrative arc: 'hook' grabs attention, 'problem' establishes the challenge or gap, 'evidence' presents data, 'insight' delivers the key takeaway, 'transition' bridges sections, 'call_to_action' motivates next steps or future work, 'conclusion' wraps up"
    )


class ProtoSlide(BaseModel):
    slide_number: int = Field(description="The slide number")
    content: SlideContent = Field(description="The structured content of the slide")
    chunk_references: List[str] = Field(
        description=(
            "Ordered list of exact text chunk IDs from research.db assigned to this slide. "
            "Order is significant: chunks are listed in the sequence they should be read/cited."
        )
    )


def make_slide_batch_model(slide_count: int) -> type[BaseModel]:
    """Return a schema for a slide batch with an exact number of slides."""
    return create_model(
        f"SlideBatch_{slide_count}",
        slides=(
            List[SlideContent],
            Field(
                description=f"The synthesized slides for this batch. Must contain exactly {slide_count} slides.",
                min_length=slide_count,
                max_length=slide_count,
            ),
        ),
    )


def slide_output_prompt_contract(slide_count: int) -> str:
    """Return schema-derived prompt guidance for slide generation."""
    batch_model = make_slide_batch_model(slide_count)
    schema_json = json.dumps(batch_model.model_json_schema(), indent=2)
    rules = [
        f'Return exactly {slide_count} slide objects in the top-level `slides` array.',
        "All information must be strictly grounded in the provided research chunks.",
        "Use Markdown and LaTeX only when they materially improve clarity.",
        "Display equations should appear as the sole content of a sub_bullet string.",
    ]
    lines = [
        "### REQUIRED ROOT JSON SHAPE:",
        "- Return exactly ONE top-level JSON object matching the schema below.",
        '- The top-level key MUST be `slides`.',
        "- Do NOT return multiple top-level objects.",
        "- Do NOT return newline-delimited JSON.",
        "- Do NOT return a top-level array.",
        "- Do NOT include any text before or after the JSON object.",
        "",
        "### ADDITIONAL RULES:",
        *[f"- {rule}" for rule in rules],
        "",
        "### EXACT JSON SCHEMA:",
        schema_json,
    ]
    return "\n".join(lines)


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

# Images table stores extracted image data
CREATE_IMAGES_TABLE = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    base64_data TEXT,
    storage_path TEXT,
    page_number INTEGER,
    caption TEXT,
    contextualized_text TEXT,
    bbox TEXT,
    source_filename TEXT,
    confidence REAL,
    category TEXT,
    vlm_caption TEXT,
    mermaid TEXT,
    figure_group_id TEXT,
    figure_label TEXT,
    figure_number INTEGER,
    panel_index INTEGER,
    panel_role TEXT,
    identity_signal TEXT,
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

# Physical full-text index used by keyword retrieval.
# FTS tokenizes and indexes only `search_text`. UNINDEXED fields are metadata payload.
# Rows are synchronized by DML triggers on source artifact tables.
CREATE_ARTIFACT_SEARCH_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS artifact_search_fts USING fts5(
    item_id UNINDEXED,
    document_id UNINDEXED,
    kind UNINDEXED,
    search_text
);
"""

# This is not an actual row store but just a view to normalize all searchable artifacts into one shape.
# Retrieval text policy: concatenate contextualized + raw content when both exist.
CREATE_ARTIFACT_SEARCH_SOURCE_VIEW = """
CREATE VIEW IF NOT EXISTS artifact_search_source AS
SELECT
    id AS item_id,
    document_id,
    'chunk' AS kind,
    CASE
        WHEN NULLIF(contextualized_text, '') IS NOT NULL
             AND NULLIF(text, '') IS NOT NULL
            THEN contextualized_text || char(10) || char(10) || text
        ELSE COALESCE(NULLIF(contextualized_text, ''), text)
    END AS search_text
FROM text_chunks
UNION ALL
SELECT
    id AS item_id,
    document_id,
    'table' AS kind,
    CASE
        WHEN NULLIF(contextualized_text, '') IS NOT NULL
             AND NULLIF(content, '') IS NOT NULL
            THEN contextualized_text || char(10) || char(10) || content
        ELSE COALESCE(NULLIF(contextualized_text, ''), content)
    END AS search_text
FROM tables
UNION ALL
SELECT
    id AS item_id,
    document_id,
    'equation' AS kind,
    CASE
        WHEN NULLIF(contextualized_text, '') IS NOT NULL
             AND NULLIF(text, '') IS NOT NULL
            THEN contextualized_text || char(10) || char(10) || text
        ELSE COALESCE(NULLIF(contextualized_text, ''), text)
    END AS search_text
FROM equations
UNION ALL
SELECT
    id AS item_id,
    document_id,
    'image' AS kind,
    CASE
        WHEN NULLIF(contextualized_text, '') IS NOT NULL
             AND NULLIF(COALESCE(caption, storage_path), '') IS NOT NULL
            THEN contextualized_text || char(10) || char(10) || COALESCE(caption, storage_path)
        ELSE COALESCE(NULLIF(contextualized_text, ''), caption, storage_path)
    END AS search_text
FROM images;
"""

# This transactionally aligned with base tables after updates  to specific tables.
# INSERT adds index row, UPDATE performs delete+insert, DELETE removes index row.
# Maintenance applies per artifact table; delete key is `(item_id, kind)` since IDs are table-local.
CREATE_ARTIFACT_SEARCH_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_text_chunks_ai
    AFTER INSERT ON text_chunks
    BEGIN
        INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
        VALUES (
            NEW.id,
            NEW.document_id,
            'chunk',
            CASE
                WHEN NULLIF(NEW.contextualized_text, '') IS NOT NULL
                     AND NULLIF(NEW.text, '') IS NOT NULL
                    THEN NEW.contextualized_text || char(10) || char(10) || NEW.text
                ELSE COALESCE(NULLIF(NEW.contextualized_text, ''), NEW.text)
            END
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_text_chunks_au
    AFTER UPDATE ON text_chunks
    BEGIN
        DELETE FROM artifact_search_fts WHERE item_id = OLD.id AND kind = 'chunk';
        INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
        VALUES (
            NEW.id,
            NEW.document_id,
            'chunk',
            CASE
                WHEN NULLIF(NEW.contextualized_text, '') IS NOT NULL
                     AND NULLIF(NEW.text, '') IS NOT NULL
                    THEN NEW.contextualized_text || char(10) || char(10) || NEW.text
                ELSE COALESCE(NULLIF(NEW.contextualized_text, ''), NEW.text)
            END
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_text_chunks_ad
    AFTER DELETE ON text_chunks
    BEGIN
        DELETE FROM artifact_search_fts WHERE item_id = OLD.id AND kind = 'chunk';
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_tables_ai
    AFTER INSERT ON tables
    BEGIN
        INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
        VALUES (
            NEW.id,
            NEW.document_id,
            'table',
            CASE
                WHEN NULLIF(NEW.contextualized_text, '') IS NOT NULL
                     AND NULLIF(NEW.content, '') IS NOT NULL
                    THEN NEW.contextualized_text || char(10) || char(10) || NEW.content
                ELSE COALESCE(NULLIF(NEW.contextualized_text, ''), NEW.content)
            END
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_tables_au
    AFTER UPDATE ON tables
    BEGIN
        DELETE FROM artifact_search_fts WHERE item_id = OLD.id AND kind = 'table';
        INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
        VALUES (
            NEW.id,
            NEW.document_id,
            'table',
            CASE
                WHEN NULLIF(NEW.contextualized_text, '') IS NOT NULL
                     AND NULLIF(NEW.content, '') IS NOT NULL
                    THEN NEW.contextualized_text || char(10) || char(10) || NEW.content
                ELSE COALESCE(NULLIF(NEW.contextualized_text, ''), NEW.content)
            END
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_tables_ad
    AFTER DELETE ON tables
    BEGIN
        DELETE FROM artifact_search_fts WHERE item_id = OLD.id AND kind = 'table';
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_equations_ai
    AFTER INSERT ON equations
    BEGIN
        INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
        VALUES (
            NEW.id,
            NEW.document_id,
            'equation',
            CASE
                WHEN NULLIF(NEW.contextualized_text, '') IS NOT NULL
                     AND NULLIF(NEW.text, '') IS NOT NULL
                    THEN NEW.contextualized_text || char(10) || char(10) || NEW.text
                ELSE COALESCE(NULLIF(NEW.contextualized_text, ''), NEW.text)
            END
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_equations_au
    AFTER UPDATE ON equations
    BEGIN
        DELETE FROM artifact_search_fts WHERE item_id = OLD.id AND kind = 'equation';
        INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
        VALUES (
            NEW.id,
            NEW.document_id,
            'equation',
            CASE
                WHEN NULLIF(NEW.contextualized_text, '') IS NOT NULL
                     AND NULLIF(NEW.text, '') IS NOT NULL
                    THEN NEW.contextualized_text || char(10) || char(10) || NEW.text
                ELSE COALESCE(NULLIF(NEW.contextualized_text, ''), NEW.text)
            END
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_equations_ad
    AFTER DELETE ON equations
    BEGIN
        DELETE FROM artifact_search_fts WHERE item_id = OLD.id AND kind = 'equation';
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_images_ai
    AFTER INSERT ON images
    BEGIN
        INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
        VALUES (
            NEW.id,
            NEW.document_id,
            'image',
            CASE
                WHEN NULLIF(NEW.contextualized_text, '') IS NOT NULL
                     AND NULLIF(COALESCE(NEW.caption, NEW.storage_path), '') IS NOT NULL
                    THEN NEW.contextualized_text || char(10) || char(10) || COALESCE(NEW.caption, NEW.storage_path)
                ELSE COALESCE(NULLIF(NEW.contextualized_text, ''), NEW.caption, NEW.storage_path)
            END
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_images_au
    AFTER UPDATE ON images
    BEGIN
        DELETE FROM artifact_search_fts WHERE item_id = OLD.id AND kind = 'image';
        INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
        VALUES (
            NEW.id,
            NEW.document_id,
            'image',
            CASE
                WHEN NULLIF(NEW.contextualized_text, '') IS NOT NULL
                     AND NULLIF(COALESCE(NEW.caption, NEW.storage_path), '') IS NOT NULL
                    THEN NEW.contextualized_text || char(10) || char(10) || COALESCE(NEW.caption, NEW.storage_path)
                ELSE COALESCE(NULLIF(NEW.contextualized_text, ''), NEW.caption, NEW.storage_path)
            END
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_artifact_search_images_ad
    AFTER DELETE ON images
    BEGIN
        DELETE FROM artifact_search_fts WHERE item_id = OLD.id AND kind = 'image';
    END;
    """,
]

CREATE_PROTO_SLIDES_TABLE = """
CREATE TABLE IF NOT EXISTS proto_slides (
    slide_number INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    chunk_references TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    previous_content TEXT,
    previous_chunk_references TEXT,
    previous_updated_at TEXT
);
"""

CREATE_SLIDE_REVIEW_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS slide_review_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cycle_number INTEGER NOT NULL,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    check_type TEXT NOT NULL,
    assignment_id TEXT,
    issue_code TEXT,
    severity TEXT,
    fingerprint TEXT,
    rewrite_instruction_summary TEXT,
    affected_slide_numbers TEXT,
    decision TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

# Agent retrieval call log (formerly in wip.db)
CREATE_RETRIEVED_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS retrieved_chunks (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    document_id TEXT NOT NULL,
    text_content TEXT NOT NULL,
    score REAL,
    retrieved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    agent_type TEXT,
    query TEXT
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
    "CREATE INDEX IF NOT EXISTS idx_slide_review_events_session_cycle ON slide_review_events(session_id, cycle_number);",
]
