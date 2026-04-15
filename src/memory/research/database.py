from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

from src.logging.logger import AgentLogger
from ..provider.provider import DatabaseProvider
from .config import DEFAULT_CONFIG, StorageConfig, TABLE_NAMES
from . import document
from .schema import (
    CREATE_DOCUMENTS_TABLE,
    CREATE_EQUATIONS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_INDEXES,
    CREATE_PROTO_SLIDES_TABLE,
    CREATE_TABLES_TABLE,
    CREATE_TEXT_CHUNKS_TABLE,
    CREATE_TEXT_CHUNKS_VEC_TABLE,
)
from . import slide


def load_sqlite_vec_extension(conn: sqlite3.Connection) -> None:
    """Register sqlite-vec so `vec0` virtual tables can be used on this connection."""
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception as e:
        raise RuntimeError(f"Failed to load sqlite-vec: {e}") from e
    finally:
        conn.enable_load_extension(False)


def connect_sqlite_with_vec(db_path: Path | str, **kwargs: Any) -> sqlite3.Connection:
    """Open `db_path` and load sqlite-vec (same as `ResearchDatabase` uses internally)."""
    conn = sqlite3.connect(str(db_path), **kwargs)
    load_sqlite_vec_extension(conn)
    return conn


class ResearchDatabase(DatabaseProvider):
    """
    SQLite implementation of the DatabaseProvider.
    Handles persistent document storage and vector search using sqlite-vec.
    """

    def __init__(self, config: StorageConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._conn: sqlite3.Connection | None = None
        self._logger = AgentLogger()
        self.connect()

    def connect(self) -> None:
        """Opens a connection and initializes schema."""
        if self._conn is not None:
            return

        if self.config.auto_create_dirs:
            self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.config.db_path),
            check_same_thread=self.config.check_same_thread,
            isolation_level=self.config.isolation_level
        )
        self._conn.row_factory = sqlite3.Row

        self._conn.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
        self._conn.execute(f"PRAGMA foreign_keys={'ON' if self.config.foreign_keys else 'OFF'}")

        self._load_vec_extension()
        self.setup()

    def disconnect(self) -> None:
        """Closes the connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ResearchDatabase:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def _load_vec_extension(self) -> None:
        """Loads the sqlite-vec extension."""
        if not self._conn:
            raise ValueError("Database not connected.")
        load_sqlite_vec_extension(self._conn)

    def setup(self) -> None:
        """Creates tables and indexes."""
        statements = [
            CREATE_DOCUMENTS_TABLE,
            CREATE_IMAGES_TABLE,
            CREATE_TABLES_TABLE,
            CREATE_EQUATIONS_TABLE,
            CREATE_TEXT_CHUNKS_TABLE,
            CREATE_TEXT_CHUNKS_VEC_TABLE.format(vec_dimensions=self.config.vec_dimensions),
            CREATE_PROTO_SLIDES_TABLE,
        ]
        statements.extend(CREATE_INDEXES)

        with self._conn:
            for stmt in statements:
                self._conn.execute(stmt)

            try:
                doc_columns = [info["name"] for info in self._conn.execute("PRAGMA table_info(documents)").fetchall()]
                if doc_columns:
                    if "run_id" not in doc_columns:
                        self._conn.execute("ALTER TABLE documents ADD COLUMN run_id TEXT;")
                    if "schema" not in doc_columns:
                        self._conn.execute("ALTER TABLE documents ADD COLUMN schema TEXT;")
                    if "paper_metadata" not in doc_columns:
                        self._conn.execute("ALTER TABLE documents ADD COLUMN paper_metadata TEXT;")
            except Exception as e:
                self._logger.log(f"[ResearchDatabase] Schema migration error: {e}")

    def reset(self) -> None:
        """Drops all tables and recreates them."""
        with self._conn:
            for table in TABLE_NAMES:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.setup()

    def document_exists(self, content_hash: str) -> bool:
        return document.document_exists(self, content_hash)

    def load_document_by_hash(self, content_hash: str):
        return document.load_document_by_hash(self, content_hash)

    def save_document(self, result):
        return document.save_document(self, result)

    def load_document(self, doc_id: str):
        return document.load_document(self, doc_id)

    def get_image(self, image_id: str):
        return document.get_image(self, image_id)

    def get_table(self, table_id: str):
        return document.get_table(self, table_id)

    def get_chunks_for_dispatch(self, doc_id: str):
        return document.get_chunks_for_dispatch(self, doc_id)

    def save_slide(self, slide_item):
        return slide.save_slide(self, slide_item)

    def load_slide(self, slide_number: int):
        return slide.load_slide(self, slide_number)
'''
    def load_document(self, doc_id: str) -> ExtractionResult | None:
        """Loads an ExtractionResult from the database."""
        # 1. Load Document
        doc_row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not doc_row:
            return None

        # 2. Load Images
        img_rows = self._conn.execute("SELECT * FROM images WHERE document_id = ?", (doc_id,)).fetchall()
        images = [
            ExtractedImage(
                id=row["id"],
                mime_type=row["mime_type"],
                base64_data=row["base64_data"] or "",
                page=row["page_number"],
                caption=row["caption"] or "",
                storage_path=row["storage_path"],
                contextualized_text=row["contextualized_text"],
            ) for row in img_rows
        ]

        # 3. Load Tables
        tbl_rows = self._conn.execute("SELECT * FROM tables WHERE document_id = ?", (doc_id,)).fetchall()
        tables = [
            ExtractedTable(
                id=row["id"],
                content=row["content"],
                page=row["page_number"],
                title=row["caption"] or "",
                contextualized_text=row["contextualized_text"],
                col_count=row["col_count"],
                row_count=row["row_count"]
            ) for row in tbl_rows
        ]

        # 4. Load Equations
        eq_rows = self._conn.execute("SELECT * FROM equations WHERE document_id = ?", (doc_id,)).fetchall()
        equations = [
            ExtractedEquation(
                id=row["id"],
                latex_or_text=row["text"],
                display_mode=row["display_mode"] or "block",
                page=row["page_number"],
                caption=row["caption"] or "",
                contextualized_text=row["contextualized_text"]
            ) for row in eq_rows
        ]

        paper_metadata = None
        if doc_row["paper_metadata"]:
            paper_metadata = PaperMetadata(**json.loads(doc_row["paper_metadata"]))

        # 5. Load Chunks
        chunk_rows = self._conn.execute(
            "SELECT * FROM text_chunks WHERE document_id = ? ORDER BY COALESCE(CAST(json_extract(meta_data, '$.chunk_index') AS INTEGER), id)", 
            (doc_id,)
        ).fetchall()
        source_chunks = [
            ExtractedChunk(
                id=row["id"],
                text=row["text"],
                contextualized_text=row["contextualized_text"],
                meta_data=json.loads(row["meta_data"])
            ) for row in chunk_rows
        ]

        return ExtractionResult(
            doc_id=doc_id,
            source_path=doc_row["source_path"],
            markdown=doc_row["markdown"],
            source_chunks=source_chunks,
            images=images,
            tables=tables,
            equations=equations,
            page_count=doc_row["page_count"],
            run_id=doc_row["run_id"],
            schema=doc_row["schema"],
            paper_metadata=paper_metadata,
        )

    def get_image(self, image_id: str) -> ExtractedImage | None:
        """Loads a specific image from the database by its ID."""
        row = self._conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if not row:
            return None
        return ExtractedImage(
            id=row["id"],
            mime_type=row["mime_type"],
            base64_data=row["base64_data"] or "",
            page=row["page_number"],
            caption=row["caption"] or "",
            storage_path=row["storage_path"],
            contextualized_text=row["contextualized_text"],
        )

    def get_table(self, table_id: str) -> ExtractedTable | None:
        """Loads a specific table from the database by its ID."""
        row = self._conn.execute("SELECT * FROM tables WHERE id = ?", (table_id,)).fetchone()
        if not row:
            return None
        return ExtractedTable(
            id=row["id"],
            content=row["content"],
            page=row["page_number"],
            title=row["caption"] or "",
            contextualized_text=row["contextualized_text"],
            col_count=row["col_count"],
            row_count=row["row_count"]
        )

    def get_chunks_for_dispatch_multi(self, doc_ids: list[str]) -> dict[str, list[dict]]:
        """
        Return ordered chunks for multiple documents in one call.

        Returns a dict keyed by doc_id, each value being the same list[dict]
        that get_chunks_for_dispatch() returns for that doc_id.  Documents with
        no chunks are included as empty lists so callers can detect missing docs.
        """
        return {doc_id: self.get_chunks_for_dispatch(doc_id) for doc_id in doc_ids}

    def get_chunks_for_dispatch(self, doc_id: str) -> list[dict]:
        """
        Return all text chunks for a document ordered by chunk_index, as lightweight
        dicts with keys: id, text, meta_data (parsed dict).

        This is a cheaper alternative to load_document() — it skips images, tables,
        equations, and embeddings so the PlanExecutor can analyse section structure
        without loading the full ExtractionResult.
        """
        rows = self._conn.execute(
            """
            SELECT id, text, meta_data
            FROM   text_chunks
            WHERE  document_id = ?
            ORDER BY COALESCE(
                CAST(json_extract(meta_data, '$.chunk_index') AS INTEGER),
                rowid
            )
            """,
            (doc_id,),
        ).fetchall()

        return [
            {
                "id":        row["id"],
                "text":      row["text"],
                "meta_data": json.loads(row["meta_data"]) if row["meta_data"] else {},
            }
            for row in rows
        ]

'''
    def list_slide_numbers(self) -> list[int]:
        return slide.list_slide_numbers(self)

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("Database disconnected.")
        return self._conn
