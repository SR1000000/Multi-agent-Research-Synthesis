from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

from src.logging.logger import AgentLogger
from src.retriever.types import RetrievedItem
from src.memory.provider.provider import DatabaseProvider
from src.memory.research.config import DEFAULT_CONFIG, StorageConfig, TABLE_NAMES
from src.memory.research import document
from src.memory.research import slide
from src.memory.research.schema import (
    CREATE_DOCUMENTS_TABLE,
    CREATE_EQUATIONS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_INDEXES,
    CREATE_PROTO_SLIDES_TABLE,
    CREATE_RETRIEVED_CHUNKS_TABLE,
    CREATE_SLIDE_REVIEW_EVENTS_TABLE,
    CREATE_TABLES_TABLE,
    CREATE_TEXT_CHUNKS_TABLE,
    CREATE_TEXT_CHUNKS_VEC_TABLE,
)
from src.memory.research.retrieval import (
    FETCH_ALL_EQUATIONS_SQL,
    FETCH_ALL_IMAGES_SQL,
    FETCH_ALL_TABLES_SQL,
    FETCH_ALL_TEXT_CHUNKS_SQL,
    KNN_TEXT_CHUNKS_SQL,
)


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
            isolation_level=self.config.isolation_level,
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
            CREATE_RETRIEVED_CHUNKS_TABLE,
            CREATE_SLIDE_REVIEW_EVENTS_TABLE,
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

                image_columns = [info["name"] for info in self._conn.execute("PRAGMA table_info(images)").fetchall()]
                if image_columns:
                    if "bbox" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN bbox TEXT;")
                    if "source_filename" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN source_filename TEXT;")
                    if "confidence" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN confidence REAL;")
                    if "category" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN category TEXT;")
                    if "vlm_caption" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN vlm_caption TEXT;")
                    if "mermaid" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN mermaid TEXT;")
                    if "figure_group_id" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN figure_group_id TEXT;")
                    if "figure_label" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN figure_label TEXT;")
                    if "figure_number" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN figure_number INTEGER;")
                    if "panel_index" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN panel_index INTEGER;")
                    if "panel_role" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN panel_role TEXT;")
                    if "identity_signal" not in image_columns:
                        self._conn.execute("ALTER TABLE images ADD COLUMN identity_signal TEXT;")

                slide_columns = [
                    info["name"] for info in self._conn.execute("PRAGMA table_info(proto_slides)").fetchall()
                ]
                if slide_columns:
                    if "previous_content" not in slide_columns:
                        self._conn.execute("ALTER TABLE proto_slides ADD COLUMN previous_content TEXT;")
                    if "previous_chunk_references" not in slide_columns:
                        self._conn.execute(
                            "ALTER TABLE proto_slides ADD COLUMN previous_chunk_references TEXT;"
                        )
                    if "previous_updated_at" not in slide_columns:
                        self._conn.execute("ALTER TABLE proto_slides ADD COLUMN previous_updated_at TEXT;")

                review_event_columns = [
                    info["name"]
                    for info in self._conn.execute("PRAGMA table_info(slide_review_events)").fetchall()
                ]
                if review_event_columns and "affected_slide_numbers" not in review_event_columns:
                    self._conn.execute(
                        "ALTER TABLE slide_review_events ADD COLUMN affected_slide_numbers TEXT;"
                    )
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

    def list_slide_numbers(self) -> list[int]:
        return slide.list_slide_numbers(self)

    def clear_proto_slides(self) -> None:
        return slide.clear_proto_slides(self)

    def clear_slide_review_events(self) -> None:
        return slide.clear_slide_review_events(self)

    def save_review_event(self, **kwargs) -> None:
        return slide.save_review_event(self, **kwargs)

    def list_review_events(self, session_id: str) -> list[dict]:
        return slide.list_review_events(self, session_id)

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("Database disconnected.")
        return self._conn

    def knn_text_chunks_by_embedding(self, embedding_blob: bytes, k: int) -> list[dict[str, Any]]:
        """SQLite-vec KNN over `text_chunks_vec`, joined to `text_chunks`."""
        rows = self._conn.execute(KNN_TEXT_CHUNKS_SQL, (embedding_blob, k)).fetchall()
        return [dict(r) for r in rows]

    def fetch_all_text_chunks_for_retrieval(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(FETCH_ALL_TEXT_CHUNKS_SQL).fetchall()
        return [dict(r) for r in rows]

    def fetch_all_tables_for_retrieval(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(FETCH_ALL_TABLES_SQL).fetchall()
        return [dict(r) for r in rows]

    def fetch_all_equations_for_retrieval(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(FETCH_ALL_EQUATIONS_SQL).fetchall()
        return [dict(r) for r in rows]

    def fetch_all_images_for_retrieval(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(FETCH_ALL_IMAGES_SQL).fetchall()
        return [dict(r) for r in rows]

    def save_retrieved_chunk(
        self,
        item: RetrievedItem,
        session_id: str,
        agent_type: str,
        query: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO retrieved_chunks
                (id, kind, document_id, text_content, score, session_id, agent_type, query, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    item.id,
                    item.kind,
                    item.document_id,
                    item.text,
                    item.score,
                    session_id,
                    agent_type,
                    query,
                ),
            )
        self._logger.log(f"[ResearchDatabase] Saved retrieved chunk {item.id}")

    def load_retrieved_chunks(self, session_id: str | None = None) -> list[RetrievedItem]:
        sql = "SELECT id, kind, document_id, text_content, score FROM retrieved_chunks"
        params: tuple[Any, ...] = ()
        if session_id:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            RetrievedItem(
                id=row["id"],
                kind=row["kind"],
                document_id=row["document_id"],
                text=row["text_content"],
                score=row["score"],
            )
            for row in rows
        ]
