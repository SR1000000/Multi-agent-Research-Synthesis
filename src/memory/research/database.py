from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlite_vec

from src.logging.logger import AgentLogger
from src.memory.provider.provider import DatabaseProvider
from src.memory.research import document, slide
from src.memory.research.config import DEFAULT_CONFIG, StorageConfig, TABLE_NAMES
from src.memory.research.retrieval import (
    ensure_artifact_search_index as retrieval_ensure_artifact_search_index,
    fetch_all_equations_for_retrieval as retrieval_fetch_all_equations_for_retrieval,
    fetch_all_images_for_retrieval as retrieval_fetch_all_images_for_retrieval,
    fetch_all_tables_for_retrieval as retrieval_fetch_all_tables_for_retrieval,
    fetch_all_text_chunks_for_retrieval as retrieval_fetch_all_text_chunks_for_retrieval,
    knn_text_chunks_by_embedding as retrieval_knn_text_chunks_by_embedding,
    load_normalized_artifacts_for_call as retrieval_load_normalized_artifacts_for_call,
    load_normalized_artifacts_for_keys as retrieval_load_normalized_artifacts_for_keys,
    load_normalized_artifacts_for_session as retrieval_load_normalized_artifacts_for_session,
    load_retrieved_chunks as retrieval_load_retrieved_chunks,
    query_artifact_search as retrieval_query_artifact_search,
    rebuild_artifact_search_index as retrieval_rebuild_artifact_search_index,
    save_session_retrieval_batch as retrieval_save_session_retrieval_batch,
)
from src.memory.research.schema import (
    CREATE_ARTIFACT_SEARCH_FTS_TABLE,
    CREATE_ARTIFACT_SEARCH_SOURCE_VIEW,
    CREATE_DOCUMENTS_TABLE,
    CREATE_EQUATIONS_TABLE,
    CREATE_FTS_ROWID_MAP_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_INDEXES,
    CREATE_PROTO_SLIDES_TABLE,
    CREATE_RETRIEVED_CHUNKS_TABLE,
    CREATE_SLIDE_REVIEW_EVENTS_TABLE,
    CREATE_TABLES_TABLE,
    CREATE_TEXT_CHUNKS_TABLE,
    CREATE_TEXT_CHUNKS_VEC_TABLE,
    ImageMetadata,
)

if TYPE_CHECKING:
    from src.retriever import RetrievedItem


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
        """Creates tables and indexes.

        The ALTER TABLE migration block below is a temporary measure to keep
        existing development databases usable as the schema evolves. It will
        not be present in the final build — schema versioning or a clean
        migration tool will replace it.
        """
        statements = [
            CREATE_DOCUMENTS_TABLE,
            CREATE_IMAGES_TABLE,
            CREATE_TABLES_TABLE,
            CREATE_EQUATIONS_TABLE,
            CREATE_TEXT_CHUNKS_TABLE,
            CREATE_TEXT_CHUNKS_VEC_TABLE.format(vec_dimensions=self.config.vec_dimensions),
            CREATE_ARTIFACT_SEARCH_SOURCE_VIEW,
            CREATE_ARTIFACT_SEARCH_FTS_TABLE,
            CREATE_FTS_ROWID_MAP_TABLE,
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
                    if "document_context" not in doc_columns:
                        self._conn.execute("ALTER TABLE documents ADD COLUMN document_context TEXT;")

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
                if review_event_columns:
                    if "affected_slide_numbers" not in review_event_columns:
                        self._conn.execute(
                            "ALTER TABLE slide_review_events ADD COLUMN affected_slide_numbers TEXT;"
                        )
                    if "location" not in review_event_columns:
                        self._conn.execute(
                            "ALTER TABLE slide_review_events ADD COLUMN location TEXT;"
                        )
                    if "plan_number" not in review_event_columns and "plan_generation" not in review_event_columns:
                        self._conn.execute(
                            "ALTER TABLE slide_review_events "
                            "ADD COLUMN plan_number INTEGER NOT NULL DEFAULT 0;"
                        )

                # Rename legacy plan_generation -> plan_number on slide_review_events (column rename via rebuild).
                sre_info = self._conn.execute("PRAGMA table_info(slide_review_events)").fetchall()
                sre_names = {r["name"] for r in sre_info}
                if sre_names and "plan_generation" in sre_names and "plan_number" not in sre_names:
                    self._conn.execute("ALTER TABLE slide_review_events RENAME TO slide_review_events_old;")
                    self._conn.execute(CREATE_SLIDE_REVIEW_EVENTS_TABLE)
                    self._conn.execute("INSERT INTO slide_review_events SELECT * FROM slide_review_events_old;")
                    self._conn.execute("DROP TABLE slide_review_events_old;")

                rc_columns = [
                    info["name"]
                    for info in self._conn.execute("PRAGMA table_info(retrieved_chunks)").fetchall()
                ]
                if rc_columns and "plan_number" not in rc_columns and "plan_generation" not in rc_columns:
                    self._conn.execute("ALTER TABLE retrieved_chunks RENAME TO retrieved_chunks_old;")
                    self._conn.execute(CREATE_RETRIEVED_CHUNKS_TABLE)
                    self._conn.execute(
                        """
                        INSERT INTO retrieved_chunks (
                            row_id, session_id, call_id, kind, artifact_id, document_id, text_content,
                            score, rank, strategy, retrieved_at, agent_type, query, plan_number
                        )
                        SELECT
                            row_id, session_id, call_id, kind, artifact_id, document_id, text_content,
                            score, rank, strategy, retrieved_at, agent_type, query, 0
                        FROM retrieved_chunks_old;
                        """
                    )
                    self._conn.execute("DROP TABLE retrieved_chunks_old;")
                    for idx_sql in (
                        "CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_session_id "
                        "ON retrieved_chunks(session_id);",
                        "CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_call_id "
                        "ON retrieved_chunks(call_id);",
                        "CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_session_call "
                        "ON retrieved_chunks(session_id, call_id);",
                    ):
                        self._conn.execute(idx_sql)
                rcc_names = {
                    r["name"] for r in self._conn.execute("PRAGMA table_info(retrieved_chunks)").fetchall()
                }
                if rcc_names and "plan_generation" in rcc_names and "plan_number" not in rcc_names:
                    self._conn.execute("ALTER TABLE retrieved_chunks RENAME TO retrieved_chunks_old;")
                    self._conn.execute(CREATE_RETRIEVED_CHUNKS_TABLE)
                    self._conn.execute("INSERT INTO retrieved_chunks SELECT * FROM retrieved_chunks_old;")
                    self._conn.execute("DROP TABLE retrieved_chunks_old;")
                    for idx_sql in (
                        "CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_session_id "
                        "ON retrieved_chunks(session_id);",
                        "CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_call_id "
                        "ON retrieved_chunks(call_id);",
                        "CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_session_call "
                        "ON retrieved_chunks(session_id, call_id);",
                    ):
                        self._conn.execute(idx_sql)

                # Per-plan columns must exist before these indexes (migrations add columns on old DBs).
                sre_columns2 = [
                    info["name"]
                    for info in self._conn.execute("PRAGMA table_info(slide_review_events)").fetchall()
                ]
                if sre_columns2 and "plan_number" in sre_columns2:
                    self._conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_slide_review_events_session_plan "
                        "ON slide_review_events(session_id, plan_number);"
                    )
                rcc_names2 = [
                    info["name"] for info in self._conn.execute("PRAGMA table_info(retrieved_chunks)").fetchall()
                ]
                if rcc_names2 and "plan_number" in rcc_names2:
                    self._conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_session_plan "
                        "ON retrieved_chunks(session_id, plan_number);"
                    )
            except Exception as e:
                self._logger.log(f"[ResearchDatabase] Schema migration error: {e}")
        retrieval_ensure_artifact_search_index(self)

    def reset(self) -> None:
        """Drops all tables and recreates them."""
        with self._conn:
            for table in TABLE_NAMES:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            self._conn.execute("DROP TABLE IF EXISTS fts_rowid_map")
            self._conn.execute("DROP TABLE IF EXISTS schema_version")
            self._conn.execute("DROP TABLE IF EXISTS schema_migrations")
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

    def get_images_for_chunks(self, chunk_ids: list[str]) -> list[ImageMetadata]:
        return document.get_images_for_chunks(self, chunk_ids)

    def get_table(self, table_id: str):
        return document.get_table(self, table_id)

    def get_equation(self, equation_id: str):
        return document.get_equation(self, equation_id)

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

    def list_review_events(
        self, session_id: str, plan_number: int | None = None
    ) -> list[dict]:
        return slide.list_review_events(self, session_id, plan_number=plan_number)

    def delete_slides_not_in(self, slide_numbers: list[int]) -> None:
        return slide.delete_slides_not_in(self, slide_numbers)

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("Database disconnected.")
        return self._conn

    def knn_text_chunks_by_embedding(self, embedding_blob: bytes, k: int) -> list[dict[str, Any]]:
        return retrieval_knn_text_chunks_by_embedding(self, embedding_blob, k)

    def fetch_all_text_chunks_for_retrieval(self) -> list[dict[str, Any]]:
        return retrieval_fetch_all_text_chunks_for_retrieval(self)

    def fetch_all_tables_for_retrieval(self) -> list[dict[str, Any]]:
        return retrieval_fetch_all_tables_for_retrieval(self)

    def fetch_all_equations_for_retrieval(self) -> list[dict[str, Any]]:
        return retrieval_fetch_all_equations_for_retrieval(self)

    def fetch_all_images_for_retrieval(self) -> list[dict[str, Any]]:
        return retrieval_fetch_all_images_for_retrieval(self)

    def save_session_retrieval_batch(
        self,
        session_id: str,
        call_id: str,
        items: list[RetrievedItem],
        query: str,
        strategy: str,
        agent_type: str,
        plan_number: int = 0,
    ) -> None:
        return retrieval_save_session_retrieval_batch(
            self,
            session_id,
            call_id,
            items,
            query,
            strategy,
            agent_type,
            plan_number=plan_number,
        )

    def load_normalized_artifacts_for_keys(
        self, items: list[RetrievedItem]
    ) -> list[dict[str, Any]]:
        return retrieval_load_normalized_artifacts_for_keys(self, items)

    def load_normalized_artifacts_for_call(
        self, session_id: str, call_id: str
    ) -> list[dict[str, Any]]:
        return retrieval_load_normalized_artifacts_for_call(self, session_id, call_id)

    def load_normalized_artifacts_for_session(
        self, session_id: str, plan_number: int | None = None
    ) -> list[dict[str, Any]]:
        return retrieval_load_normalized_artifacts_for_session(
            self, session_id, plan_number=plan_number
        )

    def load_retrieved_chunks(self, session_id: str | None = None) -> list[dict[str, Any]]:
        return retrieval_load_retrieved_chunks(self, session_id)

    def rebuild_artifact_search_index(self) -> None:
        return retrieval_rebuild_artifact_search_index(self)

    def ensure_artifact_search_index(self) -> None:
        return retrieval_ensure_artifact_search_index(self)

    def query_artifact_search(self, query: str, k: int) -> list[dict[str, Any]]:
        return retrieval_query_artifact_search(self, query, k)
