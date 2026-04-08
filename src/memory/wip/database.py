import sqlite3
import json
from pathlib import Path
from typing import Optional, List

from src.logging.logger import AgentLogger
from src.memory.wip.schema import CREATE_PROTO_SLIDES_TABLE, ProtoSlide, SlideContent

class WIPDatabase:
    """
    SQLite implementation for the Work-In-Progress (WIP) storage.
    Handles persistent storage of proto-slides.
    """

    def __init__(self, db_path: Path | str = Path("data/wip.db")) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._logger = AgentLogger()
        self.connect()

    def connect(self) -> None:
        """Opens a connection and initializes the schema."""
        if self._conn is not None:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.setup()

    def disconnect(self) -> None:
        """Closes the basic connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "WIPDatabase":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def setup(self) -> None:
        """Creates tables for the WIP database."""
        with self._conn:
            self._conn.execute(CREATE_PROTO_SLIDES_TABLE)

    def reset(self) -> None:
        """Drops all tables and recreates them."""
        with self._conn:
            self._conn.execute("DROP TABLE IF EXISTS proto_slides")
        self.setup()

    def save_slide(self, slide: ProtoSlide) -> None:
        """Persists a ProtoSlide to the WIP database, overwriting if it exists."""
        content_json = slide.content.model_dump_json()
        chunks_json = json.dumps(slide.chunk_references)
        
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO proto_slides (slide_number, content, chunk_references, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(slide_number) DO UPDATE SET
                    content=excluded.content,
                    chunk_references=excluded.chunk_references,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (slide.slide_number, content_json, chunks_json)
            )
        self._logger.log(f"[WIPDatabase] Saved slide {slide.slide_number}")

    def load_slide(self, slide_number: int) -> Optional[ProtoSlide]:
        """Loads a ProtoSlide from the WIP database."""
        row = self._conn.execute(
            "SELECT * FROM proto_slides WHERE slide_number = ?", 
            (slide_number,)
        ).fetchone()
        
        if not row:
            return None
            
        content = SlideContent.model_validate_json(row["content"])
        chunk_refs = json.loads(row["chunk_references"])
        
        return ProtoSlide(
            slide_number=row["slide_number"],
            content=content,
            chunk_references=chunk_refs
        )
        
    def list_slide_numbers(self) -> List[int]:
        """Returns a list of all existing slide numbers ordered ascending."""
        rows = self._conn.execute("SELECT slide_number FROM proto_slides ORDER BY slide_number ASC").fetchall()
        return [row["slide_number"] for row in rows]

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("WIP Database disconnected.")
        return self._conn
