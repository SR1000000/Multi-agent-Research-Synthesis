from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.util import VEC_DIMENSIONS


@dataclass
class StorageConfig:
    """Configuration for the document storage and vector database."""
    db_path: Path = Path("data/research.db")
    vec_dimensions: int = VEC_DIMENSIONS
    auto_create_dirs: bool = True
    
    # SQLite connection parameters
    check_same_thread: bool = False
    isolation_level: str | None = None
    journal_mode: str = "WAL"
    foreign_keys: bool = True


DEFAULT_CONFIG = StorageConfig()
TABLE_NAMES = [
    "images",
    "tables",
    "equations",
    "text_chunks",
    "text_chunks_vec",
    "artifact_search_fts",
    "documents",
    "proto_slides",
    "best_proto_slides",      # immutable best-seen snapshot; cleared on program start
    "retrieved_chunks",
    "slide_review_events"
]