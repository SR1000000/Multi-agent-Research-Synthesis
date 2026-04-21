from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path
import sys

import sqlite_vec

# Add project root to sys.path
root = Path(__file__).resolve().parents[3]  # Goes up from scripts -> memory -> src -> project_root
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from src.memory.research.schema import CREATE_INDEXES, CREATE_RETRIEVED_CHUNKS_TABLE


def _build_search_text(contextualized_text: str | None, raw_value: str | None) -> str:
    ctx = (contextualized_text or "").strip()
    raw = (raw_value or "").strip()
    if ctx and raw:
        return f"{ctx}\n\n{raw}"
    return ctx or raw


def _connect_sqlite_with_vec(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
    return conn


MIGRATIONS: list[tuple[int, str]] = [
    (1, "ALTER TABLE documents ADD COLUMN run_id TEXT"),
    (2, "ALTER TABLE documents ADD COLUMN schema TEXT"),
    (3, "ALTER TABLE documents ADD COLUMN paper_metadata TEXT"),
]


def _assert_required_tables(conn: sqlite3.Connection) -> None:
    required_tables = {"documents", "artifact_search_fts"}
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    ).fetchall()
    available = {row[0] for row in rows}
    missing = sorted(required_tables - available)
    if missing:
        raise RuntimeError(
            "Database is missing required tables for migration: "
            + ", ".join(missing)
        )


def _ensure_schema_version_table(conn: sqlite3.Connection) -> int:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _apply_column_migrations(conn: sqlite3.Connection, current_version: int) -> int:
    latest_version = current_version
    for version, sql in MIGRATIONS:
        if version <= current_version:
            continue
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        latest_version = version
    return latest_version


def _ensure_rowid_map_document_id(conn: sqlite3.Connection) -> None:
    existing_tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "fts_rowid_map" not in existing_tables:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fts_rowid_map (
                item_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                UNIQUE(item_id, kind)
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO fts_rowid_map(rowid, item_id, document_id, kind)
            SELECT rowid, item_id, document_id, kind
            FROM artifact_search_fts
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fts_rowid_map_document_id ON fts_rowid_map(document_id)"
        )
        return

    rowid_map_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(fts_rowid_map)").fetchall()
    }
    if "document_id" in rowid_map_columns:
        return
    conn.execute("DROP TABLE IF EXISTS fts_rowid_map_new")
    conn.execute(
        """
        CREATE TABLE fts_rowid_map_new (
            item_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            UNIQUE(item_id, kind)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO fts_rowid_map_new(rowid, item_id, document_id, kind)
        SELECT m.rowid, m.item_id, f.document_id, m.kind
        FROM fts_rowid_map m
        JOIN artifact_search_fts f ON f.rowid = m.rowid
        """
    )
    conn.execute("DROP TABLE fts_rowid_map")
    conn.execute("ALTER TABLE fts_rowid_map_new RENAME TO fts_rowid_map")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fts_rowid_map_document_id ON fts_rowid_map(document_id)"
    )


def _reset_retrieved_chunks(conn: sqlite3.Connection) -> None:
    """Session retrieval log only: DROP and recreate from schema (all prior log rows lost)."""
    conn.execute("DROP TABLE IF EXISTS retrieved_chunks")
    conn.execute(CREATE_RETRIEVED_CHUNKS_TABLE)
    for stmt in CREATE_INDEXES[-3:]:
        conn.execute(stmt)


def _validate_fts_consistency(conn: sqlite3.Connection) -> None:
    source_count = conn.execute(
        "SELECT COUNT(*) FROM artifact_search_source"
    ).fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM artifact_search_fts").fetchone()[0]
    if source_count != fts_count:
        raise RuntimeError(
            f"FTS/source row count mismatch after migration: source={source_count}, fts={fts_count}"
        )


def _rebuild_artifact_search_index(conn: sqlite3.Connection) -> None:
    artifact_rows = []
    artifact_rows.extend(
        conn.execute(
            "SELECT id AS item_id, document_id, 'chunk' AS kind, contextualized_text, text AS raw_value FROM text_chunks"
        ).fetchall()
    )
    artifact_rows.extend(
        conn.execute(
            "SELECT id AS item_id, document_id, 'table' AS kind, contextualized_text, content AS raw_value FROM tables"
        ).fetchall()
    )
    artifact_rows.extend(
        conn.execute(
            "SELECT id AS item_id, document_id, 'equation' AS kind, contextualized_text, text AS raw_value FROM equations"
        ).fetchall()
    )
    artifact_rows.extend(
        conn.execute(
            "SELECT id AS item_id, document_id, 'image' AS kind, contextualized_text, COALESCE(NULLIF(caption, ''), storage_path) AS raw_value FROM images"
        ).fetchall()
    )

    conn.execute("DELETE FROM fts_rowid_map")
    conn.execute("DELETE FROM artifact_search_fts")
    for row in artifact_rows:
        search_text = _build_search_text(row["contextualized_text"], row["raw_value"])
        if not search_text:
            continue
        mapping_cursor = conn.execute(
            "INSERT INTO fts_rowid_map(item_id, document_id, kind) VALUES (?, ?, ?)",
            (row["item_id"], row["document_id"], row["kind"]),
        )
        rowid = mapping_cursor.lastrowid
        conn.execute(
            """
            INSERT INTO artifact_search_fts(rowid, item_id, document_id, kind, search_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (rowid, row["item_id"], row["document_id"], row["kind"], search_text),
        )


def migrate(db_path: Path, create_backup: bool = True) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    if create_backup:
        backup_path = db_path.with_suffix(db_path.suffix + ".bak")
        shutil.copy2(db_path, backup_path)
        print(f"Backup created at: {backup_path}")

    conn = _connect_sqlite_with_vec(db_path)
    try:
        _assert_required_tables(conn)
        with conn:
            current_version = _ensure_schema_version_table(conn)
            latest_version = _apply_column_migrations(conn, current_version)
            _ensure_rowid_map_document_id(conn)
            _reset_retrieved_chunks(conn)
            _rebuild_artifact_search_index(conn)

        # Post-commit sanity check: validates persisted row counts after migration transaction.
        _validate_fts_consistency(conn)
        print(f"Migration complete. Schema version is now: {latest_version}")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-time migration for research.db schema and FTS mapping. "
            "Also drops and recreates retrieved_chunks (session log only)."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/research.db"),
        help="Path to research SQLite database.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable pre-migration backup creation.",
    )
    args = parser.parse_args()
    migrate(args.db_path.resolve(), create_backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
