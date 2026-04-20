"""SQL used by retrieval; kept separate from SQLiteDatabase for readability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.memory.research.document import build_search_text

if TYPE_CHECKING:
    from src.memory.research.database import ResearchDatabase

KNN_TEXT_CHUNKS_SQL = """
WITH knn AS (
    SELECT chunk_id, distance
    FROM text_chunks_vec
    WHERE embedding MATCH ? AND k = ?
)
SELECT tc.id, tc.document_id, tc.text, tc.contextualized_text, knn.distance AS dist
FROM knn
JOIN text_chunks tc ON tc.id = knn.chunk_id
ORDER BY knn.distance
"""

FETCH_ALL_TEXT_CHUNKS_SQL = (
    "SELECT id, document_id, text, contextualized_text FROM text_chunks"
)

FETCH_ALL_TABLES_SQL = (
    "SELECT id, document_id, content, contextualized_text FROM tables"
)

FETCH_ALL_EQUATIONS_SQL = (
    "SELECT id, document_id, text, contextualized_text FROM equations"
)

FETCH_ALL_IMAGES_SQL = (
    "SELECT id, document_id, caption, storage_path, contextualized_text FROM images"
)

# compared to FTS row count to detect drift/missing historical index rows.
COUNT_ARTIFACT_ROWS_SQL = "SELECT COUNT(*) AS row_count FROM artifact_search_source"
COUNT_ARTIFACT_SEARCH_ROWS_SQL = "SELECT COUNT(*) AS row_count FROM artifact_search_fts"

# Keyword retrieval through FTS index.
# `MATCH` executes against FTS index; `bm25()` provides rank score (lower is better).
QUERY_ARTIFACT_SEARCH_SQL = """
SELECT
    item_id,
    document_id,
    kind,
    search_text,
    bm25(artifact_search_fts) AS score
FROM artifact_search_fts
WHERE artifact_search_fts MATCH ?
ORDER BY score
LIMIT ?
"""

# one-shot rebuild primitives used only when guard detects index drift.
# normal writes are trigger-maintained; rebuild is startup/self-heal fallback.
DELETE_ALL_ARTIFACT_SEARCH_SQL = "DELETE FROM artifact_search_fts"
INSERT_ALL_ARTIFACTS_FTS_SQL = """
INSERT INTO artifact_search_fts(item_id, document_id, kind, search_text)
SELECT item_id, document_id, kind, search_text
FROM artifact_search_source
WHERE search_text IS NOT NULL AND TRIM(search_text) <> ''
"""


def knn_text_chunks_by_embedding(
    db: ResearchDatabase,
    embedding_blob: bytes,
    k: int,
) -> list[dict[str, Any]]:
    rows = db.connection.execute(KNN_TEXT_CHUNKS_SQL, (embedding_blob, k)).fetchall()
    return [dict(r) for r in rows]


def fetch_all_text_chunks_for_retrieval(db: ResearchDatabase) -> list[dict[str, Any]]:
    rows = db.connection.execute(FETCH_ALL_TEXT_CHUNKS_SQL).fetchall()
    return [dict(r) for r in rows]


def fetch_all_tables_for_retrieval(db: ResearchDatabase) -> list[dict[str, Any]]:
    rows = db.connection.execute(FETCH_ALL_TABLES_SQL).fetchall()
    return [dict(r) for r in rows]


def fetch_all_equations_for_retrieval(db: ResearchDatabase) -> list[dict[str, Any]]:
    rows = db.connection.execute(FETCH_ALL_EQUATIONS_SQL).fetchall()
    return [dict(r) for r in rows]


def fetch_all_images_for_retrieval(db: ResearchDatabase) -> list[dict[str, Any]]:
    rows = db.connection.execute(FETCH_ALL_IMAGES_SQL).fetchall()
    return [dict(r) for r in rows]


def rebuild_artifact_search_index(db: ResearchDatabase) -> None:
    artifact_rows = []
    artifact_rows.extend(
        db.connection.execute(
            "SELECT id AS item_id, document_id, 'chunk' AS kind, contextualized_text, text AS raw_value FROM text_chunks"
        ).fetchall()
    )
    artifact_rows.extend(
        db.connection.execute(
            "SELECT id AS item_id, document_id, 'table' AS kind, contextualized_text, content AS raw_value FROM tables"
        ).fetchall()
    )
    artifact_rows.extend(
        db.connection.execute(
            "SELECT id AS item_id, document_id, 'equation' AS kind, contextualized_text, text AS raw_value FROM equations"
        ).fetchall()
    )
    artifact_rows.extend(
        db.connection.execute(
            "SELECT id AS item_id, document_id, 'image' AS kind, contextualized_text, COALESCE(caption, storage_path) AS raw_value FROM images"
        ).fetchall()
    )

    with db.connection:
        db.connection.execute("DELETE FROM fts_rowid_map")
        db.connection.execute(DELETE_ALL_ARTIFACT_SEARCH_SQL)
        for row in artifact_rows:
            search_text = build_search_text(
                row["contextualized_text"],
                row["raw_value"],
            )
            if not search_text:
                continue
            mapping_cursor = db.connection.execute(
                "INSERT INTO fts_rowid_map(item_id, document_id, kind) VALUES (?, ?, ?)",
                (row["item_id"], row["document_id"], row["kind"]),
            )
            rowid = mapping_cursor.lastrowid
            db.connection.execute(
                """
                INSERT INTO artifact_search_fts(rowid, item_id, document_id, kind, search_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rowid, row["item_id"], row["document_id"], row["kind"], search_text),
            )


def ensure_artifact_search_index(db: ResearchDatabase) -> None:
    source_count = db.connection.execute("SELECT COUNT(*) FROM artifact_search_source").fetchone()[0]
    fts_count = db.connection.execute("SELECT COUNT(*) FROM artifact_search_fts").fetchone()[0]
    if source_count != fts_count:
        rebuild_artifact_search_index(db)


def query_artifact_search(
    db: ResearchDatabase,
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    rows = db.connection.execute(QUERY_ARTIFACT_SEARCH_SQL, (query, k)).fetchall()
    return [
        {
            "item_id": row["item_id"],
            "document_id": row["document_id"],
            "kind": row["kind"],
            "search_text": row["search_text"],
            "bm25_score": float(row["score"]) if row["score"] is not None else None,
        }
        for row in rows
    ]


def save_retrieved_chunk(
    db: ResearchDatabase,
    item_id: str,
    kind: str,
    document_id: str,
    text_content: str,
    score: float | None,
    session_id: str,
    agent_type: str,
    query: str,
) -> None:
    with db.connection:
        db.connection.execute(
            """
            INSERT OR REPLACE INTO retrieved_chunks
            (id, kind, document_id, text_content, score, session_id, agent_type, query, retrieved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                item_id,
                kind,
                document_id,
                text_content,
                score,
                session_id,
                agent_type,
                query,
            ),
        )
    db._logger.log(f"[ResearchDatabase] Saved retrieved chunk {item_id}")


def load_retrieved_chunks(
    db: ResearchDatabase,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT id, kind, document_id, text_content, score FROM retrieved_chunks"
    params: tuple[Any, ...] = ()
    if session_id:
        sql += " WHERE session_id = ?"
        params = (session_id,)
    rows = db.connection.execute(sql, params).fetchall()
    return [
        {
            "id": row["id"],
            "kind": row["kind"],
            "document_id": row["document_id"],
            "text_content": row["text_content"],
            "score": row["score"],
        }
        for row in rows
    ]
