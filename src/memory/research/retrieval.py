"""SQL used by retrieval; kept separate from SQLiteDatabase for readability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence

from src.memory.research.document import build_search_text

if TYPE_CHECKING:
    from src.memory.research.database import ResearchDatabase
    from src.retriever import RetrievedItem


class ArtifactKey(Protocol):
    kind: str
    id: str
    document_id: str
    score: float | None

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
    artifact_search_fts.item_id,
    artifact_search_fts.document_id,
    artifact_search_fts.kind,
    artifact_search_fts.search_text,
    bm25(artifact_search_fts) AS score,
    tc.text AS chunk_text,
    tc.contextualized_text AS chunk_contextualized_text,
    tbl.content AS table_content,
    tbl.contextualized_text AS table_contextualized_text,
    tbl.caption AS table_caption,
    eq.text AS equation_text,
    eq.contextualized_text AS equation_contextualized_text,
    eq.caption AS equation_caption,
    img.storage_path AS image_storage_path,
    img.contextualized_text AS image_contextualized_text,
    img.caption AS image_caption
FROM artifact_search_fts
LEFT JOIN text_chunks tc
    ON artifact_search_fts.kind = 'chunk'
    AND tc.id = artifact_search_fts.item_id
LEFT JOIN tables tbl
    ON artifact_search_fts.kind = 'table'
    AND tbl.id = artifact_search_fts.item_id
LEFT JOIN equations eq
    ON artifact_search_fts.kind = 'equation'
    AND eq.id = artifact_search_fts.item_id
LEFT JOIN images img
    ON artifact_search_fts.kind = 'image'
    AND img.id = artifact_search_fts.item_id
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
            "SELECT id AS item_id, document_id, 'image' AS kind, contextualized_text, COALESCE(NULLIF(caption, ''), storage_path) AS raw_value FROM images"
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
    return [dict(row) for row in rows]


_NORMALIZED_FROM_KEYS = """
SELECT
    k.rank AS rank,
    k.kind AS kind,
    k.artifact_id AS artifact_id,
    k.document_id AS document_id,
    k.score AS score,
    CASE k.kind
        WHEN 'chunk' THEN tc.text
        WHEN 'table' THEN tbl.content
        WHEN 'equation' THEN eq.text
        WHEN 'image' THEN COALESCE(NULLIF(img.caption, ''), img.storage_path)
    END AS text,
    CASE k.kind
        WHEN 'chunk' THEN tc.contextualized_text
        WHEN 'table' THEN tbl.contextualized_text
        WHEN 'equation' THEN eq.contextualized_text
        WHEN 'image' THEN img.contextualized_text
    END AS contextualized_text,
    CASE k.kind
        WHEN 'table' THEN tbl.caption
        WHEN 'equation' THEN eq.caption
        WHEN 'image' THEN img.caption
        ELSE ''
    END AS caption
FROM keys k
LEFT JOIN text_chunks tc ON k.kind = 'chunk' AND tc.id = k.artifact_id
LEFT JOIN tables tbl ON k.kind = 'table' AND tbl.id = k.artifact_id
LEFT JOIN equations eq ON k.kind = 'equation' AND eq.id = k.artifact_id
LEFT JOIN images img ON k.kind = 'image' AND img.id = k.artifact_id
"""


def _normalized_rows_from_keys_sql(n: int) -> str:
    placeholders = ",".join(["(?, ?, ?, ?, ?)"] * n)
    return (
        "WITH keys(rank, kind, artifact_id, document_id, score) AS (VALUES "
        + placeholders
        + ") "
        + _NORMALIZED_FROM_KEYS
        + " ORDER BY k.rank"
    )


def load_normalized_artifacts_for_keys(
    db: ResearchDatabase,
    items: Sequence[ArtifactKey],
) -> list[dict[str, Any]]:
    if not items:
        return []
    batch_size = 180
    normalized_rows: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        params: list[Any] = []
        for offset, item in enumerate(batch):
            params.extend(
                [
                    start + offset,
                    item.kind,
                    item.id,
                    item.document_id,
                    item.score,
                ]
            )
        sql = _normalized_rows_from_keys_sql(len(batch))
        rows = db.connection.execute(sql, params).fetchall()
        normalized_rows.extend(dict(r) for r in rows)
    return normalized_rows


_LEDGER_KEYS_CALL = """
WITH keys AS (
    SELECT
        s.rank AS rank,
        s.kind AS kind,
        s.artifact_id AS artifact_id,
        s.document_id AS document_id,
        s.score AS score
    FROM retrieved_chunks s
    WHERE s.session_id = ? AND s.call_id = ?
)
"""

_LEDGER_KEYS_SESSION = """
WITH keys AS (
    SELECT
        s.call_id AS call_id,
        s.rank AS rank,
        s.kind AS kind,
        s.artifact_id AS artifact_id,
        s.document_id AS document_id,
        s.score AS score
    FROM retrieved_chunks s
    WHERE s.session_id = ?
)
"""

_SESSION_OUTER_SELECT = """
SELECT
    k.call_id AS call_id,
    k.rank AS rank,
    k.kind AS kind,
    k.artifact_id AS artifact_id,
    k.document_id AS document_id,
    k.score AS score,
    CASE k.kind
        WHEN 'chunk' THEN tc.text
        WHEN 'table' THEN tbl.content
        WHEN 'equation' THEN eq.text
        WHEN 'image' THEN COALESCE(NULLIF(img.caption, ''), img.storage_path)
    END AS text,
    CASE k.kind
        WHEN 'chunk' THEN tc.contextualized_text
        WHEN 'table' THEN tbl.contextualized_text
        WHEN 'equation' THEN eq.contextualized_text
        WHEN 'image' THEN img.contextualized_text
    END AS contextualized_text,
    CASE k.kind
        WHEN 'table' THEN tbl.caption
        WHEN 'equation' THEN eq.caption
        WHEN 'image' THEN img.caption
        ELSE ''
    END AS caption
FROM keys k
LEFT JOIN text_chunks tc ON k.kind = 'chunk' AND tc.id = k.artifact_id
LEFT JOIN tables tbl ON k.kind = 'table' AND tbl.id = k.artifact_id
LEFT JOIN equations eq ON k.kind = 'equation' AND eq.id = k.artifact_id
LEFT JOIN images img ON k.kind = 'image' AND img.id = k.artifact_id
ORDER BY k.call_id, k.rank, k.artifact_id
"""


def load_normalized_artifacts_for_call(
    db: ResearchDatabase,
    session_id: str,
    call_id: str,
) -> list[dict[str, Any]]:
    sql = _LEDGER_KEYS_CALL + _NORMALIZED_FROM_KEYS + " ORDER BY k.rank, k.artifact_id"
    rows = db.connection.execute(sql, (session_id, call_id)).fetchall()
    return [dict(r) for r in rows]


def load_normalized_artifacts_for_session(
    db: ResearchDatabase,
    session_id: str,
) -> list[dict[str, Any]]:
    sql = _LEDGER_KEYS_SESSION + _SESSION_OUTER_SELECT
    rows = db.connection.execute(sql, (session_id,)).fetchall()
    return [dict(r) for r in rows]


def save_session_retrieval_batch(
    db: ResearchDatabase,
    session_id: str,
    call_id: str,
    items: Sequence[ArtifactKey],
    query: str,
    strategy: str,
    agent_type: str,
) -> None:
    if not items:
        return
    sql = """
        INSERT OR IGNORE INTO retrieved_chunks (
            session_id, call_id, kind, artifact_id, document_id, text_content,
            score, rank, strategy, agent_type, query, retrieved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """
    with db.connection:
        for rank, item in enumerate(items):
            db.connection.execute(
                sql,
                (
                    session_id,
                    call_id,
                    item.kind,
                    item.id,
                    item.document_id,
                    None,
                    item.score,
                    rank,
                    strategy,
                    agent_type,
                    query,
                ),
            )
    db._logger.log(
        f"[ResearchDatabase] Saved {len(items)} retrieved rows call_id={call_id} session_id={session_id}"
    )


def load_retrieved_chunks(
    db: ResearchDatabase,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    sql = (
        "SELECT artifact_id AS id, kind, document_id, text_content, score, "
        "call_id, session_id FROM retrieved_chunks"
    )
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
            "call_id": row["call_id"],
            "session_id": row["session_id"],
        }
        for row in rows
    ]
