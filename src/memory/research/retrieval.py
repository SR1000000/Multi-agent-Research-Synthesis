"""SQL used by retrieval; kept separate from SQLiteDatabase for readability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.retriever.types import RetrievedItem

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


def save_retrieved_chunk(
    db: ResearchDatabase,
    item: RetrievedItem,
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
    db._logger.log(f"[ResearchDatabase] Saved retrieved chunk {item.id}")


def load_retrieved_chunks(
    db: ResearchDatabase,
    session_id: str | None = None,
) -> list[RetrievedItem]:
    sql = "SELECT id, kind, document_id, text_content, score FROM retrieved_chunks"
    params: tuple[Any, ...] = ()
    if session_id:
        sql += " WHERE session_id = ?"
        params = (session_id,)
    rows = db.connection.execute(sql, params).fetchall()
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
