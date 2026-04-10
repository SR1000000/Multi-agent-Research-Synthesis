"""SQL used by retrieval; kept separate from SQLiteDatabase for readability."""

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
    "SELECT id, document_id, caption, storage_path FROM images"
)
