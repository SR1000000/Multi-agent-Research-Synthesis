from __future__ import annotations

import json

from .schema import ProtoSlide, SlideContent


def save_slide(db, slide: ProtoSlide) -> None:
    """Persists a ProtoSlide to the research database, overwriting if it exists."""
    content_json = slide.content.model_dump_json()
    chunks_json = json.dumps(slide.chunk_references)

    with db._conn:
        db._conn.execute(
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
    db._logger.log(f"[ResearchDatabase] Saved slide {slide.slide_number}")


def load_slide(db, slide_number: int) -> ProtoSlide | None:
    """Loads a ProtoSlide from the research database."""
    row = db._conn.execute(
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
        chunk_references=chunk_refs,
    )


def list_slide_numbers(db) -> list[int]:
    """Returns a list of all existing slide numbers ordered ascending."""
    rows = db._conn.execute("SELECT slide_number FROM proto_slides ORDER BY slide_number ASC").fetchall()
    return [row["slide_number"] for row in rows]


def clear_proto_slides(db) -> None:
    """Deletes all proto-slides from the research database."""
    with db._conn:
        db._conn.execute("DELETE FROM proto_slides")
