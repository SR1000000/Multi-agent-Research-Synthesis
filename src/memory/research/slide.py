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


def clear_slide_review_events(db) -> None:
    """Deletes all slide review events (critic/supervisor audit trail) from the research database."""
    with db._conn:
        db._conn.execute("DELETE FROM slide_review_events")


def save_review_event(
    db,
    *,
    session_id: str,
    cycle_number: int,
    scope_type: str,
    scope_id: str,
    check_type: str,
    assignment_id: str | None = None,
    issue_code: str | None = None,
    severity: str | None = None,
    fingerprint: str | None = None,
    rewrite_instruction_summary: str | None = None,
    decision: str | None = None,
) -> None:
    """Persists a compact review event for recurrence tracking and auditability."""
    with db._conn:
        db._conn.execute(
            """
            INSERT INTO slide_review_events (
                session_id,
                cycle_number,
                scope_type,
                scope_id,
                check_type,
                assignment_id,
                issue_code,
                severity,
                fingerprint,
                rewrite_instruction_summary,
                decision
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                cycle_number,
                scope_type,
                scope_id,
                check_type,
                assignment_id,
                issue_code,
                severity,
                fingerprint,
                rewrite_instruction_summary,
                decision,
            ),
        )


def list_review_events(db, session_id: str) -> list[dict]:
    rows = db._conn.execute(
        """
        SELECT session_id, cycle_number, scope_type, scope_id, check_type,
               assignment_id, issue_code, severity, fingerprint,
               rewrite_instruction_summary, decision, created_at
        FROM slide_review_events
        WHERE session_id = ?
        ORDER BY cycle_number ASC, id ASC
        """,
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]
