from __future__ import annotations

import json
import sqlite3

from .schema import ProtoSlide, SlideContent


def save_slide(db, slide: ProtoSlide) -> None:
    """Persists a ProtoSlide to the research database, overwriting if it exists.

    On update, the row's prior content/chunk_references/updated_at are copied into
    previous_* columns so at most one revision back remains for debugging.
    """
    content_json = slide.content.model_dump_json()
    chunks_json = json.dumps(slide.chunk_references)

    with db._conn:
        db._conn.execute(
            """
            UPDATE proto_slides SET
                previous_content = content,
                previous_chunk_references = chunk_references,
                previous_updated_at = updated_at
            WHERE slide_number = ?
            """,
            (slide.slide_number,),
        )
        db._conn.execute(
            """
            INSERT INTO proto_slides (slide_number, content, chunk_references, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(slide_number) DO UPDATE SET
                content=excluded.content,
                chunk_references=excluded.chunk_references,
                updated_at=CURRENT_TIMESTAMP
            """,
            (slide.slide_number, content_json, chunks_json),
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
    affected_slide_numbers: list[int] | None = None,
    decision: str | None = None,
) -> None:
    """Persists a compact review event for recurrence tracking and auditability."""
    affected_json: str | None
    if affected_slide_numbers:
        affected_json = json.dumps(affected_slide_numbers)
    else:
        affected_json = None
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
                affected_slide_numbers,
                decision
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                affected_json,
                decision,
            ),
        )


def _row_affected_slide_numbers(row: sqlite3.Row) -> list[int] | None:
    raw = row["affected_slide_numbers"] if "affected_slide_numbers" in row.keys() else None
    if raw is None or raw == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    out: list[int] = []
    for x in parsed:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out or None


def list_review_events(db, session_id: str) -> list[dict]:
    rows = db._conn.execute(
        """
        SELECT session_id, cycle_number, scope_type, scope_id, check_type,
               assignment_id, issue_code, severity, fingerprint,
               rewrite_instruction_summary, affected_slide_numbers, decision, created_at
        FROM slide_review_events
        WHERE session_id = ?
        ORDER BY cycle_number ASC, id ASC
        """,
        (session_id,),
    ).fetchall()
    result: list[dict] = []
    for row in rows:
        d = dict(row)
        d["affected_slide_numbers"] = _row_affected_slide_numbers(row)
        result.append(d)
    return result
