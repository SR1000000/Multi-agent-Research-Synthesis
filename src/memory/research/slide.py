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


def delete_slides_not_in(db, slide_numbers: list[int]) -> None:
    """Deletes proto-slides whose slide_number is not in the given list."""
    if not slide_numbers:
        return
    with db._conn:
        placeholders = ",".join("?" * len(slide_numbers))
        db._conn.execute(
            f"DELETE FROM proto_slides WHERE slide_number NOT IN ({placeholders})",
            slide_numbers,
        )


def save_review_event(
    db,
    *,
    session_id: str,
    cycle_number: int,
    plan_number: int = 0,
    scope_type: str,
    scope_id: str,
    check_type: str,
    assignment_id: str | None = None,
    issue_code: str | None = None,
    severity: str | None = None,
    location: str | None = None,
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
                plan_number,
                scope_type,
                scope_id,
                check_type,
                assignment_id,
                issue_code,
                severity,
                location,
                fingerprint,
                rewrite_instruction_summary,
                affected_slide_numbers,
                decision
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                cycle_number,
                plan_number,
                scope_type,
                scope_id,
                check_type,
                assignment_id,
                issue_code,
                severity,
                location,
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


def list_review_events(
    db, session_id: str, plan_number: int | None = None
) -> list[dict]:
    if plan_number is not None:
        rows = db._conn.execute(
            """
            SELECT session_id, cycle_number, plan_number, scope_type, scope_id, check_type,
                   assignment_id, issue_code, severity, location, fingerprint,
                   rewrite_instruction_summary, affected_slide_numbers, decision, created_at
            FROM slide_review_events
            WHERE session_id = ? AND plan_number = ?
            ORDER BY cycle_number ASC, id ASC
            """,
            (session_id, plan_number),
        ).fetchall()
    else:
        rows = db._conn.execute(
            """
            SELECT session_id, cycle_number, plan_number, scope_type, scope_id, check_type,
                   assignment_id, issue_code, severity, location, fingerprint,
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


# ---------------------------------------------------------------------------
# Best-seen proto-slide snapshot
# ---------------------------------------------------------------------------

def _is_better_severity(
    current: dict[str, int],
    stored: dict[str, int],
) -> bool:
    """Return True iff *current* severity counts are strictly better than *stored*.

    Comparison is lexicographic by priority: critical → major → minor.
    A lower count at the highest non-equal severity level wins.  Equal counts
    return False so no unnecessary DB write is triggered.

    This is a pure function with no side-effects — it is the single extension
    point for richer scoring in future iterations.
    """
    for key in ("critical", "major", "minor"):
        c, s = current.get(key, 0), stored.get(key, 0)
        if c < s:
            return True
        if c > s:
            return False
    return False  # exactly equal — no update needed


def check_promote_best_slides(
    db,
    slides: list[ProtoSlide],
    severity_counts: dict[str, int],
    cycle_number: int,
    plan_number: int,
) -> bool:
    """Replace best_proto_slides with *slides* if *severity_counts* beats stored.

    Reads severity_snapshot from any existing best row (LIMIT 1) and compares
    via _is_better_severity.  On the very first call (empty table) the sentinel
    {"critical": 999, "major": 999, "minor": 999} ensures promotion always fires.

    Uses DELETE + INSERT inside a single transaction rather than INSERT OR REPLACE
    so that stale rows from a prior plan with a different slide count cannot
    survive into the new best set.

    Returns True if the promotion occurred, False if the stored set was retained.
    """
    row = db._conn.execute(
        "SELECT severity_snapshot FROM best_proto_slides LIMIT 1"
    ).fetchone()
    stored_counts: dict[str, int] = (
        json.loads(row["severity_snapshot"])
        if row
        else {"critical": 999, "major": 999, "minor": 999}
    )

    if not _is_better_severity(current=severity_counts, stored=stored_counts):
        return False

    snapshot_json = json.dumps(severity_counts)
    with db._conn:
        db._conn.execute("DELETE FROM best_proto_slides")
        db._conn.executemany(
            """
            INSERT INTO best_proto_slides
                (slide_number, content, chunk_references,
                 severity_snapshot, cycle_number, plan_number)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    slide.slide_number,
                    slide.content.model_dump_json(),
                    json.dumps(slide.chunk_references),
                    snapshot_json,
                    cycle_number,
                    plan_number,
                )
                for slide in slides
            ],
        )
    db._logger.log(
        f"[ResearchDatabase] best_proto_slides promoted: "
        f"plan={plan_number} cycle={cycle_number} counts={severity_counts}"
    )
    return True


def load_best_slides(db) -> list[ProtoSlide]:
    """Load all best-seen proto-slides ordered by slide_number ascending.

    Returns an empty list if no best set has been promoted yet.
    Deserialises content and chunk_references identically to load_slide().
    """
    rows = db._conn.execute(
        "SELECT slide_number, content, chunk_references "
        "FROM best_proto_slides ORDER BY slide_number ASC"
    ).fetchall()
    return [
        ProtoSlide(
            slide_number=row["slide_number"],
            content=SlideContent.model_validate_json(row["content"]),
            chunk_references=json.loads(row["chunk_references"]),
        )
        for row in rows
    ]


def load_best_severity_snapshot(db) -> dict[str, int] | None:
    """Return the severity counts of the current best set, or None if empty.

    Used at export time to log which cycle's slides are being exported.
    """
    row = db._conn.execute(
        "SELECT severity_snapshot FROM best_proto_slides LIMIT 1"
    ).fetchone()
    return json.loads(row["severity_snapshot"]) if row else None


def clear_best_proto_slides(db) -> None:
    """Delete all rows from best_proto_slides.

    Called on program start alongside clear_proto_slides() and
    clear_slide_review_events() so every run begins with a clean slate.
    """
    with db._conn:
        db._conn.execute("DELETE FROM best_proto_slides")
