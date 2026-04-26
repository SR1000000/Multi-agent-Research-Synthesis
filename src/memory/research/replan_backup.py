"""Replan debug backup — optional snapshot of ``research.db`` before a full replan.

**Development and debugging only** — set ``ENABLE_REPLAN_DEBUG_BACKUP`` to False or remove
this module in production. Failures are logged and do not block the pipeline.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.logging.logger import AgentLogger

if TYPE_CHECKING:
    from src.memory.research.database import ResearchDatabase

_logger = AgentLogger()

# Set to False to disable backup writes in production.
ENABLE_REPLAN_DEBUG_BACKUP = True


def backup_replan_debug_snapshot(
    research_db: "ResearchDatabase",
    *,
    plan_number: int,
    session_id: str,
    graph_metadata: dict[str, Any],
    presentation_plan_json: str | None,
) -> Path | None:
    """Copy the live research SQLite file to ``<db_dir>/backup_plan{plan_number}.db`` and
    append a small ``replan_debug_metadata`` table with JSON for extra context.

    Uses SQLite's ``backup`` API on the same connection the app already holds, so a
    consistent snapshot is taken (WAL readers see a checkpoint-style copy).
    """
    if not ENABLE_REPLAN_DEBUG_BACKUP:
        return None
    out_path = research_db.config.db_path.parent / f"backup_plan{plan_number}.db"
    try:
        research_db.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        dest = sqlite3.connect(str(out_path))
        try:
            research_db.connection.backup(dest)
            dest.execute(
                "CREATE TABLE IF NOT EXISTS replan_debug_metadata (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
            )
            payload = {
                "plan_number": plan_number,
                "session_id": session_id,
                "presentation_plan": presentation_plan_json,
                "graph": graph_metadata,
            }
            dest.execute(
                "INSERT INTO replan_debug_metadata (k, v) VALUES (?, ?)",
                ("snapshot", json.dumps(payload, default=str)),
            )
            dest.commit()
        finally:
            dest.close()
        _logger.log(f"[replan_backup] Wrote {out_path}", level="info")
        return out_path
    except Exception as exc:  # noqa: BLE001
        _logger.log(f"[replan_backup] Non-fatal backup failure: {exc}", level="warning")
        return None
