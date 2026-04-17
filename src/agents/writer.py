"""
WriterAgent — DORMANT
=====================
Not connected to the active graph. Preserved for future reactivation when
the Critic/Writer review cycle is re-introduced.
"""
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.types import Command
from pydantic import BaseModel

from src.state import ResearchState
from src.agents.base import BaseLLMAgent


# ---------------------------------------------------------------------------
# Local type stubs (types formerly in state.py, kept here for dormant logic)
# ---------------------------------------------------------------------------

class Draft(TypedDict):
    version:    int
    document:   str
    word_count: int
    action:     str   # 'initial' | 'revision'
    created_at: str


def _render_history(history: list[str], kind: str) -> str:
    if not history:
        return ""
    lines = [f"PRIOR {kind.upper()} HISTORY — do not repeat these mistakes:"]
    for i, entry in enumerate(history):
        lines.append(f"  Cycle {i + 1}: {entry}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent (dormant)
# ---------------------------------------------------------------------------

class WriterAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__("writer")

    def run(self, state: ResearchState) -> Command:
        self._set_session_id(state)
        doc_ctx     = state.get("document_context", "")
        is_revision = len(state.get("revision_history", [])) > 0

        initial_user = (
            f"Context from user given documents:\n{doc_ctx}\n\n"
            "By synthesizing the context, write the full draft."
        )

        if not is_revision:
            turns = [{"role": "user", "content": initial_user}]
        else:
            history_str = _render_history(state.get("revision_history", []), "revision")
            draft_doc   = (state.get("draft") or {}).get("document", "")
            turns = [
                {"role": "user",      "content": initial_user},
                {"role": "assistant", "content": draft_doc},
                {"role": "user",      "content": f"Revise the draft.\n\n{history_str}"},
            ]

        document = self._call(turns)
        existing  = state.get("draft") or {}
        version   = existing.get("version", 0) + 1
        draft = Draft(
            version=version,
            document=document,
            word_count=len(document.split()),
            action="revision" if is_revision else "initial",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        msg = f"[writer] draft v{version} ({draft['word_count']} words)"
        return Command(update={"draft": draft, "messages": [msg]})


def writer_node(state: ResearchState) -> Command:
    return WriterAgent().run(state)
