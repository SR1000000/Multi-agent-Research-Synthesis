"""
CriticAgent — DORMANT
=====================
Not connected to the active graph. Preserved for future reactivation when
the Critic review cycle is re-introduced.
"""
from typing import List

from langgraph.types import Command
from pydantic import BaseModel, Field

from src.state import ResearchState
from src.agents.base import BaseLLMAgent


# ---------------------------------------------------------------------------
# Local type stubs (types formerly in state.py, kept here for dormant logic)
# ---------------------------------------------------------------------------

class IssueItem(BaseModel):
    id:          str = Field(description="e.g. ISS_001")
    location:    str
    type:        str   # factual_inaccuracy | hallucination | unsupported_claim | logical_gap | structural | clarity | contradiction
    severity:    str   # critical | major | minor
    description: str


class CritiqueOutput(BaseModel):
    summary: str
    issues:  List[IssueItem]


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

class CriticAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__("critic")

    def run(self, state: ResearchState) -> Command:
        self._set_session_id(state)
        draft     = state.get("draft") or {}
        draft_str = draft.get("document", "")
        rev_count = state.get("revision_count", 0)

        rev_hist    = _render_history(state.get("revision_history", []), "revision")
        rep_hist    = _render_history(state.get("replan_history", []), "replan")
        history_str = f"{rev_hist}\n{rep_hist}".strip()

        user = (
            f"Current Cycle: {rev_count + 1}\n"
            f"Prior History:\n{history_str or 'First review cycle.'}\n\n"
            f"Current Draft:\n{draft_str}\n\n"
            "Identify remaining issues. Mark recurring ones. Assign severity."
        )
        turns  = [{"role": "user", "content": user}]
        result: CritiqueOutput = self._call(turns, schema=CritiqueOutput)
        n = len(result.issues)
        c = sum(1 for i in result.issues if i.severity == "critical")
        msg = f"[critic] {n} issues (critical={c})"
        return Command(update={"critique": result, "messages": [msg]})


def critic_node(state: ResearchState) -> Command:
    return CriticAgent().run(state)
