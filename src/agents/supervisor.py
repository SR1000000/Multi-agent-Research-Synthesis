"""
SupervisorAgent — DORMANT
=========================
Not connected to the active graph. Preserved for future reactivation when
the Critic/Writer review cycle is re-introduced.
"""
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel

from src.state import ResearchState
from src.agents.base import BaseLLMAgent


# ---------------------------------------------------------------------------
# Local type stub (formerly in state.py)
# ---------------------------------------------------------------------------

class SupervisorOutput(BaseModel):
    decision:  str   # "accept" | "revise" | "replan"
    reasoning: str
    feedback:  str = ""


# ---------------------------------------------------------------------------
# Agent (dormant)
# ---------------------------------------------------------------------------

class SupervisorAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__("supervisor")

    def run(self, state: ResearchState) -> Command:
        self._set_session_id(state)
        draft    = state.get("draft") or {}
        critique = state.get("critique")
        rev_hist = state.get("revision_history", [])
        rep_hist = state.get("replan_history", [])

        history_block = ""
        if rev_hist:
            history_block += f"Revision history ({len(rev_hist)} cycles):\n"
            history_block += "\n".join(
                f"  Cycle {i + 1}: {h}" for i, h in enumerate(rev_hist)
            )
            history_block += "\n"
        if rep_hist:
            history_block += f"Replan history ({len(rep_hist)} cycles):\n"
            history_block += "\n".join(
                f"  Cycle {i + 1}: {h}" for i, h in enumerate(rep_hist)
            )

        issues_str = ""
        if critique:
            issues_str = "\n".join(
                f"[{i.severity.upper()}] {i.id} @ {i.location}: {i.description}"
                for i in critique.issues
            )

        user = (
            f"Query:\n{state['query']}\n\n"
            f"Draft (v{draft.get('version', '?')}):\n{draft.get('document', '')}\n\n"
            f"Critique Summary:\n{critique.summary if critique else '(none)'}\n\n"
            f"Current Issues:\n{issues_str}\n\n"
            + (f"Prior Cycle History:\n{history_block}\n" if history_block else "")
            + "Decide: accept, revise, or replan. Include a feedback string."
        )

        turns  = [{"role": "user", "content": user}]
        result: SupervisorOutput = self._call(turns, schema=SupervisorOutput)

        updates: dict = {
            "messages": [f"[supervisor] {result.decision}: {result.reasoning}"],
        }

        if result.decision == "revise":
            updates["revision_history"] = [result.feedback]
            updates["revision_count"]   = state.get("revision_count", 0) + 1
            return Command(update=updates, goto="writer")
        elif result.decision == "replan":
            updates["replan_history"] = [result.feedback]
            updates["replan_count"]   = state.get("replan_count", 0) + 1
            updates["revision_count"] = 0
            return Command(update=updates, goto="planner")
        else:
            return Command(update=updates, goto=END)


def supervisor_node(state: ResearchState) -> Command:
    return SupervisorAgent().run(state)
