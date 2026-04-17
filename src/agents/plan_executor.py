"""
PlanExecutorAgent
=================
Pure dispatcher + retry loop. No LLM calls, no section detection.

First call (slides_written is empty in state):
  - Reads slide_groups from the PresentationPlan
  - Fans out one Send("slide_writer") per group
  - Records which group indices were dispatched

Subsequent calls (after Slide Writers complete and loop back):
  - Reads slides_written counts from state
  - Re-dispatches any group that produced 0 slides (up to MAX_RETRIES_PER_GROUP)
  - Proceeds to END when all groups have produced slides or retry cap is reached
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, Send

from src.state import PresentationPlan, ResearchState, SlideGroup

MAX_RETRIES_PER_GROUP = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_chunk_ids(group: SlideGroup) -> list[str]:
    """Collect the union of source_chunk_ids across all blueprints in a group, preserving order."""
    seen:   set[str]  = set()
    result: list[str] = []
    for bp in group.slide_blueprints:
        for cid in bp.source_chunk_ids:
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
    return result


def _blueprints_as_dicts(group: SlideGroup) -> list[dict]:
    return [bp.model_dump() for bp in group.slide_blueprints]


def _build_send(group: SlideGroup, group_idx: int, session_id: str) -> Send:
    return Send(
        "slide_writer",
        {
            "chunk_ids":          _group_chunk_ids(group),
            "slide_blueprints":   _blueprints_as_dicts(group),
            "group_idx":          group_idx,
            "session_id":         session_id,
            "rewrite_instructions": "",
        },
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PlanExecutorAgent:
    """Stateless dispatcher — all state lives in ResearchState."""

    def __init__(self) -> None:
        from src.logging.logger import AgentLogger
        self._logger = AgentLogger()

    def _set_session_id(self, state: dict) -> None:
        from src.llm.llm import current_session_id
        sid = state.get("session_id") if isinstance(state, dict) else None
        if sid:
            current_session_id.set(sid)

    def run(self, state: ResearchState) -> Command:
        self._set_session_id(state)

        session_id        = state.get("session_id", "")
        presentation_plan: PresentationPlan | None = state.get("presentation_plan")
        slides_written:    list[dict]               = state.get("slides_written", [])

        if presentation_plan is None:
            raise ValueError("[PlanExecutor] No presentation_plan in state.")

        groups = presentation_plan.slide_groups

        # ------------------------------------------------------------------
        # First call: initial dispatch of all groups
        # ------------------------------------------------------------------
        if not slides_written:
            self._logger.log(
                f"[PlanExecutor] Initial dispatch — {len(groups)} group(s)"
            )
            sends: list[Send] = []
            for idx, group in enumerate(groups):
                chunk_ids = _group_chunk_ids(group)
                if not chunk_ids:
                    self._logger.log(
                        f"[PlanExecutor] Warning: group {idx} has 0 chunk_ids "
                        f"(blueprints: {[bp.working_title for bp in group.slide_blueprints]})",
                        level="warning",
                    )
                sends.append(_build_send(group, idx, session_id))

            return Command(goto=sends)

        # ------------------------------------------------------------------
        # Subsequent calls: verify counts, retry failures
        # ------------------------------------------------------------------

        # Build a map: group_idx → list of counts seen so far (one per attempt)
        counts_by_group: dict[int, list[int]] = {}
        for entry in slides_written:
            gidx  = entry["group_idx"]
            count = entry["count"]
            counts_by_group.setdefault(gidx, []).append(count)

        # A group is "done" if any attempt produced > 0 slides
        failed_groups: list[int] = []
        for idx in range(len(groups)):
            attempts = counts_by_group.get(idx, [])
            succeeded = any(c > 0 for c in attempts)
            if not succeeded:
                failed_groups.append(idx)

        if not failed_groups:
            total_slides = sum(
                max(counts_by_group.get(idx, [0])) for idx in range(len(groups))
            )
            msg = f"[PlanExecutor] All {len(groups)} group(s) completed. Total slides written: {total_slides}"
            self._logger.log(msg)
            return Command(update={"messages": [msg]}, goto=END)

        # Retry failed groups that haven't exhausted their retry budget
        retries: list[Send]   = []
        exhausted: list[int]  = []

        for idx in failed_groups:
            attempts_so_far = len(counts_by_group.get(idx, []))
            if attempts_so_far <= MAX_RETRIES_PER_GROUP:
                self._logger.log(
                    f"[PlanExecutor] Retrying group {idx} "
                    f"(attempt {attempts_so_far + 1}/{MAX_RETRIES_PER_GROUP + 1})",
                    level="warning",
                )
                retries.append(_build_send(groups[idx], idx, session_id))
            else:
                exhausted.append(idx)

        for idx in exhausted:
            slide_titles = [bp.working_title for bp in groups[idx].slide_blueprints]
            err_msg = (
                f"[PlanExecutor] Group {idx} exhausted {MAX_RETRIES_PER_GROUP + 1} "
                f"attempts with 0 slides. Slides skipped: {slide_titles}"
            )
            self._logger.log(err_msg, level="error")

        if retries:
            return Command(
                update={"messages": [f"[PlanExecutor] Retrying {len(retries)} failed group(s)"]},
                goto=retries,
            )

        # All failed groups exhausted — proceed to END with partial deck
        msg = (
            f"[PlanExecutor] Done with partial deck. "
            f"{len(exhausted)} group(s) permanently failed: {exhausted}"
        )
        self._logger.log(msg, level="warning")
        return Command(update={"messages": [msg]}, goto=END)


def plan_executor_node(state: ResearchState) -> Command:
    return PlanExecutorAgent().run(state)
