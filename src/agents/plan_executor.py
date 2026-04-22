"""
PlanExecutorAgent
=================
Deterministic fanout/fan-in coordinator for initial slide writing, critics,
and rewrites.
"""
from __future__ import annotations

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


def _build_slide_writer_send(
    *,
    dispatch_id: str,
    assignment_id: str,
    group: SlideGroup,
    group_idx: int,
    session_id: str,
    rewrite_instructions: str = "",
    target_slide_numbers: list[int] | None = None,
) -> Send:
    return Send(
        "slide_writer",
        {
            "dispatch_id":        dispatch_id,
            "assignment_id":      assignment_id,
            "chunk_ids":          _group_chunk_ids(group),
            "slide_blueprints":   _blueprints_as_dicts(group),
            "group_idx":          group_idx,
            "session_id":         session_id,
            "rewrite_instructions": rewrite_instructions,
            "target_slide_numbers": target_slide_numbers or [],
        },
    )


def _build_critic_send(*, dispatch_id: str, session_id: str, assignment: dict) -> Send:
    return Send(
        "critic",
        {
            "dispatch_id": dispatch_id,
            "assignment_id": assignment["assignment_id"],
            "cycle_number": assignment["cycle_number"],
            "session_id": session_id,
            "check_type": assignment["check_type"],
            "scope_type": assignment["scope_type"],
            "scope_id": assignment["scope_id"],
            "group_idx": assignment["group_idx"],
            "chunk_ids": assignment["chunk_ids"],
            "slide_blueprints": assignment["slide_blueprints"],
            "target_slide_numbers": assignment["target_slide_numbers"],
            "rewrite_instructions": assignment.get("rewrite_instructions", ""),
        },
    )


def _exhausted_group_message(group: SlideGroup, group_idx: int) -> str:
    """Return a user-visible warning when a group's retries are exhausted."""
    slide_titles = [bp.working_title for bp in group.slide_blueprints]
    return (
        f"[PlanExecutor] RETRIES EXHAUSTED — group {group_idx} failed after "
        f"{MAX_RETRIES_PER_GROUP + 1} attempts with 0 slides. "
        f"Slides skipped: {slide_titles}"
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
        critic_results:    list[dict]               = state.get("critic_results", [])
        review = dict(state.get("review") or {})

        if presentation_plan is None:
            raise ValueError("[PlanExecutor] No presentation_plan in state.")

        groups = presentation_plan.slide_groups
        phase = review.get("phase", "initial_write")
        dispatch_counter = int(review.get("dispatch_counter", 0))
        active_dispatch = review.get("active_dispatch")

        if phase == "initial_write" and active_dispatch is None:
            self._logger.log(
                f"[PlanExecutor] Initial dispatch — {len(groups)} group(s)"
            )
            sends: list[Send] = []
            dispatch_id = f"initial-{dispatch_counter + 1}"
            for idx, group in enumerate(groups):
                chunk_ids = _group_chunk_ids(group)
                if not chunk_ids:
                    self._logger.log(
                        f"[PlanExecutor] Warning: group {idx} has 0 chunk_ids "
                        f"(blueprints: {[bp.working_title for bp in group.slide_blueprints]})",
                        level="warning",
                    )
                sends.append(
                    _build_slide_writer_send(
                        dispatch_id=dispatch_id,
                        assignment_id=f"initial-g{idx}",
                        group=group,
                        group_idx=idx,
                        session_id=session_id,
                    )
                )

            review.update(
                {
                    "phase": "initial_write",
                    "dispatch_counter": dispatch_counter + 1,
                    "active_dispatch": {
                        "dispatch_id": dispatch_id,
                        "kind": "initial_write",
                        "cycle_number": 0,
                        "expected_assignment_ids": [f"initial-g{idx}" for idx in range(len(groups))],
                    },
                }
            )
            return Command(update={"review": review}, goto=sends)

        if phase == "initial_write" and active_dispatch is not None:
            relevant = [entry for entry in slides_written if entry.get("dispatch_id") == active_dispatch["dispatch_id"]]
            if len(relevant) < len(active_dispatch["expected_assignment_ids"]):
                return Command(update={})
            counts_by_group: dict[int, list[int]] = {}
            for entry in relevant:
                gidx = entry["group_idx"]
                count = entry["count"]
                counts_by_group.setdefault(gidx, []).append(count)
            failed_groups: list[int] = []
            for idx in range(len(groups)):
                attempts = counts_by_group.get(idx, [])
                if not any(c > 0 for c in attempts):
                    failed_groups.append(idx)
            if failed_groups:
                retries: list[Send] = []
                exhausted_messages: list[str] = []
                for idx in failed_groups:
                    attempts_so_far = len(counts_by_group.get(idx, []))
                    if attempts_so_far <= MAX_RETRIES_PER_GROUP:
                        retries.append(
                            _build_slide_writer_send(
                                dispatch_id=active_dispatch["dispatch_id"],
                                assignment_id=f"initial-g{idx}",
                                group=groups[idx],
                                group_idx=idx,
                                session_id=session_id,
                            )
                        )
                    else:
                        exhausted_messages.append(_exhausted_group_message(groups[idx], idx))
                if retries:
                    return Command(update={"messages": exhausted_messages}, goto=retries)
            total_slides = sum(max(counts_by_group.get(idx, [0])) for idx in range(len(groups)))
            if state.get("skip_supervisor"):
                review.update(
                    {
                        "phase": "complete",
                        "active_dispatch": None,
                        "export_ready": True,
                        "final_decision": "skipped",
                    }
                )
                msg = (
                    f"[PlanExecutor] Initial write complete (supervisor skipped). "
                    f"Total slides written: {total_slides}"
                )
                return Command(update={"review": review, "messages": [msg]}, goto=END)
            review.update({"phase": "awaiting_supervisor", "active_dispatch": None})
            msg = f"[PlanExecutor] Initial write complete. Total slides written: {total_slides}"
            return Command(update={"review": review, "messages": [msg]}, goto="supervisor")

        if phase == "critic_dispatch":
            assignments = review.get("pending_critic_assignments", [])
            if assignments and active_dispatch is None:
                dispatch_id = f"critic-{dispatch_counter + 1}"
                sends = [
                    _build_critic_send(dispatch_id=dispatch_id, session_id=session_id, assignment=assignment)
                    for assignment in assignments
                ]
                review.update(
                    {
                        "dispatch_counter": dispatch_counter + 1,
                        "active_dispatch": {
                            "dispatch_id": dispatch_id,
                            "kind": "critic",
                            "cycle_number": review.get("cycle_number", 0),
                            "expected_assignment_ids": [assignment["assignment_id"] for assignment in assignments],
                        },
                    }
                )
                return Command(update={"review": review}, goto=sends)

            if active_dispatch:
                relevant_results = [
                    result for result in critic_results
                    if result.get("dispatch_id") == active_dispatch["dispatch_id"]
                ]
                if len(relevant_results) < len(active_dispatch["expected_assignment_ids"]):
                    return Command(update={})
                review.update(
                    {
                        "active_dispatch": None,
                        "pending_critic_assignments": [],
                        "last_critic_assignment_ids": [result["assignment_id"] for result in relevant_results],
                        "last_rewrites_required_by_assignment": {
                            result["assignment_id"]: bool(result.get("actionable"))
                            for result in relevant_results
                        },
                        "phase": "awaiting_supervisor",
                    }
                )
                return Command(update={"review": review}, goto="supervisor")

        if phase == "rewrite_dispatch":
            assignments = review.get("pending_rewrite_assignments", [])
            if assignments and active_dispatch is None:
                dispatch_id = f"rewrite-{dispatch_counter + 1}"
                sends = []
                for assignment in assignments:
                    group_idx = assignment["group_idx"]
                    sends.append(
                        _build_slide_writer_send(
                            dispatch_id=dispatch_id,
                            assignment_id=assignment["assignment_id"],
                            group=groups[group_idx],
                            group_idx=group_idx,
                            session_id=session_id,
                            rewrite_instructions=assignment.get("rewrite_instructions", ""),
                            target_slide_numbers=assignment.get("target_slide_numbers", []),
                        )
                    )
                review.update(
                    {
                        "dispatch_counter": dispatch_counter + 1,
                        "active_dispatch": {
                            "dispatch_id": dispatch_id,
                            "kind": "rewrite",
                            "cycle_number": review.get("cycle_number", 0),
                            "expected_assignment_ids": [assignment["assignment_id"] for assignment in assignments],
                        },
                    }
                )
                return Command(update={"review": review}, goto=sends)

            if active_dispatch:
                relevant_writes = [
                    entry for entry in slides_written
                    if entry.get("dispatch_id") == active_dispatch["dispatch_id"]
                ]
                if len(relevant_writes) < len(active_dispatch["expected_assignment_ids"]):
                    return Command(update={})
                review.update(
                    {
                        "active_dispatch": None,
                        "pending_rewrite_assignments": [],
                        "last_rewrite_assignment_ids": [entry["assignment_id"] for entry in relevant_writes],
                        "phase": "awaiting_supervisor",
                    }
                )
                return Command(update={"review": review}, goto="supervisor")

        if review.get("export_ready"):
            return Command(goto=END)

        return Command(update={}, goto="supervisor")


def plan_executor_node(state: ResearchState) -> Command:
    return PlanExecutorAgent().run(state)
