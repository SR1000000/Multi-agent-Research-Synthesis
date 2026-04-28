"""
SupervisorAgent — orchestrates the critic-and-rewrite quality gate for slide decks.

The supervisor sits between critic cycles and rewrite cycles in the LangGraph pipeline.
After critics report their findings it decides whether to accept the deck, dispatch targeted
rewrites, or trigger a full replan.  Decisions are forced to be conservative: critical issues
always override an LLM "accept".  Hitting the critic/rewrite cycle cap while replan budget
remains triggers a full replan; once ``plan_number`` exceeds ``MAX_REPLANS`` the run must
terminate: it exports unless aggregated critic severity counts still show critical issues.
"""
from __future__ import annotations

import json
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel

from src.memory.research.database import ResearchDatabase
from src.memory.research.replan_backup import backup_replan_debug_snapshot
from src.agents.base import BaseLLMAgent
from src.agents.prompts.critic_prompts import format_rewrite_instruction
from src.agents.prompts.supervisor_prompts import SUPERVISOR_ROLE, build_supervisor_user_prompt
from src.state import (
    MAX_CYCLES,
    MAX_REPLANS,
    PresentationPlan,
    ResearchState,
    ReviewAssignment,
    make_initial_review_state,
)


# ---------------------------------------------------------------------------
# Local type stub (formerly in state.py)
# ---------------------------------------------------------------------------

class SupervisorOutput(BaseModel):
    """Structured decision returned by the supervisor LLM call.

    The LLM proposes a decision; the agent then acts on it, subject to budget limits.
    """

    decision: Literal["accept", "revise", "replan", "critic_cycle"]  # "critic_cycle" is never output from llm call, it is only used for "decision" routing that skips the llm call
    reasoning: str
    feedback:  str = ""


def _build_grounding_assignments(*, plan: PresentationPlan, cycle_number: int) -> list[ReviewAssignment]:
    """Build one grounding critic ReviewAssignment per slide group in the presentation plan.

    Critics are scoped to groups so each assignment shares the same source chunks and
    narrative context, keeping review prompts focused and findings comparable across cycles.
    """
    assignments: list[ReviewAssignment] = []
    for idx, group in enumerate(plan.slide_groups):
        target_slide_numbers = [bp.slide_number for bp in group.slide_blueprints]
        assignments.append(
            {
                "assignment_id": f"critic-c{cycle_number}-g{idx}",
                "cycle_number": cycle_number,
                "check_type": "grounding_consistency",
                "scope_type": "group",
                "scope_id": str(idx),
                "group_idx": idx,
                "chunk_ids": list(
                    dict.fromkeys(cid for bp in group.slide_blueprints for cid in bp.source_chunk_ids)
                ),
                "slide_blueprints": [bp.model_dump() for bp in group.slide_blueprints],
                "target_slide_numbers": target_slide_numbers,
                "rewrite_instructions": "",
            }
        )
    return assignments


def _build_narrative_assignment(*, plan: PresentationPlan, cycle_number: int) -> ReviewAssignment:
    """Build the deck-scoped narrative critic assignment (one per critic cycle, ``group_idx=-1``)."""
    all_blueprints = [bp.model_dump() for g in plan.slide_groups for bp in g.slide_blueprints]
    target_slide_numbers = [bp["slide_number"] for bp in all_blueprints]
    return {
        "assignment_id": f"critic-c{cycle_number}-narrative",
        "cycle_number": cycle_number,
        "check_type": "narrative_coherence",
        "scope_type": "deck",
        "scope_id": "deck",
        "group_idx": -1,
        "chunk_ids": [],
        "slide_blueprints": all_blueprints,
        "target_slide_numbers": target_slide_numbers,
        "rewrite_instructions": "",
    }


def _build_critic_assignments(*, plan: PresentationPlan, cycle_number: int) -> list[ReviewAssignment]:
    """All critic work for one cycle: per-group grounding plus one deck-level narrative pass."""
    return _build_grounding_assignments(plan=plan, cycle_number=cycle_number) + [
        _build_narrative_assignment(plan=plan, cycle_number=cycle_number)
    ]


def _format_dispatch_targets(assignments: list[ReviewAssignment]) -> str:
    """One-line summary of critic/rewrite fan-out for terminal logs."""
    if not assignments:
        return "(none)"
    parts: list[str] = []
    for a in assignments:
        aid = a.get("assignment_id", "?")
        nums = a.get("target_slide_numbers") or []
        if nums:
            lo, hi = min(nums), max(nums)
            span = f"slides {lo}-{hi}" if lo != hi else f"slide {lo}"
        else:
            span = "slides ?"
        parts.append(f"{aid} [{span}]")
    return "; ".join(parts)


def _clip_issue_to_group_slides(issue: dict, group_slides: set[int]) -> dict | None:
    """Shallow copy of *issue* with ``affected_slide_numbers`` limited to the overlap with *group_slides*.

    Returns None when there is no overlap or when the issue has no non-empty
    ``affected_slide_numbers`` (deck-split issues with empty lists are skipped).
    Does not mutate the original *issue* dict in *results*.
    """
    raw = issue.get("affected_slide_numbers") or []
    if not raw:
        return None
    clipped = [n for n in raw if n in group_slides]
    if not clipped:
        return None
    out = dict(issue)
    out["affected_slide_numbers"] = clipped
    return out


def _ordered_union_slide_numbers(issues: list[dict]) -> list[int]:
    """Preserving first-seen order, union all ``affected_slide_numbers`` in *issues*."""
    out: list[int] = []
    seen: set[int] = set()
    for iss in issues:
        for n in iss.get("affected_slide_numbers", []):
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _build_group_rewrite_assignment(
    result: dict, plan: PresentationPlan, cycle_number: int
) -> ReviewAssignment:
    """One rewrite batch from a group-scoped critic result (``group_idx`` >= 0)."""
    group = plan.slide_groups[result["group_idx"]]
    chunk_ids = list(
        dict.fromkeys(
            cid for bp in group.slide_blueprints for cid in bp.source_chunk_ids
        )
    )
    issues = result.get("issues", [])
    valid_group_slides = set(result.get("target_slide_numbers", []))

    all_have_slide_numbers = bool(issues) and all(
        issue.get("affected_slide_numbers") for issue in issues
    )

    if all_have_slide_numbers:
        affected = list(
            dict.fromkeys(
                n
                for issue in issues
                for n in issue["affected_slide_numbers"]
                if n in valid_group_slides
            )
        )
        if not affected:
            affected = list(result.get("target_slide_numbers", []))
    else:
        affected = list(result.get("target_slide_numbers", []))

    return {
        "assignment_id": f"rewrite-{result['assignment_id']}",
        "cycle_number": cycle_number,
        "check_type": result["check_type"],
        "scope_type": result["scope_type"],
        "scope_id": result["scope_id"],
        "group_idx": result["group_idx"],
        "chunk_ids": chunk_ids,
        "slide_blueprints": [bp.model_dump() for bp in group.slide_blueprints],
        "target_slide_numbers": affected,
        "rewrite_instructions": result.get("rewrite_instructions", ""),
    }


def _split_deck_result_into_rewrite_assignments(
    result: dict, plan: PresentationPlan, cycle_number: int
) -> list[ReviewAssignment]:
    """Fan out a deck-scoped critic result (``group_idx == -1``) into one rewrite per affected group.

    For each group, build per-writer issues by clipping ``affected_slide_numbers`` to that
    group's slide set (a shallow copy; ``critic_results`` issue dicts are not mutated).
    Rebuilds ``rewrite_instructions`` from the clipped copies only.
    """
    out: list[ReviewAssignment] = []
    for group_idx, group in enumerate(plan.slide_groups):
        group_slides = {bp.slide_number for bp in group.slide_blueprints}
        clipped_issues: list[dict] = []
        for issue in result.get("issues", []):
            c = _clip_issue_to_group_slides(issue, group_slides)
            if c is not None:
                clipped_issues.append(c)
        if not clipped_issues:
            continue
        chunk_ids = list(
            dict.fromkeys(
                cid for bp in group.slide_blueprints for cid in bp.source_chunk_ids
            )
        )
        rewrite_lines = [
            format_rewrite_instruction(iss)
            for iss in clipped_issues
            if str(iss.get("rewrite_instruction", "")).strip()
        ]
        rewrite_instructions = "\n".join(rewrite_lines)
        target_slide_numbers = _ordered_union_slide_numbers(clipped_issues)
        out.append(
            {
                "assignment_id": f"rewrite-{result['assignment_id']}-g{group_idx}",
                "cycle_number": cycle_number,
                "check_type": result["check_type"],
                "scope_type": result["scope_type"],
                "scope_id": result["scope_id"],
                "group_idx": group_idx,
                "chunk_ids": chunk_ids,
                "slide_blueprints": [bp.model_dump() for bp in group.slide_blueprints],
                "target_slide_numbers": target_slide_numbers,
                "rewrite_instructions": rewrite_instructions,
            }
        )
    return out


def _build_rewrite_assignments(
    *, plan: PresentationPlan, results: list[dict], cycle_number: int
) -> list[ReviewAssignment]:
    """Build targeted rewrite ReviewAssignments from actionable critic results.

    Group-scoped (grounding) results map 1:1. Deck-scoped narrative results
    (``group_idx < 0``) are split per slide group with clipped slide numbers.
    """
    assignments: list[ReviewAssignment] = []
    for result in results:
        gidx = int(result.get("group_idx", 0))
        if gidx < 0:
            assignments.extend(
                _split_deck_result_into_rewrite_assignments(result, plan, cycle_number)
            )
        else:
            assignments.append(
                _build_group_rewrite_assignment(result, plan, cycle_number)
            )
    return assignments


class SupervisorAgent(BaseLLMAgent):
    """Stateful control-plane agent that evaluates critic results and routes the pipeline.

    On each invocation it inspects the current review state and critic findings to decide
    between three outcomes:
      - accept:  deck meets the quality bar → route to END.
      - revise:  targeted issues found → dispatch rewrite assignments then re-run critics.
      - replan:  fundamental structural problems → reset review state and call the planner.

    The decision to accept, revise, or replan is entirely determined by the LLM,
    though budget safeguards ensure that cycle limits are respected.
    """

    def __init__(self) -> None:
        """Initialise with the supervisor system prompt."""
        super().__init__("supervisor", system_prompt=SUPERVISOR_ROLE)

    def _replan(
        self,
        state: ResearchState,
        review: dict,
        cycle_number: int,
        result: SupervisorOutput | None = None,
        severity_counts: dict[str, int] | None = None,
    ) -> Command:
        """Full replan: optional debug DB backup, then reset review and bump ``plan_number``."""
        plan = state.get("presentation_plan")
        plan_json: str | None
        if plan is not None:
            plan_json = plan.model_dump_json()
        else:
            plan_json = None

        graph_metadata = {
            "review_phase": review.get("phase"),
            "cycle_number": cycle_number,
            "pending_critic": review.get("pending_critic_assignments", []),
            "pending_rewrite": review.get("pending_rewrite_assignments", []),
            "last_critic_dispatch_id": review.get("last_critic_dispatch_id"),
            "last_rewrite_ids": review.get("last_rewrite_assignment_ids", []),
            "review_summaries": state.get("review_summaries", []),
        }
        if result is not None:
            graph_metadata["supervisor_model_reasoning"] = result.reasoning
            graph_metadata["supervisor_model_feedback"] = result.feedback
        if severity_counts is not None:
            graph_metadata["severity_counts"] = severity_counts

        with ResearchDatabase() as research_db:
            backup_replan_debug_snapshot(
                research_db,
                plan_number=int(state.get("plan_number", 1)),
                session_id=state["session_id"],
                graph_metadata=graph_metadata,
                presentation_plan_json=plan_json,
            )
            research_db.save_review_event(
                session_id=state["session_id"],
                cycle_number=cycle_number,
                plan_number=int(state.get("plan_number", 1)),
                scope_type="deck",
                scope_id="deck",
                # Supervisor-level routing decision event (not a critic check).
                check_type="supervisor",
                decision="replan",
            )

        old_counter = int(review.get("dispatch_counter", 0))
        new_review = make_initial_review_state(max_cycles=review.get("max_cycles", MAX_CYCLES))
        new_review["dispatch_counter"] = old_counter

        return Command(
            update={
                "plan_number": int(state.get("plan_number", 1)) + 1,
                "presentation_plan": None,
                "review": new_review,
                "messages": ["[supervisor] replan: returning to planner"],
            },
            goto="planner",
        )

    def _accept_and_end(
        self,
        state: ResearchState,
        review: dict,
        plan_number: int,
        cycle_number: int,
        severity_counts: dict[str, int],
        rewrites_required: dict[str, bool],
        budget_exhausted: bool = False,
    ) -> Command:
        """Mark review complete, persist accept, emit summary, and route to END."""
        review.update({"final_decision": "accept", "export_ready": True, "phase": "complete"})
        
        critical_n = severity_counts.get("critical", 0)
        major_n = severity_counts.get("major", 0)
        
        if budget_exhausted:
            if critical_n > 0 or major_n > 0:
                self._logger.log(f"WARNING: Replan budget exhausted. Exporting deck with {critical_n} critical and {major_n} major issues remaining.")
            else:
                self._logger.log(f"Replan budget exhausted. Exporting deck (only minor issues remaining).")

        summary = {
            "plan_number": plan_number,
            "cycle_number": cycle_number,
            "issue_counts": severity_counts,
            "decision": "accept",
            "routing": "END",
            "rewrites_required_by_assignment": rewrites_required,
        }
        if budget_exhausted:
            summary["replan_budget_exhausted"] = True
            
        with ResearchDatabase() as research_db:
            research_db.save_review_event(
                session_id=state["session_id"],
                cycle_number=cycle_number,
                plan_number=plan_number,
                scope_type="deck",
                scope_id="deck",
                # Supervisor-level routing decision event (not a critic check).
                check_type="supervisor",
                decision="accept",
            )
        return Command(
            update={
                "review": review,
                "review_summaries": [summary],
                "messages": [json.dumps({"supervisor_cycle_summary": summary}, sort_keys=True)],
            },
            goto=END,
        )

    def run(self, state: ResearchState) -> Command:
        """Evaluate the current critic cycle and return the next routing Command."""
        self._set_session_id(state)
        self._set_plan_number(state)

        review = dict(state.get("review") or {})
        plan = state.get("presentation_plan")
        if plan is None:
            raise ValueError("[Supervisor] No presentation_plan in state.")

        plan_number = int(state.get("plan_number", 1))
        cycle_number = review.get("cycle_number", 0)
        phase = review.get("phase", "awaiting_supervisor")
        max_cycles = review.get("max_cycles", MAX_CYCLES)
        at_cycle_cap = cycle_number >= max_cycles
        at_cap_forced = (
            bool(state.get("force_replan_at_max_cycles"))
            and at_cycle_cap
            and plan_number <= MAX_REPLANS
        )
        last_did = review.get("last_critic_dispatch_id")
        critic_results = [
            r
            for r in state.get("critic_results", [])
            if last_did and r.get("dispatch_id") == last_did
        ]

        # Aggregate issue counts by severity across all critic results for a single cycle.
        severity_counts = {"critical": 0, "major": 0, "minor": 0}
        for result in critic_results:
            for issue in result.get("issues", []):
                severity = issue.get("severity")
                if severity in severity_counts:
                    severity_counts[severity] += 1
                    
        rewrites_required = {
            r["assignment_id"]: bool(r.get("actionable")) for r in critic_results
        }
        history = []
        with ResearchDatabase() as research_db:
            history = research_db.list_review_events(
                state["session_id"], plan_number=plan_number
            )
        recurring: dict[str, int] = {}
        for event in history:
            fingerprint = event.get("fingerprint")
            if fingerprint:
                recurring[fingerprint] = recurring.get(fingerprint, 0) + 1

        summaries = "\n".join(
            f"- {result['assignment_id']}: actionable={result['actionable']} summary={result['summary']}"
            for result in critic_results
        ) or "(no critic results yet)"
        recurring_lines = "\n".join(
            f"- {fp}: {count}x"
            for fp, count in sorted(recurring.items())
            if count >= 2
        ) or "(none)"

        
        # Begin Decision Making
        
        decision: str | None = None
        budget_exhausted: bool = False
        llm_result: SupervisorOutput | None = None

        # Ensure a critic pass happens if results are missing or we just finished rewrites.
        if not critic_results or review.get("last_rewrite_assignment_ids"):
            decision = "critic_cycle"

        elif at_cap_forced:
            decision = "replan"

        else:
            # LLM Call for decision

            actionable_results = [r for r in critic_results if r.get("actionable")]

            user = build_supervisor_user_prompt(
                query=state["query"],
                cycle_number=cycle_number,
                max_cycles=max_cycles,
                severity_counts=severity_counts,
                summaries=summaries,
                recurring_lines=recurring_lines,
            )

            llm_result = self._call(
                [{"role": "user", "content": user}],
                schema=SupervisorOutput,
                model="supervisor",
            )
            decision = llm_result.decision

            if decision == "revise" and not actionable_results:
                decision = "accept"
            if at_cycle_cap and decision == "revise":
                if plan_number <= MAX_REPLANS:
                    decision = "replan"
                else:
                    decision = "accept"
                    budget_exhausted = True
            if decision == "replan" and plan_number > MAX_REPLANS:
                decision = "accept"
                budget_exhausted = True

        # Force accept if toggled
        if decision == "replan":
            should_force_accept = (
                bool(state.get("force_accept_first_plan_at_cap"))
                and at_cycle_cap
                and plan_number == 1
            )
            if should_force_accept:
                decision = "accept"

        # Log the decision
        if llm_result:
            _reasoning_cleaned = " ".join((llm_result.reasoning or "").split())
            _short_reasoning = (
                _reasoning_cleaned
                if len(_reasoning_cleaned) <= 240
                else _reasoning_cleaned[:239] + "…"
            )
            self._logger.log(
                f"[supervisor] cycle {cycle_number}: model={llm_result.decision} -> effective={decision} "
                f"| counts={severity_counts} | {_short_reasoning}"
            )

        # Time to Act on the decision made

        if decision == "replan":
            return self._replan(
                state,
                review,
                cycle_number,
                result=llm_result,
                severity_counts=severity_counts,
            )

        if decision == "accept":
            return self._accept_and_end(
                state, review, plan_number, cycle_number, severity_counts, rewrites_required,
                budget_exhausted=budget_exhausted,
            )

        # Execute the next critic cycle (to Plan Executor)
        if decision == "critic_cycle":
            next_cycle = max(1, cycle_number + 1) if not critic_results else cycle_number + 1
            assignments = _build_critic_assignments(plan=plan, cycle_number=next_cycle)
            review.update(
                {
                    "cycle_number": next_cycle,
                    "phase": "critic_dispatch",
                    "pending_critic_assignments": assignments,
                    "pending_rewrite_assignments": [],
                    "last_rewrite_assignment_ids": [],
                    "last_issue_counts": severity_counts if critic_results else {"critical": 0, "major": 0, "minor": 0},
                    "last_rewrites_required_by_assignment": rewrites_required if critic_results else {},
                }
            )
            updates: dict = {"review": review}
            if not critic_results:
                msg = (
                    f"[supervisor] revise: begin critic cycle {next_cycle}"
                    if phase != "complete"
                    else f"[supervisor] revise: restart critic cycle {next_cycle}"
                )
                self._logger.log(f"{msg} | dispatch {len(assignments)} critic(s): {_format_dispatch_targets(assignments)}")
                updates["messages"] = [msg]
            else:
                summary = {
                    "plan_number": plan_number,
                    "cycle_number": cycle_number,
                    "issue_counts": severity_counts,
                    "decision": "revise",
                    "routing": "critic_cycle",
                    "rewrites_required_by_assignment": rewrites_required,
                }
                updates["review_summaries"] = [summary]
                updates["messages"] = [json.dumps({"supervisor_cycle_summary": summary}, sort_keys=True)]
                self._logger.log(
                    f"[supervisor] cycle {cycle_number}: post-rewrite critic pass -> cycle {next_cycle} "
                    f"| dispatch {len(assignments)} critic(s): {_format_dispatch_targets(assignments)} "
                    f"| prior counts={severity_counts}"
                )
            return Command(update=updates, goto="plan_executor")

        if decision == "revise":
            # This case corresponds to a fresh LLM 'revise' decision with actionable results
            actionable_results = [r for r in critic_results if r.get("actionable")]
            rewrite_assignments = _build_rewrite_assignments(
                plan=plan,
                results=actionable_results,
                cycle_number=cycle_number,
            )
            self._logger.log(
                f"[supervisor] dispatch {len(rewrite_assignments)} slide rewriter(s): "
                f"{_format_dispatch_targets(rewrite_assignments)}"
            )
            review.update(
                {
                    "phase": "rewrite_dispatch",
                    "pending_rewrite_assignments": rewrite_assignments,
                    "last_issue_counts": severity_counts,
                    "last_rewrites_required_by_assignment": rewrites_required,
                }
            )
            summary = {
                "plan_number": plan_number,
                "cycle_number": cycle_number,
                "issue_counts": severity_counts,
                "decision": decision,
                "routing": "rewrite_cycle",
                "rewrites_required_by_assignment": rewrites_required,
            }
            updates = {
                "review": review,
                "review_summaries": [summary],
                "messages": [json.dumps({"supervisor_cycle_summary": summary}, sort_keys=True)],
            }
            return Command(update=updates, goto="plan_executor")


def supervisor_node(state: ResearchState) -> Command:
    """LangGraph node entry point that constructs a SupervisorAgent and delegates to its run() method."""
    return SupervisorAgent().run(state)
