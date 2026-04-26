"""
SupervisorAgent — orchestrates the critic-and-rewrite quality gate for slide decks.

The supervisor sits between critic cycles and rewrite cycles in the LangGraph pipeline.
After critics report their findings it decides whether to accept the deck, dispatch targeted
rewrites, or trigger a full replan.  Decisions are forced to be conservative: critical issues
always override an LLM "accept".  Hitting the critic/rewrite cycle cap while replan budget
remains triggers a full replan; once ``plan_number`` exceeds ``MAX_REPLANS`` the run must
terminate: it exports unless aggregated critic severity counts still show critical issues.
"""
import json

from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel

from src.memory.research.database import ResearchDatabase
from src.memory.research.replan_backup import backup_replan_debug_snapshot
from src.agents.base import BaseLLMAgent
from src.agents.prompts.supervisor_prompts import SUPERVISOR_ROLE, build_supervisor_user_prompt
from src.state import MAX_CYCLES, MAX_REPLANS, ResearchState, ReviewAssignment, make_initial_review_state


# ---------------------------------------------------------------------------
# Local type stub (formerly in state.py)
# ---------------------------------------------------------------------------

class SupervisorOutput(BaseModel):
    """Structured decision returned by the supervisor LLM call.

    The LLM proposes a decision; the agent then applies override rules (e.g. forcing
    "revise" when critical issues are present) before acting on the final decision.
    """

    decision:  str   # "accept" | "revise" | "replan"
    reasoning: str
    feedback:  str = ""

def _severity_counts(results: list[dict]) -> dict[str, int]:
    """Aggregate issue counts by severity across all critic results for a single cycle."""
    counts = {"critical": 0, "major": 0, "minor": 0}
    for result in results:
        for issue in result.get("issues", []):
            severity = issue.get("severity")
            if severity in counts:
                counts[severity] += 1
    return counts


def _all_actionable_issues_are_persistent_minor(
    results: list[dict],
    recurring_counts: dict[str, int],
) -> bool:
    """Return True when every outstanding issue is minor and has recurred at least twice.

    Persistent minor findings can be accepted to break endless rewrite loops where the LLM
    repeatedly flags low-risk stylistic concerns it is unable to resolve.
    Returns False when there are no actionable issues at all (caller must not accept on vacuous truth).
    """
    saw_issue = False
    for result in results:
        if not result.get("actionable"):
            continue
        for issue in result.get("issues", []):
            saw_issue = True
            if issue.get("severity") != "minor":
                return False
            fingerprint = issue.get("fingerprint")
            if not fingerprint or recurring_counts.get(fingerprint, 0) < 2:
                return False
    return saw_issue


def _build_group_assignments(*, plan, cycle_number: int) -> list[ReviewAssignment]:
    """Build one critic ReviewAssignment per slide group in the presentation plan.

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
                "chunk_ids": list(dict.fromkeys(cid for bp in group.slide_blueprints for cid in bp.source_chunk_ids)),
                "slide_blueprints": [bp.model_dump() for bp in group.slide_blueprints],
                "target_slide_numbers": target_slide_numbers,
                "rewrite_instructions": "",
            }
        )
    return assignments


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


def _build_rewrite_assignments(
    *, plan, results: list[dict], cycle_number: int
) -> list[ReviewAssignment]:
    """Build targeted rewrite ReviewAssignments from actionable critic results.

    Prefers to rewrite only the specific slides named by the critic; falls back to the
    full group when no individual slide numbers are provided or when all named slides are
    outside the valid group range (i.e. the critic hallucinated out-of-bounds numbers).
    """
    assignments: list[ReviewAssignment] = []
    for result in results:
        group = plan.slide_groups[result["group_idx"]]
        chunk_ids = list(
            dict.fromkeys(
                cid
                for bp in group.slide_blueprints
                for cid in bp.source_chunk_ids
            )
        )
        issues = result.get("issues", [])
        valid_group_slides = set(result.get("target_slide_numbers", []))

        all_have_slide_numbers = bool(issues) and all(
            issue.get("affected_slide_numbers") for issue in issues
        )

        if all_have_slide_numbers:
            affected = list(dict.fromkeys(
                n for issue in issues for n in issue["affected_slide_numbers"]
                if n in valid_group_slides
            ))
            if not affected:
                affected = result.get("target_slide_numbers", [])
        else:
            affected = result.get("target_slide_numbers", [])

        assignments.append(
            {
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
        )
    return assignments


class SupervisorAgent(BaseLLMAgent):
    """Stateful control-plane agent that evaluates critic results and routes the pipeline.

    On each invocation it inspects the current review state and critic findings to decide
    between three outcomes:
      - accept:  deck meets the quality bar → route to END.
      - revise:  targeted issues found → dispatch rewrite assignments then re-run critics.
      - replan:  fundamental structural problems → reset review state and call the planner.

    Override rules ensure critical issues are never silently accepted and that cycle limits
    are respected regardless of what the LLM proposes.
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
                check_type="grounding_consistency",
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
        severity_counts = _severity_counts(critic_results)
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

        def end_after_replan_budget_gone(
            severity_cts: dict[str, int],
            rewrites_req: dict[str, bool],
            log_context: str,
        ) -> Command:
            critical = int(severity_cts.get("critical") or 0)
            if critical > 0:
                review.update({"final_decision": None, "export_ready": False, "phase": "complete"})
                summary = {
                    "plan_number": plan_number,
                    "cycle_number": cycle_number,
                    "issue_counts": severity_cts,
                    "decision": "max_cycles_exhausted_critical",
                    "routing": "END",
                    "rewrites_required_by_assignment": rewrites_req,
                }
                self._logger.log(
                    f"{log_context} — {critical} critical issue(s); replan budget exhausted; not exporting."
                )
                return Command(
                    update={
                        "review": review,
                        "review_summaries": [summary],
                        "messages": [json.dumps({"supervisor_cycle_summary": summary}, sort_keys=True)],
                    },
                    goto=END,
                )
            review.update({"final_decision": "accept", "export_ready": True, "phase": "complete"})
            summary = {
                "plan_number": plan_number,
                "cycle_number": cycle_number,
                "issue_counts": severity_cts,
                "decision": "accept",
                "routing": "END",
                "rewrites_required_by_assignment": rewrites_req,
                "replan_budget_exhausted": True,
            }
            self._logger.log(
                f"{log_context} — replan budget exhausted; exporting (no critical issues, major/minor may remain)."
            )
            with ResearchDatabase() as research_db:
                research_db.save_review_event(
                    session_id=state["session_id"],
                    cycle_number=cycle_number,
                    plan_number=plan_number,
                    scope_type="deck",
                    scope_id="deck",
                    check_type="grounding_consistency",
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

        # No critic results yet for the current checkpoint — launch or relaunch a critic cycle.
        if not critic_results:
            next_cycle = max(1, cycle_number + 1)
            if at_cycle_cap and review.get("last_rewrite_assignment_ids"):
                if plan_number <= MAX_REPLANS:
                    return self._replan(
                        state,
                        review,
                        cycle_number,
                        severity_counts=review.get(
                            "last_issue_counts", {"critical": 0, "major": 0, "minor": 0}
                        ),
                    )
                last_counts = review.get("last_issue_counts", {"critical": 0, "major": 0, "minor": 0})
                return end_after_replan_budget_gone(
                    last_counts,
                    review.get("last_rewrites_required_by_assignment", {}),
                    (
                        f"[supervisor] critic cycle cap (cycle {cycle_number}): rewrites pending, "
                        "no new critic batch yet"
                    ),
                )

            assignments = _build_group_assignments(plan=plan, cycle_number=next_cycle)
            review.update(
                {
                    "cycle_number": next_cycle,
                    "phase": "critic_dispatch",
                    "pending_critic_assignments": assignments,
                    "pending_rewrite_assignments": [],
                    "last_rewrite_assignment_ids": [],
                    "last_issue_counts": {"critical": 0, "major": 0, "minor": 0},
                    "last_rewrites_required_by_assignment": {},
                }
            )
            msg = (
                f"[supervisor] revise: begin critic cycle {next_cycle}"
                if phase != "complete"
                else f"[supervisor] revise: restart critic cycle {next_cycle}"
            )
            self._logger.log(
                f"{msg} | dispatch {len(assignments)} critic(s): {_format_dispatch_targets(assignments)}"
            )
            return Command(update={"review": review, "messages": [msg]}, goto="plan_executor")

        # Post-rewrite: need another critic pass over updated slides.
        if review.get("last_rewrite_assignment_ids"):
            if at_cycle_cap:
                if plan_number <= MAX_REPLANS:
                    return self._replan(state, review, cycle_number, severity_counts=severity_counts)
                return end_after_replan_budget_gone(
                    severity_counts,
                    rewrites_required,
                    (
                        f"[supervisor] critic cycle cap (cycle {cycle_number}): after rewrite pass, "
                        f"counts={severity_counts}"
                    ),
                )

            next_cycle = cycle_number + 1
            assignments = _build_group_assignments(plan=plan, cycle_number=next_cycle)
            review.update(
                {
                    "cycle_number": next_cycle,
                    "phase": "critic_dispatch",
                    "pending_critic_assignments": assignments,
                    "pending_rewrite_assignments": [],
                    "last_rewrite_assignment_ids": [],
                    "last_issue_counts": severity_counts,
                    "last_rewrites_required_by_assignment": rewrites_required,
                }
            )
            summary = {
                "plan_number": plan_number,
                "cycle_number": cycle_number,
                "issue_counts": severity_counts,
                "decision": "revise",
                "routing": "critic_cycle",
                "rewrites_required_by_assignment": rewrites_required,
            }
            self._logger.log(
                f"[supervisor] cycle {cycle_number}: post-rewrite critic pass -> cycle {next_cycle} "
                f"| dispatch {len(assignments)} critic(s): {_format_dispatch_targets(assignments)} "
                f"| prior counts={severity_counts}"
            )
            return Command(
                update={
                    "review": review,
                    "review_summaries": [summary],
                    "messages": [json.dumps({"supervisor_cycle_summary": summary}, sort_keys=True)],
                },
                goto="plan_executor",
            )

        if at_cap_forced:
            return self._replan(state, review, cycle_number, severity_counts=severity_counts)

        actionable_results = [r for r in critic_results if r.get("actionable")]

        user = build_supervisor_user_prompt(
            query=state["query"],
            cycle_number=cycle_number,
            max_cycles=max_cycles,
            severity_counts=severity_counts,
            summaries=summaries,
            recurring_lines=recurring_lines,
        )

        result: SupervisorOutput = self._call(
            [{"role": "user", "content": user}],
            schema=SupervisorOutput,
        )
        model_decision = result.decision
        # Return True if any actionable critic result contains at least one
        # critical- or major-severity issue.
        has_critical_actionable = any(
            i.get("severity") == "critical"
            for r in actionable_results
            for i in r.get("issues", [])
        )
        has_major_actionable = any(
            i.get("severity") == "major"
            for r in actionable_results
            for i in r.get("issues", [])
        )
        has_only_persistent_minor_actionable = _all_actionable_issues_are_persistent_minor(
            actionable_results,
            recurring,
        )
        decision = model_decision
        if decision == "accept":
            if has_critical_actionable:
                decision = "revise"
            elif not at_cycle_cap and has_major_actionable:
                decision = "revise"
            elif not at_cycle_cap and actionable_results and not has_only_persistent_minor_actionable:
                decision = "revise"
        elif decision == "revise" and not actionable_results:
            decision = "accept"
        if at_cycle_cap and decision == "revise":
            if plan_number <= MAX_REPLANS:
                decision = "replan"
            else:
                critical_n = int(severity_counts.get("critical") or 0)
                if critical_n > 0:
                    return end_after_replan_budget_gone(
                        severity_counts,
                        rewrites_required,
                        (
                            f"[supervisor] cycle {cycle_number}: critic cycle cap with further "
                            "revisions needed but replan budget exhausted"
                        ),
                    )
                decision = "accept"

        if decision == "replan" and plan_number > MAX_REPLANS:
            critical_n = int(severity_counts.get("critical") or 0)
            if critical_n > 0:
                return end_after_replan_budget_gone(
                    severity_counts,
                    rewrites_required,
                    (
                        f"[supervisor] cycle {cycle_number}: model requested replan but replan "
                        "budget is exhausted"
                    ),
                )
            decision = "accept"

        summary = {
            "plan_number": plan_number,
            "cycle_number": cycle_number,
            "issue_counts": severity_counts,
            "decision": decision,
            "routing": (
                "planner"
                if decision == "replan"
                else "rewrite_cycle"
                if decision == "revise"
                else "accept"
            ),
            "rewrites_required_by_assignment": rewrites_required,
        }
        updates: dict = {
            "review_summaries": [summary],
            "messages": [json.dumps({"supervisor_cycle_summary": summary}, sort_keys=True)],
        }

        _reasoning_one = " ".join((result.reasoning or "").split())
        _short_reasoning = (
            _reasoning_one
            if len(_reasoning_one) <= 240
            else _reasoning_one[:239] + "…"
        )
        self._logger.log(
            f"[supervisor] cycle {cycle_number}: model={model_decision} -> effective={decision} "
            f"| counts={severity_counts} | {_short_reasoning}"
        )

        if decision == "replan":
            return self._replan(state, review, cycle_number, result=result, severity_counts=severity_counts)

        if decision == "accept":
            review.update({"final_decision": "accept", "export_ready": True, "phase": "complete"})
            updates["review"] = review
            with ResearchDatabase() as research_db:
                research_db.save_review_event(
                    session_id=state["session_id"],
                    cycle_number=cycle_number,
                    plan_number=plan_number,
                    scope_type="deck",
                    scope_id="deck",
                    check_type="grounding_consistency",
                    decision="accept",
                )
            return Command(update=updates, goto=END)

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
        updates["review"] = review
        return Command(update=updates, goto="plan_executor")


def supervisor_node(state: ResearchState) -> Command:
    """LangGraph node entry point that constructs a SupervisorAgent and delegates to its run() method."""
    return SupervisorAgent().run(state)
