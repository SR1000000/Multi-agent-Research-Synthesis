"""
SupervisorAgent — orchestrates the critic-and-rewrite quality gate for slide decks.

The supervisor sits between critic cycles and rewrite cycles in the LangGraph pipeline.
After critics report their findings it decides whether to accept the deck, dispatch targeted
rewrites, or trigger a full replan.  Decisions are forced to be conservative: critical issues
always override an LLM "accept", and the cycle cap prevents infinite rewrite loops.
"""
import json

from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel

from src.memory.research.database import ResearchDatabase
from src.agents.base import BaseLLMAgent
from src.agents.prompts.supervisor_prompts import SUPERVISOR_ROLE, build_supervisor_user_prompt
from src.state import MAX_CYCLES, ResearchState, ReviewAssignment, make_initial_review_state


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


def _build_group_assignments(
    *, plan, cycle_number: int, plan_generation: int
) -> list[ReviewAssignment]:
    """Build one critic ReviewAssignment per slide group in the presentation plan.

    Critics are scoped to groups so each assignment shares the same source chunks and
    narrative context, keeping review prompts focused and findings comparable across cycles.
    """
    assignments: list[ReviewAssignment] = []
    for idx, group in enumerate(plan.slide_groups):
        target_slide_numbers = [bp.slide_number for bp in group.slide_blueprints]
        assignments.append(
            {
                "plan_generation": plan_generation,
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
    *, plan, results: list[dict], cycle_number: int, plan_generation: int
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
            # Fallback if the critic only hallucinated out-of-bounds slide numbers
            if not affected:
                affected = result.get("target_slide_numbers", [])
        else:
            affected = result.get("target_slide_numbers", [])

        assignments.append(
            {
                "plan_generation": plan_generation,
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
      - accept:  deck meets the quality bar → route to END (or force-export on cycle cap).
      - revise:  targeted issues found → dispatch rewrite assignments then re-run critics.
      - replan:  fundamental structural problems → reset review state and call the planner.

    Override rules ensure critical issues are never silently accepted and that cycle limits
    are respected regardless of what the LLM proposes.
    """

    def __init__(self) -> None:
        """Initialise with the supervisor system prompt."""
        super().__init__("supervisor", system_prompt=SUPERVISOR_ROLE)

    def run(self, state: ResearchState) -> Command:
        """Evaluate the current critic cycle and return the next routing Command.

        Reads ``critic_results`` filtered to the current cycle and plan generation, queries
        the database for recurring issue fingerprints, calls the LLM for a proposed decision,
        then applies conservative override rules before returning a Command routed to the next
        node.  Review state is mutated in-place and written back to the graph state on every path.
        """
        self._set_session_id(state)
        review = dict(state.get("review") or {})
        plan = state.get("presentation_plan")
        if plan is None:
            raise ValueError("[Supervisor] No presentation_plan in state.")

        cycle_number = review.get("cycle_number", 0)
        plan_generation = review.get("plan_generation", 0)
        phase = review.get("phase", "awaiting_supervisor")
        critic_results = [
            result
            for result in state.get("critic_results", [])
            if result.get("cycle_number") == cycle_number
            and result.get("plan_generation", 0) == plan_generation
        ]
        severity_counts = _severity_counts(critic_results)
        # Map each assignment ID to whether its critic result was actionable.
        rewrites_required = {
            r["assignment_id"]: bool(r.get("actionable")) for r in critic_results
        }
        history = []
        with ResearchDatabase() as research_db:
            history = research_db.list_review_events(
                state["session_id"], plan_generation=plan_generation
            )
        recurring = {}
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

        # No critic results yet for the current checkpoint means the supervisor should
        # launch or relaunch a critic cycle, not attempt acceptance from stale state.
        if not critic_results:
            next_cycle = max(1, cycle_number + 1)
            if cycle_number >= review.get("max_cycles", MAX_CYCLES) and review.get("last_rewrite_assignment_ids"):
                review.update({"final_decision": "accept", "export_ready": True, "phase": "complete"})
                summary = {
                    "plan_generation": plan_generation,
                    "cycle_number": cycle_number,
                    "issue_counts": review.get("last_issue_counts", {"critical": 0, "major": 0, "minor": 0}),
                    "decision": "accept",
                    "routing": "END",
                    "rewrites_required_by_assignment": review.get("last_rewrites_required_by_assignment", {}),
                }
                self._logger.log(
                    "[supervisor] accept: max critic cycles reached while rewrites were still pending; exporting partial deck "
                    f"(cycle {cycle_number})"
                )
                return Command(
                    update={
                        "review": review,
                        "review_summaries": [summary],
                        "messages": [json.dumps({"supervisor_cycle_summary": summary}, sort_keys=True)],
                    },
                    goto=END,
                )

            assignments = _build_group_assignments(
                plan=plan, cycle_number=next_cycle, plan_generation=plan_generation
            )
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

        # If rewrites already ran for this cycle, the next step is another critic pass
        # over the updated slides rather than acceptance based on stale critic findings.
        if review.get("last_rewrite_assignment_ids"):
            if cycle_number >= review.get("max_cycles", MAX_CYCLES):
                review.update({"final_decision": "accept", "export_ready": True, "phase": "complete"})
                summary = {
                    "plan_generation": plan_generation,
                    "cycle_number": cycle_number,
                    "issue_counts": severity_counts,
                    "decision": "accept",
                    "routing": "END",
                    "rewrites_required_by_assignment": rewrites_required,
                }
                self._logger.log(
                    f"[supervisor] accept: max cycles reached after rewrite pass; exporting partial deck (cycle {cycle_number}, "
                    f"counts={severity_counts})"
                )
                return Command(
                    update={
                        "review": review,
                        "review_summaries": [summary],
                        "messages": [json.dumps({"supervisor_cycle_summary": summary}, sort_keys=True)],
                    },
                    goto=END,
                )

            next_cycle = cycle_number + 1
            assignments = _build_group_assignments(
                plan=plan, cycle_number=next_cycle, plan_generation=plan_generation
            )
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
                "plan_generation": plan_generation,
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

        actionable_results = [r for r in critic_results if r.get("actionable")]
        max_cycles = review.get("max_cycles", MAX_CYCLES)

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
        at_cycle_cap = cycle_number >= max_cycles
        # Return True if any actionable critic result contains at least one
        # critical- or major-severity issue. (actionable_results is pre-filtered
        # to actionable=True, so the old per-result actionable guard is redundant here.)
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


        summary = {
            "plan_generation": plan_generation,
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

        # Collapse multi-line LLM reasoning to a single truncated line suitable for terminal logs.
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
            old_plan_generation = review.get("plan_generation", 0)
            with ResearchDatabase() as research_db:
                research_db.save_review_event(
                    session_id=state["session_id"],
                    cycle_number=cycle_number,
                    plan_generation=old_plan_generation,
                    scope_type="deck",
                    scope_id="deck",
                    check_type="grounding_consistency",
                    decision="replan",
                )
            new_review = make_initial_review_state(max_cycles=review.get("max_cycles", MAX_CYCLES))
            new_review["dispatch_counter"] = review.get("dispatch_counter", 0)
            new_review["plan_generation"] = old_plan_generation + 1
            updates["review"] = new_review
            return Command(update=updates, goto="planner")

        if decision == "accept":
            review.update({"final_decision": "accept", "export_ready": True, "phase": "complete"})
            updates["review"] = review
            with ResearchDatabase() as research_db:
                research_db.save_review_event(
                    session_id=state["session_id"],
                    cycle_number=cycle_number,
                    plan_generation=plan_generation,
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
            plan_generation=plan_generation,
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
