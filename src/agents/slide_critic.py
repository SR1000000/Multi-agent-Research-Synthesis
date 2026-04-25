from __future__ import annotations

import hashlib
from typing import Any, Literal, TypedDict

from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.base import BaseLLMAgent
from src.agents.prompts.common import format_image_assets_block, format_slide_for_prompt, ordered_chunk_texts
from src.agents.prompts.critic_prompts import (
    CRITIC_ROLE,
    build_critic_user_prompt,
    format_retrieved_artifact_row,
    format_rewrite_instruction,
)
from src.memory.research.database import ResearchDatabase
from src.memory.research.schema import ProtoSlide
from src.state import (
    CriticResultRecord,
    ResearchState,
    ReviewAssignment,
)


class CriticDispatch(TypedDict):
    """State payload delivered to a critic node via LangGraph's Send() API.

    Carries everything the critic needs: which slides to review, the source
    chunks those slides were written from, the plan blueprint for intent context.
    """

    plan_generation: int
    dispatch_id: str
    assignment_id: str
    cycle_number: int
    session_id: str
    check_type: str
    scope_type: str
    scope_id: str
    group_idx: int
    chunk_ids: list[str]
    slide_blueprints: list[dict]
    target_slide_numbers: list[int]


class CriticIssue(BaseModel):
    """A single structured issue identified by the critic LLM for one review assignment."""

    issue_code: str = Field(description="Unique issue id like ISS_001")
    severity: Literal["critical", "major", "minor"]
    issue_type: str
    location: str
    affected_slide_numbers: list[int] = Field(default_factory=list)
    rewrite_instruction: str = Field(description="Precise instruction describing how to fix the issue.")


class CriticOutput(BaseModel):
    """Structured output returned by the critic LLM for a single review assignment."""

    summary: str = Field(
        description="A concise 1-2 sentence overview of the review. Do NOT list specific issues here; use the issues array for structured issues."
    )
    actionable: bool = Field(
        description="True if there is at least one issue that requires correcting. False otherwise."
    )
    issues: list[CriticIssue] = Field(
        default_factory=list,
        description="List of specific issues found. Leave empty if actionable is False."
    )


class SlideCriticAgent(BaseLLMAgent):
    """LLM-backed agent that reviews a batch of slides for grounding and consistency issues.

    Each instance is dispatched for a single critic assignment (one slide group per cycle).
    It loads the current slide drafts, the baseline source chunks, and the session-wide
    retrieval log, then asks the LLM to produce a structured list of issues.  Results are
    fingerprinted and persisted to the research database so the supervisor can detect
    recurring problems across cycles and avoid infinite rewrite loops on stable findings.
    """

    def __init__(self, *, log_display: str | None = None) -> None:
        """Initialise with the critic system prompt; ``log_display`` labels parallel critic logs."""
        super().__init__("critic", system_prompt=CRITIC_ROLE, log_display=log_display)

    def _load_session_retrieval_log(
        self, session_id: str, plan_generation: int | None = None
    ) -> str:
        """Load and format all normalized research artifacts retrieved during the session.

        Returns a newline-separated block of artifact rows suitable for injection into
        the review prompt, or an empty string when no session ID is provided.
        """
        if not session_id:
            return ""
        with ResearchDatabase() as research_db:
            rows = research_db.load_normalized_artifacts_for_session(
                session_id, plan_generation=plan_generation
            )
        ordered: list[str] = []
        for row in rows:
            ordered.append(format_retrieved_artifact_row(row))
        return "\n\n".join(ordered)

    def _build_user_prompt(self, state: CriticDispatch) -> str:
        """Assemble the full review packet for the LLM.

        Combines the plan intent (blueprint block), current slide drafts, baseline source
        chunks, the session-wide retrieval log, and any embedded image assets into a single
        structured prompt consumed by the critic LLM.
        """
        slide_numbers = state.get("target_slide_numbers", [])
        # Fetch the current persisted draft for each requested slide number from the research database.
        slides: list[ProtoSlide] = []
        with ResearchDatabase() as research_db:
            for slide_number in slide_numbers:
                slide = research_db.load_slide(slide_number)
                if slide is not None:
                    slides.append(slide)
        
        plan_gen = int(state.get("plan_generation", 0))
        retrieval_log = self._load_session_retrieval_log(
            state.get("session_id", ""), plan_generation=plan_gen
        )
        blueprints = state.get("slide_blueprints", [])
        blueprint_block = "\n".join(
            f"Slide {bp.get('slide_number')}: {bp.get('working_title', '')} — {bp.get('intent', '')}"
            for bp in blueprints
            if bp.get("slide_number") in set(slide_numbers)
        )
        slides_block = "\n\n".join(format_slide_for_prompt(slide) for slide in slides) or "No slides found."
        chunk_ids = state.get("chunk_ids", [])
        baseline_chunks_block = ""
        with ResearchDatabase() as research_db:
            image_metadatas = research_db.get_images_for_chunks(chunk_ids)
            if chunk_ids:
                placeholders = ",".join(["?"] * len(chunk_ids))
                rows = research_db.connection.execute(
                    f"SELECT id, text, contextualized_text FROM text_chunks "
                    f"WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
                # Return chunk text blocks in the caller-provided chunk order (see ordered_chunk_texts in common).
                ordered_texts = ordered_chunk_texts(rows, chunk_ids)
                baseline_chunks_block = "\n\n".join(ordered_texts)

        image_block = format_image_assets_block(image_metadatas)
        return build_critic_user_prompt(
            cycle_number=state["cycle_number"],
            check_type=state["check_type"],
            scope_type=state["scope_type"],
            scope_id=state["scope_id"],
            target_slide_numbers=slide_numbers,
            blueprint_block=blueprint_block,
            slides_block=slides_block,
            baseline_chunks_block=baseline_chunks_block,
            retrieval_log=retrieval_log,
            image_block=image_block,
            output_model=CriticOutput,
        )

    def run(self, state: CriticDispatch) -> Command:
        """Execute the critic review for one assignment and return a LangGraph Command.

        Calls the LLM with the assembled review prompt, converts each issue to a
        fingerprinted dict, and persists every issue (or a pass event when none are found)
        to the research database.  Returns a Command carrying a CriticResultRecord for the
        supervisor to evaluate on its next invocation.
        """
        self._set_session_id(state)
        self._set_plan_generation(state)
        plan_gen = int(state.get("plan_generation", 0))
        prompt = self._build_user_prompt(state)
        result: CriticOutput = self._call(
            [{"role": "user", "content": prompt}],
            schema=CriticOutput,
            model="critic",
        )
        issues: list[dict] = []
        # Return a short, stable hash that uniquely identifies an issue by its structural attributes.
        # The fingerprint is scoped to (scope_type, scope_id, issue_type, location) so the
        # supervisor can track whether the same logical problem recurs across review cycles.
        for issue in result.issues:
            issues.append(
                {
                    "issue_code": issue.issue_code,
                    "severity": issue.severity,
                    "issue_type": issue.issue_type,
                    "location": issue.location,
                    "fingerprint": hashlib.sha1(
                        f"{state['scope_type']}|{state['scope_id']}|{issue.issue_type}|{issue.location}".encode("utf-8")
                    ).hexdigest()[:12],
                    "affected_slide_numbers": issue.affected_slide_numbers,
                    "rewrite_instruction": issue.rewrite_instruction,
                }
            )
            
        rewrite_instructions = "\n".join(
            format_rewrite_instruction(issue) for issue in issues if issue["rewrite_instruction"].strip()
        )
        critic_result: CriticResultRecord = {
            "plan_generation": plan_gen,
            "dispatch_id": state["dispatch_id"],
            "assignment_id": state["assignment_id"],
            "cycle_number": state["cycle_number"],
            "check_type": "grounding_consistency",
            "scope_type": state["scope_type"],
            "scope_id": state["scope_id"],
            "group_idx": state["group_idx"],
            "target_slide_numbers": state.get("target_slide_numbers", []),
            "actionable": bool(result.actionable and issues),
            "rewrite_instructions": rewrite_instructions,
            "summary": result.summary,
            "issues": issues,
        }
        msg = (
            f"[critic] assignment={state['assignment_id']} "
            f"actionable={critic_result['actionable']} issues={len(issues)}"
        )
        with ResearchDatabase() as research_db:
            for issue in issues:
                research_db.save_review_event(
                    session_id=state["session_id"],
                    cycle_number=state["cycle_number"],
                    plan_generation=plan_gen,
                    scope_type=state["scope_type"],
                    scope_id=state["scope_id"],
                    check_type=state["check_type"],
                    assignment_id=state["assignment_id"],
                    issue_code=issue["issue_code"],
                    severity=issue["severity"],
                    location=issue["location"],
                    fingerprint=issue["fingerprint"],
                    rewrite_instruction_summary=issue["rewrite_instruction"],
                    affected_slide_numbers=issue.get("affected_slide_numbers") or None,
                )
            if not issues:
                research_db.save_review_event(
                    session_id=state["session_id"],
                    cycle_number=state["cycle_number"],
                    plan_generation=plan_gen,
                    scope_type=state["scope_type"],
                    scope_id=state["scope_id"],
                    check_type=state["check_type"],
                    assignment_id=state["assignment_id"],
                    decision="pass",
                )
                
        return Command(update={"critic_results": [critic_result], "messages": [msg]})


def critic_node(state: CriticDispatch | ResearchState) -> Command:
    """LangGraph node entry point for a critic assignment.

    Constructs a labelled SlideCriticAgent and delegates to its run() method.
    The type annotation is wider than the runtime type because LangGraph routes
    through ResearchState; the narrower CriticDispatch is always received at runtime.
    """

    # Modify log prefixes so parallel critics are easy to tell apart in logs.
    # Fall back to blueprint slide numbers when the supervisor did not narrow the assignment.
    nums = list(state.get("target_slide_numbers") or [])
    if not nums:
        for bp in state.get("slide_blueprints") or []:
            n = bp.get("slide_number")
            if n is not None:
                nums.append(n)
    if not nums:
        log_display = "Critic[empty]"
    else:
        ordered = sorted(int(n) for n in nums)
        lo, hi = ordered[0], ordered[-1]
        span = f"slides {lo}-{hi}" if lo != hi else f"slide {lo}"
        log_display = f"Critic[{span}, group {state.get('group_idx', '?')}]"

    # type: ignore[arg-type] is intentional — see docstring above.
    return SlideCriticAgent(log_display=log_display).run(state)  # type: ignore[arg-type]
