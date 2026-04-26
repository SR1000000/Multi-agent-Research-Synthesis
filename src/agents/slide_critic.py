from __future__ import annotations

import hashlib
from typing import Literal, TypedDict, cast

from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.base import BaseLLMAgent
from src.agents.prompts.common import format_image_assets_block, format_slide_for_prompt, ordered_chunk_texts
from src.agents.prompts.critic_prompts import (
    CRITIC_ROLE,
    GROUNDING_REVIEW_CRITERIA,
    NARRATIVE_REVIEW_CRITERIA,
    build_grounding_critic_user_prompt,
    build_narrative_critic_user_prompt,
    format_retrieved_artifact_row,
    format_rewrite_instruction,
)
from src.memory.research.database import ResearchDatabase
from src.memory.research.schema import ProtoSlide
from src.state import CriticResultRecord, ResearchState, ReviewCheckType


class CriticDispatch(TypedDict):
    """State payload delivered to a critic node via LangGraph's Send() API.

    Carries scope, plan metadata, chunk IDs and blueprints (for grounding), and
    the slides to review. Narrative reviews may fill slide lists from blueprints
    when ``target_slide_numbers`` is empty.
    """

    plan_number: int
    dispatch_id: str
    assignment_id: str
    cycle_number: int
    session_id: str
    check_type: ReviewCheckType
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
    location: str = Field(description="Pinpoint what to change (e.g. slide number and bullet or heading).")
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
    """Base critic: shared LLM call, issue fingerprinting, DB persistence, and result shaping.
    Each instance is dispatched for a single critic assignment. Results are
    fingerprinted and persisted to the research database so the supervisor can detect
    recurring problems across cycles and avoid infinite rewrite loops on stable findings.

    Subclasses supply ``review_criteria`` and implement ``_build_user_prompt`` (and
    optionally ``_resolved_target_slide_numbers`` for what was actually in scope).
    """

    def __init__(self, *, log_display: str | None = None, review_criteria: str) -> None:
        """Initialise with the critic role and review criteria system prompt.
         ``log_display`` labels parallel critic logs."""
        super().__init__("critic", system_prompt=CRITIC_ROLE + review_criteria, log_display=log_display)

    def _resolved_target_slide_numbers(self, state: CriticDispatch) -> list[int]:
        """Slide numbers in review scope; default is explicit ``target_slide_numbers`` only."""
        return list(state.get("target_slide_numbers") or [])

    def _load_slides(self, slide_numbers: list[int]) -> list[ProtoSlide]:
        """Shared helper to load slides from the database."""
        slides: list[ProtoSlide] = []
        with ResearchDatabase() as research_db:
            for slide_number in slide_numbers:
                slide = research_db.load_slide(slide_number)
                if slide is not None:
                    slides.append(slide)
        return slides

    def _format_blueprint_block(
        self,
        blueprints: list[dict],
        target_slide_numbers: list[int],
        *,
        all_blueprints_if_no_target: bool = False,
    ) -> str:
        """Shared helper to format the SLIDE ASSIGNMENTS text block."""
        if not target_slide_numbers:
            bps = list(blueprints) if all_blueprints_if_no_target else []
        else:
            want = set(target_slide_numbers)
            bps = [bp for bp in blueprints if bp.get("slide_number") in want]
        return "\n".join(
            f"Slide {bp.get('slide_number')}: {bp.get('working_title', '')} — {bp.get('intent', '')}"
            for bp in bps
        )

    def _build_user_prompt(self, state: CriticDispatch) -> str:
        """Subclasses assemble the user message (grounding vs narrative, etc.)."""
        raise NotImplementedError

    def run(self, state: CriticDispatch) -> Command:
        """Execute the critic review for one assignment and return a LangGraph Command.

        Calls the LLM with the assembled review prompt, converts each issue to a
        fingerprinted dict, and persists every issue (or a pass event when none are found)
        to the research database.  Returns a Command carrying a CriticResultRecord for the
        Supervisor to evaluate on its next invocation.
        """
        self._set_session_id(state)
        self._set_plan_number(state)
        plan_num = int(state.get("plan_number", 1))
        check_type = state["check_type"]
        reviewed_slide_numbers = self._resolved_target_slide_numbers(state)
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
            "dispatch_id": state["dispatch_id"],
            "assignment_id": state["assignment_id"],
            "cycle_number": state["cycle_number"],
            "check_type": check_type,
            "scope_type": state["scope_type"],
            "scope_id": state["scope_id"],
            "group_idx": state["group_idx"],
            "target_slide_numbers": reviewed_slide_numbers,
            "actionable": bool(result.actionable and issues),
            "rewrite_instructions": rewrite_instructions,
            "summary": result.summary,
            "issues": issues,
        }
        msg = (
            f"[critic] check_type={check_type} assignment={state['assignment_id']} "
            f"actionable={critic_result['actionable']} issues={len(issues)}"
        )
        with ResearchDatabase() as research_db:
            for issue in issues:
                research_db.save_review_event(
                    session_id=state["session_id"],
                    cycle_number=state["cycle_number"],
                    plan_number=plan_num,
                    scope_type=state["scope_type"],
                    scope_id=state["scope_id"],
                    check_type=check_type,
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
                    plan_number=plan_num,
                    scope_type=state["scope_type"],
                    scope_id=state["scope_id"],
                    check_type=check_type,
                    assignment_id=state["assignment_id"],
                    decision="pass",
                )

        return Command(update={"critic_results": [critic_result], "messages": [msg]})


class GroundingCriticAgent(SlideCriticAgent):
    """Grounding / consistency: baseline chunks, in-session retrieval log, and image assets."""

    def __init__(self, *, log_display: str | None = None) -> None:
        super().__init__(log_display=log_display, review_criteria=GROUNDING_REVIEW_CRITERIA)

    def _load_session_retrieval_log(
        self, session_id: str, plan_number: int | None = None
    ) -> str:
        if not session_id:
            return ""
        with ResearchDatabase() as research_db:
            rows = research_db.load_normalized_artifacts_for_session(
                session_id, plan_number=plan_number
            )
        return "\n\n".join(format_retrieved_artifact_row(row) for row in rows)

    def _build_user_prompt(self, state: CriticDispatch) -> str:
        slide_numbers = self._resolved_target_slide_numbers(state)
        slides = self._load_slides(slide_numbers)
        pnum = int(state.get("plan_number", 1))
        retrieval_log = self._load_session_retrieval_log(state.get("session_id", ""), plan_number=pnum)
        blueprints = state.get("slide_blueprints", [])
        blueprint_block = self._format_blueprint_block(
            blueprints, slide_numbers, all_blueprints_if_no_target=False
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
                ordered_texts = ordered_chunk_texts(rows, chunk_ids)
                baseline_chunks_block = "\n\n".join(ordered_texts)
        image_block = format_image_assets_block(image_metadatas)
        return build_grounding_critic_user_prompt(
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


class NarrativeCriticAgent(SlideCriticAgent):
    """Narrative coherence: proto-slides and plan intent only (no source chunks, RAG log, or image metadata)."""

    def __init__(self, *, log_display: str | None = None) -> None:
        super().__init__(log_display=log_display, review_criteria=NARRATIVE_REVIEW_CRITERIA)

    def _resolved_target_slide_numbers(self, state: CriticDispatch) -> list[int]:
        raw = list(state.get("target_slide_numbers") or [])
        if raw:
            return raw
        return [
            int(n)
            for n in (bp.get("slide_number") for bp in (state.get("slide_blueprints") or []))
            if n is not None
        ]

    def _build_user_prompt(self, state: CriticDispatch) -> str:
        slide_numbers = self._resolved_target_slide_numbers(state)
        slides = self._load_slides(slide_numbers)
        blueprints = state.get("slide_blueprints", [])
        blueprint_block = self._format_blueprint_block(
            blueprints, slide_numbers, all_blueprints_if_no_target=not slide_numbers
        )
        slides_block = "\n\n".join(format_slide_for_prompt(slide) for slide in slides) or "No slides found."
        return build_narrative_critic_user_prompt(
            cycle_number=state["cycle_number"],
            check_type=state["check_type"],
            scope_type=state["scope_type"],
            scope_id=state["scope_id"],
            target_slide_numbers=slide_numbers,
            blueprint_block=blueprint_block,
            slides_block=slides_block,
            output_model=CriticOutput,
        )


_CRITIC_AGENT_MAP: dict[ReviewCheckType, type[SlideCriticAgent]] = {
    "grounding_consistency": GroundingCriticAgent,
    "narrative_coherence": NarrativeCriticAgent,
}


def _critic_class_for(check_type: str) -> type[SlideCriticAgent]:
    if check_type not in _CRITIC_AGENT_MAP:
        expected = tuple(_CRITIC_AGENT_MAP)
        raise KeyError(
            f"Unknown critic check_type {check_type!r}; expected one of {expected}"
        )
    return _CRITIC_AGENT_MAP[cast(ReviewCheckType, check_type)]


def critic_node(state: CriticDispatch | ResearchState) -> Command:
    """LangGraph node entry point for a critic assignment.
    Constructs a labelled SlideCriticAgent and delegates to its run() method.
    The type annotation is wider than the runtime type because LangGraph routes
    through ResearchState; the narrower CriticDispatch is always received at runtime.
    """

    check_type = str(state.get("check_type") or "grounding_consistency")

    # Modify log prefixes so parallel critics are easy to tell apart in logs.
    # Fall back to blueprint slide numbers when the supervisor did not narrow the assignment.
    nums = list(state.get("target_slide_numbers") or [])
    if not nums:
        for bp in state.get("slide_blueprints") or []:
            n = bp.get("slide_number")
            if n is not None:
                nums.append(n)
    if not nums:
        log_display = f"Critic[empty, {check_type}]"
    else:
        ordered = sorted(int(n) for n in nums)
        lo, hi = ordered[0], ordered[-1]
        span = f"slides {lo}-{hi}" if lo != hi else f"slide {lo}"
        log_display = f"Critic[{check_type}, {span}, group {state.get('group_idx', '?')}]"

    agent_cls = _critic_class_for(check_type)
    return agent_cls(log_display=log_display).run(state)  # type: ignore[arg-type]
