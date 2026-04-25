from __future__ import annotations

import hashlib
from typing import Any, Literal, TypedDict

from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.base import BaseLLMAgent
from src.agents.prompts.common import format_image_assets_block, format_slide_for_prompt
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
    rewrite_instructions: str


def _critic_log_label(state: CriticDispatch) -> str:
    """Match SlideWriter-style prefixes so parallel critics are easy to tell apart in logs.
        Fall back to blueprint slide numbers when the supervisor did not narrow the assignment."""
    nums = list(state.get("target_slide_numbers") or [])
    if not nums:
        for bp in state.get("slide_blueprints") or []:
            n = bp.get("slide_number")
            if n is not None:
                nums.append(n)
    if not nums:
        return "Critic[empty]"
    ordered = sorted(int(n) for n in nums)
    lo, hi = ordered[0], ordered[-1]
    span = f"slides {lo}-{hi}" if lo != hi else f"slide {lo}"
    return f"Critic[{span}, group {state.get('group_idx', '?')}]"


class CriticIssue(BaseModel):
    issue_code: str = Field(description="Unique issue id like ISS_001")
    severity: Literal["critical", "major", "minor"]
    issue_type: str
    location: str
    affected_slide_numbers: list[int] = Field(default_factory=list)
    rewrite_instruction: str = Field(description="Precise instruction describing how to fix the issue.")


class CriticOutput(BaseModel):
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


def _fingerprint(*, scope_type: str, scope_id: str, issue_type: str, location: str) -> str:
    # Stable issue fingerprints let the supervisor detect repeated problems across cycles.
    raw = f"{scope_type}|{scope_id}|{issue_type}|{location}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


class SlideCriticAgent(BaseLLMAgent):
    def __init__(self, *, log_display: str | None = None) -> None:
        super().__init__("critic", system_prompt=CRITIC_ROLE, log_display=log_display)

    def _load_session_retrieval_log(
        self, session_id: str, plan_generation: int | None = None
    ) -> str:
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

    def _load_slides(self, slide_numbers: list[int]) -> list[ProtoSlide]:
        slides: list[ProtoSlide] = []
        with ResearchDatabase() as research_db:
            for slide_number in slide_numbers:
                slide = research_db.load_slide(slide_number)
                if slide is not None:
                    slides.append(slide)
        return slides

    def _build_user_prompt(self, state: CriticDispatch) -> str:
        # Assemble slides, plan intent, baseline chunks, retrieved artifacts, and images in one review packet.
        slide_numbers = state.get("target_slide_numbers", [])
        slides = self._load_slides(slide_numbers)
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
                
                rows_by_id = {row["id"]: row for row in rows}
                ordered_texts = []
                for chunk_id in chunk_ids:
                    row = rows_by_id.get(chunk_id)
                    if row:
                        text = row["contextualized_text"] if row["contextualized_text"] else row["text"]
                        ordered_texts.append(f"--- Chunk ID: {row['id']} ---\n{text}")
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
        # Persist each issue before returning it so later supervisor cycles can detect recurrence.
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
        for issue in result.issues:
            issues.append(
                {
                    "issue_code": issue.issue_code,
                    "severity": issue.severity,
                    "issue_type": issue.issue_type,
                    "location": issue.location,
                    "fingerprint": _fingerprint(
                        scope_type=state["scope_type"],
                        scope_id=state["scope_id"],
                        issue_type=issue.issue_type,
                        location=issue.location,
                    ),
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
    # Type ignore is intentional: LangGraph supplies the narrower CriticDispatch at runtime.
    return SlideCriticAgent(log_display=_critic_log_label(state)).run(state)  # type: ignore[arg-type]
