from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, TypedDict

from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents._image_utils import format_image_assets_block
from src.agents.base import BaseLLMAgent, schema_prompt_contract
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
    """Match SlideWriter-style prefixes so parallel critics are easy to tell apart in logs."""
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
    description: str
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


def _critic_output_format() -> str:
    """Schema-derived JSON contract for critic structured output (matches planner/writer pattern)."""
    return schema_prompt_contract(
        CriticOutput,
        extra_rules=[
            "Top-level keys MUST be exactly `summary`, `actionable`, and `issues` — do not wrap the payload in another key.",
            "If no meaningful issues exist, set actionable=false and issues=[].",
            "If one or more issues exist, set actionable=true and include every required field on each issue "
            "(issue_code, severity, issue_type, location, description, rewrite_instruction).",
            "issue_code values must be unique within this response (e.g. ISS_001, ISS_002).",
            "Use the exact field names issue_code and issue_type — not `id`, `classification`, or other synonyms.",
            "location must pinpoint what to change (e.g. slide number and bullet or heading).",
            "rewrite_instruction must be one concrete edit directive per issue, not only a restatement of the problem.",
        ],
    )


def _format_slide(slide: ProtoSlide) -> str:
    return json.dumps(
        {
            "slide_number": slide.slide_number,
            "content": slide.content.model_dump(mode="json"),
            "chunk_references": slide.chunk_references,
        },
        indent=2,
    )


def _fingerprint(*, scope_type: str, scope_id: str, issue_type: str, location: str) -> str:
    raw = f"{scope_type}|{scope_id}|{issue_type}|{location}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


class SlideCriticAgent(BaseLLMAgent):
    def __init__(self, *, log_display: str | None = None) -> None:
        super().__init__("critic", log_display=log_display)

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
            ordered.append(self._format_retrieved_artifact(row))
        return "\n\n".join(ordered)

    def _format_retrieved_artifact(self, row: dict[str, Any]) -> str:
        kind = str(row.get("kind", "chunk"))
        artifact_id = str(row.get("artifact_id") or "")
        call_id = str(row.get("call_id") or "")
        document_id = str(row.get("document_id") or "")
        score = row.get("score")
        contextualized = str(row.get("contextualized_text") or "").strip()
        text = str(row.get("text") or "").strip()
        caption = str(row.get("caption") or "").strip()
        lines = [
            f"--- Retrieved {kind} {artifact_id} (call_id={call_id}, doc={document_id}, score={score}) ---"
        ]
        if caption:
            lines.append(f"caption: {caption}")
        if contextualized:
            lines.append(f"contextualized description: {contextualized}")
        if text:
            lines.append(f"value: {text}")
        return "\n".join(lines)

    def _load_slides(self, slide_numbers: list[int]) -> list[ProtoSlide]:
        slides: list[ProtoSlide] = []
        with ResearchDatabase() as research_db:
            for slide_number in slide_numbers:
                slide = research_db.load_slide(slide_number)
                if slide is not None:
                    slides.append(slide)
        return slides

    def _build_user_prompt(self, state: CriticDispatch) -> str:
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
        slides_block = "\n\n".join(_format_slide(slide) for slide in slides) or "No slides found."
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
        return "\n".join(
            [
                f"Cycle: {state['cycle_number']}",
                f"Check type: {state['check_type']}",
                f"Scope: {state['scope_type']}::{state['scope_id']}",
                f"Target slides: {slide_numbers}",
                "",
                "SLIDE ASSIGNMENTS:",
                blueprint_block or "(none)",
                "",
                "CURRENT SLIDES:",
                slides_block,
                "",
                "BASELINE SOURCE MATERIAL (Provided to writer):",
                baseline_chunks_block or "(none)",
                "",
                "IN-SESSION RETRIEVAL LOG (Dynamically gathered by writer):",
                retrieval_log or "(none)",
                "",
                "AVAILABLE IMAGE ASSETS:",
                image_block or "(none)",
                "",
                "Identify only significant issues that break grounding, clarity, coherence, or the review criteria. "
                "If no changes are needed, set actionable=false and issues=[].",
                "Review the slides against the BASELINE SOURCE MATERIAL and IN-SESSION RETRIEVAL LOG. Treat this combined evidence as the source of truth for grounding checks.",
                "If the combined evidence is missing support for a concrete claim on a slide, treat that as a grounding issue.",
                "Identify only significant issues that break grounding, clarity, coherence, or the review criteria. If no changes are needed, set actionable=false and issues=[].",
                "",
                _critic_output_format(),
            ]
        )

    def run(self, state: CriticDispatch) -> Command:
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
                    "description": issue.description,
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
        def _fmt_instruction(issue: dict) -> str:
            slide_nums = issue.get("affected_slide_numbers") or []
            loc = issue.get("location", "").strip()
            
            context_parts = []
            if slide_nums:
                context_parts.append(f"Slide(s) {', '.join(str(n) for n in slide_nums)}")
            if loc and loc.lower() not in ("none", "n/a", "general", "all"):
                context_parts.append(f"Location: {loc}")
                
            prefix = f"[{' | '.join(context_parts)}] " if context_parts else ""
            return f"- {prefix}{issue['rewrite_instruction']}"

        rewrite_instructions = "\n".join(
            _fmt_instruction(issue) for issue in issues if issue["rewrite_instruction"].strip()
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
                    description=issue["description"],
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
    return SlideCriticAgent(log_display=_critic_log_label(state)).run(state)  # type: ignore[arg-type]
