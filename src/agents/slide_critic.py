from __future__ import annotations

import hashlib
import json
from typing import Literal, TypedDict

from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.base import BaseLLMAgent
from src.memory.research.database import ResearchDatabase
from src.memory.research.schema import ProtoSlide
from src.state import (
    CriticResultRecord,
    ResearchState,
    ReviewAssignment,
    TITLE_SLIDE_NUMBER,
)


class CriticDispatch(TypedDict):
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


class CriticIssue(BaseModel):
    issue_code: str = Field(description="Unique issue id like ISS_001")
    severity: Literal["critical", "major", "minor"]
    issue_type: str
    location: str
    description: str
    affected_slide_numbers: list[int] = Field(default_factory=list)
    rewrite_instruction: str = Field(description="Precise instruction describing how to fix the issue.")


class CriticOutput(BaseModel):
    summary: str
    actionable: bool
    issues: list[CriticIssue] = Field(default_factory=list)


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
    def __init__(self) -> None:
        super().__init__("critic")

    def _load_chunks(self, chunk_ids: list[str]) -> str:
        if not chunk_ids:
            return ""
        with ResearchDatabase() as research_db:
            placeholders = ",".join(["?"] * len(chunk_ids))
            rows = research_db.connection.execute(
                f"SELECT id, text, contextualized_text FROM text_chunks WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        ordered = []
        for cid in chunk_ids:
            row = by_id.get(cid)
            if row is None:
                continue
            text = row["contextualized_text"] if row["contextualized_text"] else row["text"]
            ordered.append(f"--- Chunk ID: {cid} ---\n{text}")
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
        slide_numbers = state.get("target_slide_numbers", [])
        if slide_numbers == [TITLE_SLIDE_NUMBER]:
            return (
                "Assigned check is a title-slide grounding review.\n"
                "Return actionable=false with an empty issue list unless explicit factual errors are present.\n"
            )

        slides = self._load_slides(slide_numbers)
        chunks = self._load_chunks(state.get("chunk_ids", []))
        blueprints = state.get("slide_blueprints", [])
        blueprint_block = "\n".join(
            f"Slide {bp.get('slide_number')}: {bp.get('working_title', '')} — {bp.get('intent', '')}"
            for bp in blueprints
            if bp.get("slide_number") in set(slide_numbers)
        )
        slides_block = "\n\n".join(_format_slide(slide) for slide in slides) or "No slides found."
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
                "SOURCE CHUNKS:",
                chunks or "(none)",
                "",
                "Identify only meaningful issues. If no changes are needed, return actionable=false and an empty issue list.",
            ]
        )

    def run(self, state: CriticDispatch) -> Command:
        self._set_session_id(state)
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
        rewrite_instructions = "\n".join(
            f"- {issue['rewrite_instruction']}" for issue in issues if issue["rewrite_instruction"].strip()
        )
        critic_result: CriticResultRecord = {
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
                    scope_type=state["scope_type"],
                    scope_id=state["scope_id"],
                    check_type=state["check_type"],
                    assignment_id=state["assignment_id"],
                    issue_code=issue["issue_code"],
                    severity=issue["severity"],
                    fingerprint=issue["fingerprint"],
                    rewrite_instruction_summary=issue["rewrite_instruction"],
                )
            if not issues:
                research_db.save_review_event(
                    session_id=state["session_id"],
                    cycle_number=state["cycle_number"],
                    scope_type=state["scope_type"],
                    scope_id=state["scope_id"],
                    check_type=state["check_type"],
                    assignment_id=state["assignment_id"],
                    decision="pass",
                )
        return Command(update={"critic_results": [critic_result], "messages": [msg]})


def critic_node(state: CriticDispatch | ResearchState) -> Command:
    return SlideCriticAgent().run(state)  # type: ignore[arg-type]
