"""
PlannerAgent
============
Reads all chunks for every ingested document from research.db, detects section
boundaries via Markdown heading analysis, presents the LLM with a human-readable
section outline (labels like S0, S1, ...), and asks it to produce a
LLMPresentationPlan in which every slide blueprint references sections by label.

Phase 2 (Python, no LLM) validates the result strictly and resolves section
labels → concrete chunk IDs before storing the final PresentationPlan in state.
If validation fails the entire LLM call is retried (up to PLAN_MAX_RETRIES).
"""
from __future__ import annotations

import json
import re
from typing import Literal

from langgraph.types import Command

from src.agents.base import BaseLLMAgent, schema_prompt_contract
from src.memory.research.schema import SlideContent
from src.state import (
    FIRST_CONTENT_SLIDE_NUMBER,
    LLMPresentationPlan,
    PresentationPlan,
    ResearchState,
    SlideBlueprint,
    SlideGroup,
    TITLE_SLIDE_NUMBER,
)
from src.memory.research.database import ResearchDatabase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAN_MAX_RETRIES = 2
MIN_GROUP_SIZE   = 2
MAX_GROUP_SIZE   = 7

# Matches ATX headings: # / ## / ### / #### at the start of a line
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def _planner_output_format() -> str:
    """Return a schema-derived output contract for planner structured JSON."""
    return schema_prompt_contract(
        LLMPresentationPlan,
        extra_rules=[
            "Do NOT wrap the plan in `presentation_plan` or any other outer key.",
            "Use only section labels that appear in the outline exactly as shown.",
            f"Each SlideGroup must contain between {MIN_GROUP_SIZE} and {MAX_GROUP_SIZE} slide_blueprints.",
            "Provide a top-level `title` with fewer than 7 words.",
            "Provide a top-level `subtitle` for the reserved title slide.",
        ],
    )


# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------

class _Section:
    """One detected section within a paper."""
    __slots__ = ("label", "heading", "chunk_ids", "word_count")

    def __init__(self, label: str, heading: str, chunk_ids: list[str], word_count: int):
        self.label      = label
        self.heading    = heading
        self.chunk_ids  = chunk_ids
        self.word_count = word_count


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def _detect_heading(chunk_text: str) -> str | None:
    """Return the heading text if the chunk's first meaningful line is an ATX heading."""
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING_RE.match(line):
            return re.sub(r"^#{1,4}\s+", "", stripped).strip()
        # First non-empty line is not a heading
        return None
    return None


# ---------------------------------------------------------------------------
# Section grouping
# ---------------------------------------------------------------------------

def _detect_sections(raw_chunks: list[dict], label_offset: int = 0) -> list[_Section]:
    """
    Group consecutive chunks into sections using heading analysis.
    label_offset ensures section labels are unique across multiple papers.
    """
    sections: list[_Section] = []
    current_ids: list[str]   = []
    current_heading          = "(no heading)"
    current_words            = 0

    for i, chunk in enumerate(raw_chunks):
        text    = chunk["text"] or ""
        heading = _detect_heading(text)
        is_boundary = (i == 0) or (heading is not None)

        if is_boundary and current_ids:
            label = f"S{label_offset + len(sections)}"
            sections.append(_Section(
                label=label,
                heading=current_heading,
                chunk_ids=current_ids,
                word_count=current_words,
            ))
            current_ids   = []
            current_words = 0

        if is_boundary:
            current_heading = heading or ("(no heading)" if i > 0 else "(no heading)")

        current_ids.append(chunk["id"])
        current_words += len(text.split())

    # Flush last section
    if current_ids:
        label = f"S{label_offset + len(sections)}"
        sections.append(_Section(
            label=label,
            heading=current_heading,
            chunk_ids=current_ids,
            word_count=current_words,
        ))

    return sections


# ---------------------------------------------------------------------------
# Outline formatting
# ---------------------------------------------------------------------------

def _build_outline(
    all_sections: list[_Section],
    paper_titles: list[str],
    doc_ids: list[str],
    sections_per_doc: list[int],
    max_slides: int,
    total_chunks: int,
) -> str:
    total_words = sum(s.word_count for s in all_sections)
    lines = [
        f"PAPER OUTLINE ({len(all_sections)} sections across {len(doc_ids)} paper(s), "
        f"{total_chunks} total chunks, ~{total_words} words)",
        f"Soft total deck target: {max_slides} slides (including the reserved title slide)",
        "",
    ]

    sec_idx = 0
    for doc_idx, (doc_id, count) in enumerate(zip(doc_ids, sections_per_doc)):
        title = paper_titles[doc_idx] if doc_idx < len(paper_titles) else doc_id
        lines.append(f'Paper {doc_idx + 1}: "{title}"')
        lines.append(f"  {'Label':<6}  {'Heading':<40}  {'Chunks':>6}  {'~Words':>7}")
        lines.append(f"  {'-'*6}  {'-'*40}  {'-'*6}  {'-'*7}")
        for _ in range(count):
            s = all_sections[sec_idx]
            heading_col = s.heading[:40].ljust(40)
            lines.append(f"  {s.label:<6}  {heading_col}  {len(s.chunk_ids):>6}  {s.word_count:>7}")
            sec_idx += 1
        lines.append("")

    lines += [
        "INSTRUCTIONS:",
        "- Reference sections only by their label (e.g. S0, S3). Do NOT invent labels.",
        f"- Each SlideGroup must contain between {MIN_GROUP_SIZE} and {MAX_GROUP_SIZE} slides.",
        "- A slide may reference sections from different papers.",
        f"- Slide {TITLE_SLIDE_NUMBER} is a reserved title slide created automatically in Python. Your blueprints MUST start at slide_number {FIRST_CONTENT_SLIDE_NUMBER}.",
        f"- Treat {max_slides} as the total deck size, including the reserved title slide.",
        f"- Keep content slides within slide numbers {FIRST_CONTENT_SLIDE_NUMBER} through {max_slides}.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_llm_plan(
    plan: LLMPresentationPlan,
    valid_labels: set[str],
    max_slides: int,
) -> list[str]:
    """Return a list of validation failure messages (empty = valid)."""
    failures: list[str] = []
    seen_slide_numbers: set[int] = set()
    slide_numbers_in_order: list[int] = []
    title_word_count = len(plan.title.split())

    if not plan.title.strip():
        failures.append("Reserved title slide title must not be empty.")
    elif title_word_count >= 7:
        failures.append(
            f"Reserved title slide title must be fewer than 7 words; received {title_word_count}: '{plan.title}'."
        )

    if not plan.subtitle.strip():
        failures.append("Reserved title slide subtitle must not be empty.")

    for gi, group in enumerate(plan.slide_groups):
        n = len(group.slide_blueprints)
        if n < MIN_GROUP_SIZE:
            failures.append(
                f"Group {gi} has {n} blueprint(s) — minimum is {MIN_GROUP_SIZE}."
            )
        elif n > MAX_GROUP_SIZE:
            failures.append(
                f"Group {gi} has {n} blueprints — maximum is {MAX_GROUP_SIZE}."
            )

        for bi, bp in enumerate(group.slide_blueprints):
            slide_numbers_in_order.append(bp.slide_number)

            if bp.slide_number < FIRST_CONTENT_SLIDE_NUMBER:
                failures.append(
                    f"Group {gi} blueprint {bi} has slide_number {bp.slide_number}, but content slides must start at {FIRST_CONTENT_SLIDE_NUMBER}."
                )
            if bp.slide_number > max_slides:
                failures.append(
                    f"Group {gi} blueprint {bi} has slide_number {bp.slide_number}, which exceeds total deck max_slides={max_slides}."
                )
            if bp.slide_number in seen_slide_numbers:
                failures.append(
                    f"Group {gi} blueprint {bi} reuses slide_number {bp.slide_number}; slide numbers must be unique."
                )
            seen_slide_numbers.add(bp.slide_number)
            if not bp.source_sections:
                failures.append(
                    f"Group {gi} blueprint {bi} (slide {bp.slide_number}) "
                    f"has no source_sections."
                )
            for label in bp.source_sections:
                if label not in valid_labels:
                    failures.append(
                        f"Group {gi} blueprint {bi} references unknown section '{label}'."
                    )

    if slide_numbers_in_order and slide_numbers_in_order != sorted(slide_numbers_in_order):
        failures.append("Slide blueprints must be ordered by ascending slide_number across the full plan.")
    if slide_numbers_in_order:
        expected_numbers = list(
            range(
                FIRST_CONTENT_SLIDE_NUMBER,
                FIRST_CONTENT_SLIDE_NUMBER + len(slide_numbers_in_order),
            )
        )
        if slide_numbers_in_order != expected_numbers:
            failures.append(
                f"Content slide numbers must be contiguous starting at {FIRST_CONTENT_SLIDE_NUMBER}: expected {expected_numbers}, received {slide_numbers_in_order}."
            )

    return failures


def _build_title_slide_content(*, title: str, subtitle: str, thesis: str, target_audience: str) -> SlideContent:
    clean_title = " ".join(title.split())
    clean_subtitle = " ".join(subtitle.split())
    clean_thesis = " ".join(thesis.split())
    clean_audience = " ".join(target_audience.split())
    key_message = clean_thesis or clean_subtitle or clean_title
    notes = clean_thesis
    if clean_audience:
        notes = f"{notes}\n\nAudience: {clean_audience}" if notes else f"Audience: {clean_audience}"

    return SlideContent(
        title=clean_title or "Research Presentation",
        subtitle=clean_subtitle or None,
        key_message=key_message,
        bullets=[],
        speaker_notes=notes,
        layout="title_slide",
        narrative_role="hook",
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PlannerAgent(BaseLLMAgent):
    def __init__(self) -> None:
        super().__init__("planner")

    def run(self, state: ResearchState) -> Command[Literal["plan_executor"]]:
        self._set_session_id(state)

        doc_ids      = state.get("doc_ids", [])
        paper_titles = state.get("paper_titles", [])
        max_slides   = state.get("max_slides", 15)
        query        = state.get("query", "Explain this paper to an audience of laypeople")

        if not doc_ids:
            raise ValueError("[Planner] No doc_ids in state — did PDF ingestion succeed?")

        # ------------------------------------------------------------------
        # Phase 1a: load chunks and detect sections for every paper
        # ------------------------------------------------------------------
        all_sections:    list[_Section] = []
        sections_per_doc: list[int]     = []
        total_chunks = 0

        with ResearchDatabase() as db:
            for doc_id in doc_ids:
                raw_chunks = db.get_chunks_for_dispatch(doc_id)
                total_chunks += len(raw_chunks)
                if not raw_chunks:
                    self._logger.log(
                        f"[Planner] Warning: no chunks for doc_id={doc_id}", level="warning"
                    )
                    sections_per_doc.append(0)
                    continue

                doc_sections = _detect_sections(raw_chunks, label_offset=len(all_sections))
                all_sections.extend(doc_sections)
                sections_per_doc.append(len(doc_sections))

        if not all_sections:
            raise ValueError("[Planner] No sections detected across all documents.")

        valid_labels   = {s.label for s in all_sections}
        section_map    = {s.label: s.chunk_ids for s in all_sections}

        self._logger.log(
            f"[Planner] {len(all_sections)} sections across {len(doc_ids)} doc(s), "
            f"{total_chunks} total chunks"
        )

        # ------------------------------------------------------------------
        # Phase 1b: build outline string for LLM
        # ------------------------------------------------------------------
        outline = _build_outline(
            all_sections, paper_titles, doc_ids,
            sections_per_doc, max_slides, total_chunks,
        )

        user_prompt = (
            f"USER QUERY:\n{query}\n\n"
            f"{outline}\n\n"
            "Produce a PresentationPlan for the above paper(s) based on the user query.\n\n"
            f"{_planner_output_format()}"
        )
        turns = [{"role": "user", "content": user_prompt}]

        # ------------------------------------------------------------------
        # Phase 1c + Phase 2: LLM call + strict validation
        # ------------------------------------------------------------------
        llm_plan_result = self._call_structured(
            turns,
            LLMPresentationPlan,
            max_retries=PLAN_MAX_RETRIES,
            model="planner",
            runtime_validator=lambda plan: _validate_llm_plan(plan, valid_labels, max_slides),
        )
        llm_plan: LLMPresentationPlan = llm_plan_result.parsed

        # ------------------------------------------------------------------
        # Phase 2: resolve section labels → chunk IDs
        # ------------------------------------------------------------------
        resolved_groups: list[SlideGroup] = []

        title_blueprint = SlideBlueprint(
            slide_number=TITLE_SLIDE_NUMBER,
            slide_kind="title",
            working_title="Title Slide",
            narrative_role="hook",
            intent="Reserved title slide authored by Planner.",
            source_chunk_ids=[],
            prebuilt_content=_build_title_slide_content(
                title=llm_plan.title,
                subtitle=llm_plan.subtitle,
                thesis=llm_plan.thesis,
                target_audience=llm_plan.target_audience,
            ),
        )
        resolved_groups.append(SlideGroup(
            slide_blueprints=[title_blueprint],
            rationale="Title slide authored by Planner.",
        ))

        for group in llm_plan.slide_groups:
            resolved_blueprints: list[SlideBlueprint] = []
            for bp in group.slide_blueprints:
                chunk_ids: list[str] = []
                for label in bp.source_sections:
                    chunk_ids.extend(section_map.get(label, []))
                resolved_blueprints.append(SlideBlueprint(
                    slide_number=bp.slide_number,
                    slide_kind="content",
                    working_title=bp.working_title,
                    narrative_role=bp.narrative_role,
                    intent=bp.intent,
                    source_chunk_ids=chunk_ids,
                ))
            resolved_groups.append(SlideGroup(
                slide_blueprints=resolved_blueprints,
                rationale=group.rationale,
            ))

        presentation_plan = PresentationPlan(
            thesis=llm_plan.thesis,
            target_audience=llm_plan.target_audience,
            estimated_duration_minutes=llm_plan.estimated_duration_minutes,
            narrative_arc_summary=llm_plan.narrative_arc_summary,
            slide_groups=resolved_groups,
            reasoning=llm_plan.reasoning,
        )

        total_slides = sum(
            len(g.slide_blueprints) for g in presentation_plan.slide_groups
        )
        msg = (
            f"[Planner] Plan created: {total_slides} slides across "
            f"{len(presentation_plan.slide_groups)} group(s). "
            f"Thesis: {presentation_plan.thesis}"
        )
        self._logger.log(msg)

        return Command(
            update={
                "presentation_plan": presentation_plan,
                "messages": [msg],
            }
        )


def planner_node(state: ResearchState) -> Command:
    return PlannerAgent().run(state)
