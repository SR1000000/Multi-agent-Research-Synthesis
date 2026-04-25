"""
PlannerAgent
============
Reads all chunks for every ingested document from research.db, detects section
boundaries via Markdown heading analysis, builds a compact planning brief for
each paper (outline + abstract/overview + representative section snippets), and
asks the LLM to produce a LLMPresentationPlan in which every slide blueprint
references sections by label.

Phase 2 (Python, no LLM) validates the result strictly and resolves section
labels → concrete chunk IDs before storing the final PresentationPlan in state.
If validation fails the entire LLM call is retried (up to PLAN_MAX_RETRIES).
"""
from __future__ import annotations

import json
import re
from typing import Literal

from langgraph.types import Command

from src.agents.base import BaseLLMAgent
from src.agents.prompts.planner_prompts import PLANNER_ROLE, build_planner_output_format
from src.state import (
    LLMPresentationPlan,
    PresentationPlan,
    ResearchState,
    SlideBlueprint,
    SlideGroup,
)
from src.memory.research.database import ResearchDatabase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAN_MAX_RETRIES = 2
MIN_GROUP_SIZE   = 2
MAX_GROUP_SIZE   = 7
PAPER_SUMMARY_MAX_CHARS = 900
SECTION_SNIPPET_MAX_CHARS = 420
SECTION_SNIPPET_PER_CHUNK_CHARS = 180

# Matches ATX headings: # / ## / ### / #### at the start of a line
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")


# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------

class _Section:
    """One detected section within a paper."""
    __slots__ = ("label", "heading", "chunk_ids", "word_count", "snippet")

    def __init__(
        self,
        label: str,
        heading: str,
        chunk_ids: list[str],
        word_count: int,
        snippet: str,
    ):
        """Store lightweight section metadata; full chunk text remains in research.db."""
        self.label      = label
        self.heading    = heading
        self.chunk_ids  = chunk_ids
        self.word_count = word_count
        self.snippet    = snippet


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


def _normalize_planner_text(text: str) -> str:
    """Collapse markdown-ish chunk text into one readable snippet line.
    Remove headings from snippets because headings already appear separately in the outline."""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING_RE.match(line):
            continue
        cleaned_lines.append(stripped)

    cleaned = " ".join(cleaned_lines).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _truncate_text(text: str, max_chars: int) -> str:
    """Trim text to at most max_chars without splitting mid-clause.

    Tries natural punctuation boundaries (". ", "; ", ": ", ", ") before falling back
    to a word boundary, keeping prompt snippets readable even when truncated.
    """
    clean = " ".join(text.split()).strip()
    if len(clean) <= max_chars:
        return clean

    # Prefer natural punctuation boundaries so prompt snippets remain readable.
    floor = max(int(max_chars * 0.6), 1)
    for sep in (". ", "; ", ": ", ", "):
        idx = clean.rfind(sep, floor, max_chars + 1)
        if idx != -1:
            return clean[: idx + len(sep.strip())].rstrip()

    cut = clean.rfind(" ", floor, max_chars + 1)
    if cut == -1:
        cut = max_chars
    return clean[:cut].rstrip() + "..."


def _chunk_planner_text(chunk: dict) -> str:
    """Prefer contextualized chunk text when available for planning."""
    contextualized = (chunk.get("contextualized_text") or "").strip()
    raw = (chunk.get("text") or "").strip()
    return contextualized or raw


def _build_section_snippet(chunks: list[dict]) -> str:
    """Create a compact section preview from representative chunk texts."""
    if not chunks:
        return ""

    # Sample first/middle/last chunks to keep long sections represented without bloating prompts.
    candidate_indexes = sorted({0, len(chunks) // 2, len(chunks) - 1})
    snippets: list[str] = []
    seen: set[str] = set()

    for idx in candidate_indexes:
        snippet = _normalize_planner_text(_chunk_planner_text(chunks[idx]))
        if not snippet:
            continue
        snippet = _truncate_text(snippet, SECTION_SNIPPET_PER_CHUNK_CHARS)
        key = snippet.casefold()
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet)

    return _truncate_text(" | ".join(snippets), SECTION_SNIPPET_MAX_CHARS)


def _build_paper_summary(raw_chunks: list[dict], paper_abstract: str) -> str:
    """Create a short paper-level summary for the planner prompt.
        The abstract is trusted first; otherwise the first chunks approximate the paper overview."""
    abstract = _normalize_planner_text(paper_abstract)
    if abstract:
        return _truncate_text(abstract, PAPER_SUMMARY_MAX_CHARS)

    snippets: list[str] = []
    seen: set[str] = set()
    for chunk in raw_chunks[:3]:
        snippet = _normalize_planner_text(_chunk_planner_text(chunk))
        if not snippet:
            continue
        snippet = _truncate_text(snippet, SECTION_SNIPPET_PER_CHUNK_CHARS)
        key = snippet.casefold()
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet)

    return _truncate_text(" | ".join(snippets), PAPER_SUMMARY_MAX_CHARS) if snippets else "(no paper summary available)"


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
    current_chunks: list[dict] = []

    for i, chunk in enumerate(raw_chunks):
        text    = chunk["text"] or ""
        heading = _detect_heading(text)
        # A new heading starts a new section; headingless chunks stay attached to the prior section.
        is_boundary = (i == 0) or (heading is not None)

        if is_boundary and current_ids:
            label = f"S{label_offset + len(sections)}"
            sections.append(_Section(
                label=label,
                heading=current_heading,
                chunk_ids=current_ids,
                word_count=current_words,
                snippet=_build_section_snippet(current_chunks),
            ))
            current_ids   = []
            current_words = 0
            current_chunks = []

        if is_boundary:
            current_heading = heading or ("(no heading)" if i > 0 else "(no heading)")

        current_ids.append(chunk["id"])
        current_words += len(text.split())
        current_chunks.append(chunk)

    # Flush last section
    if current_ids:
        label = f"S{label_offset + len(sections)}"
        sections.append(_Section(
            label=label,
            heading=current_heading,
            chunk_ids=current_ids,
            word_count=current_words,
            snippet=_build_section_snippet(current_chunks),
        ))

    return sections


# ---------------------------------------------------------------------------
# Outline formatting
# ---------------------------------------------------------------------------

def _build_outline(
    all_sections: list[_Section],
    paper_contexts: list[dict[str, str]],
    sections_per_doc: list[int],
    max_slides: int,
    total_chunks: int,
) -> str:
    """Format a compact paper outline string for the LLM planning prompt.

    Includes paper-level summaries, per-section headings with word counts and representative
    snippets, a slide-count target, and the structural constraint instructions.  Kept compact
    to avoid bloating the context window while giving the LLM enough narrative signal to choose
    a coherent arc across one or more papers.
    """
    total_words = sum(s.word_count for s in all_sections)
    lines = [
        f"PAPER OUTLINE ({len(all_sections)} sections across {len(paper_contexts)} paper(s), "
        f"{total_chunks} total chunks, ~{total_words} words)",
        f"Soft total deck target: {max_slides} slides (title slide is added automatically; "
        f"content slide numbers run 1 through {max_slides - 1})",
        "",
    ]

    sec_idx = 0
    for doc_idx, (paper_context, count) in enumerate(zip(paper_contexts, sections_per_doc)):
        title = paper_context["title"]
        lines.append(f'Paper {doc_idx + 1}: "{title}"')
        lines.append(f"  Doc ID: {paper_context['doc_id']}")
        lines.append(f"  Paper summary: {paper_context['summary']}")
        lines.append("  Sections:")
        for _ in range(count):
            s = all_sections[sec_idx]
            lines.append(
                f"  - {s.label} | {s.heading} | {len(s.chunk_ids)} chunk(s) | ~{s.word_count} words"
            )
            lines.append(f"    Snippet: {s.snippet or '(no representative snippet available)'}")
            sec_idx += 1
        lines.append("")

    lines += [
        "INSTRUCTIONS:",
        "- Reference sections only by their label (e.g. S0, S3). Do NOT invent labels.",
        "- Use the paper summaries and section snippets to infer the thesis and storyline, not just the headings.",
        f"- Each SlideGroup must contain between {MIN_GROUP_SIZE} and {MAX_GROUP_SIZE} slides.",
        "- A slide may reference sections from different papers.",
        f"- The title slide is generated automatically from the `title` and `subtitle` fields you provide. Your blueprints MUST start at slide_number 1.",
        f"- Keep content slides within slide numbers 1 through {max_slides - 1} "
        f"(total deck = title + content = at most {max_slides} slides).",
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
    """Return a list of validation failure messages; an empty list means the plan is valid.

    Checks all cross-field invariants — title length, group size bounds, unique and contiguous
    slide numbering, valid section label references — before section labels are resolved to
    concrete chunk IDs.  Failures are collected rather than raised so every problem is visible
    in a single retry prompt, reducing the number of LLM round-trips needed to converge.
    """
    failures: list[str] = []
    seen_slide_numbers: set[int] = set()
    slide_numbers_in_order: list[int] = []
    title_word_count = len(plan.title.split())

    if not plan.title.strip():
        failures.append("Presentation title must not be empty.")
    elif title_word_count >= 7:
        failures.append(
            f"Presentation title must be fewer than 7 words; received {title_word_count}: '{plan.title}'."
        )

    if not plan.subtitle.strip():
        failures.append("Presentation subtitle must not be empty.")

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

            if bp.slide_number < 1:
                failures.append(
                    f"Group {gi} blueprint {bi} has slide_number {bp.slide_number}, but content slides must start at 1."
                )
            if bp.slide_number > max_slides - 1:
                failures.append(
                    f"Group {gi} blueprint {bi} has slide_number {bp.slide_number}, which exceeds the "
                    f"maximum content slide index {max_slides - 1} for a deck of at most {max_slides} slides "
                    f"(including the title slide from metadata)."
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
        # Contiguous numbering prevents later fan-out steps from silently dropping or duplicating slides.
        expected_numbers = list(
            range(1, 1 + len(slide_numbers_in_order))
        )
        if slide_numbers_in_order != expected_numbers:
            failures.append(
                f"Content slide numbers must be contiguous starting at 1: expected {expected_numbers}, received {slide_numbers_in_order}."
            )

    return failures


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PlannerAgent(BaseLLMAgent):
    """LLM-backed agent that converts ingested document chunks into a validated PresentationPlan.

    Operates in three phases:
      1a. Data loading: reads all chunks per document, detects section boundaries via Markdown
          heading analysis, and builds paper-level summaries and a section label map.
      1b. Outline formatting: compresses section metadata into a compact planning brief with
          structural constraint instructions for the LLM.
      1c+2. LLM call + validation: asks the LLM for a structured LLMPresentationPlan, validates
            all structural invariants via _validate_llm_plan, and resolves section labels to
            concrete chunk IDs.  The full call is retried up to PLAN_MAX_RETRIES times on failure.
    """

    def __init__(
        self,
        tools_for_agent: dict | None = None,
    ):
        """Initialise with the planner system prompt and optional agent tools."""
        super().__init__(
            "planner",
            system_prompt=PLANNER_ROLE,
            tools_for_agent=tools_for_agent,
        )

    def run(self, state: ResearchState) -> Command[Literal["plan_executor"]]:
        """Load document chunks, build an outline, call the LLM, validate, resolve, and store the plan.

        Detailed flow:
          Phase 1a — For each doc_id, fetches raw chunks from research.db, detects section
                     boundaries via heading analysis, and builds a paper summary and section map.
          Phase 1b — Formats all sections into a compact outline string that names each section
                     by label (S0, S1, …) with heading, word count, and a representative snippet.
          Phase 1c — Calls the LLM with the user query + outline, requesting a LLMPresentationPlan.
                     The runtime_validator applies _validate_llm_plan before accepting the response;
                     validation failures cause the structured call to retry up to PLAN_MAX_RETRIES times.
          Phase 2  — Resolves each blueprint's source_sections labels to concrete chunk IDs using
                     the section_map built in Phase 1a, producing the final PresentationPlan.

        Raises ValueError when no doc_ids are present or no sections could be detected.
        Routes unconditionally to plan_executor on success.
        """
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
        paper_contexts: list[dict[str, str]] = []
        total_chunks = 0

        with ResearchDatabase() as db:
            for doc_idx, doc_id in enumerate(doc_ids):
                doc_row = db.connection.execute(
                    "SELECT filename, paper_metadata FROM documents WHERE id = ?",
                    (doc_id,),
                ).fetchone()
                raw_chunks = db.get_chunks_for_dispatch(doc_id)
                total_chunks += len(raw_chunks)
                paper_metadata: dict = {}
                if doc_row and doc_row["paper_metadata"]:
                    try:
                        paper_metadata = json.loads(doc_row["paper_metadata"])
                    except json.JSONDecodeError:
                        paper_metadata = {}

                title = (
                    paper_titles[doc_idx].strip()
                    if doc_idx < len(paper_titles) and paper_titles[doc_idx].strip()
                    else (paper_metadata.get("title") or (doc_row["filename"] if doc_row else doc_id))
                )
                paper_contexts.append({
                    "doc_id": doc_id,
                    "title": title,
                    "summary": _build_paper_summary(raw_chunks, paper_metadata.get("abstract", "")),
                })
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
            all_sections, paper_contexts,
            sections_per_doc, max_slides, total_chunks,
        )

        contract = build_planner_output_format(
            min_group_size=MIN_GROUP_SIZE, max_group_size=MAX_GROUP_SIZE
        )
        user_prompt = (
            f"USER QUERY:\n{query}\n\n"
            f"{outline}\n\n"
            "Produce a PresentationPlan for the above paper(s) based on the user query.\n\n"
            f"{contract}"
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

        for group in llm_plan.slide_groups:
            resolved_blueprints: list[SlideBlueprint] = []
            for bp in group.slide_blueprints:
                chunk_ids: list[str] = []
                for label in bp.source_sections:
                    chunk_ids.extend(section_map.get(label, []))
                resolved_blueprints.append(SlideBlueprint(
                    slide_number=bp.slide_number,
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
            title=llm_plan.title,
            subtitle=llm_plan.subtitle,
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
            f"Reasoning: {presentation_plan.reasoning}"
        )
        self._logger.log(msg)

        return Command(
            update={
                "presentation_plan": presentation_plan,
                "messages": [msg],
            }
        )


def planner_node(
    state: ResearchState,
    *,
    tools_for_agent: dict | None = None,
) -> Command:
    """LangGraph node entry point that constructs a PlannerAgent and delegates to its run() method."""
    return PlannerAgent(
        tools_for_agent=tools_for_agent,
    ).run(state)
