"""
ParseSupervisorAgent
====================
Reads all chunks for a document from research.db, detects section boundaries
(LlamaParse path: regex scan for leading Markdown headings), builds a compact
section outline, and calls the LLM to decide how to group those sections into
parallel research_to_slide agent assignments.

The agent then fans the assignments out via LangGraph's Send API.
"""
from __future__ import annotations

import re
from typing import List, Literal

from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from src.state import ResearchState
from src.agents.base import BaseLLMAgent
from src.memory.research.database import ResearchDatabase
from src.logging.logger import AgentLogger

# ---------------------------------------------------------------------------
# Heading detection (LlamaParse produces standard ATX markdown headings)
# ---------------------------------------------------------------------------
# Matches lines that start a new ATX heading: # / ## / ### / ####
# We only look at the first non-empty line of each chunk so that mid-chunk
# headings (e.g. sub-bullets with hashes) don't falsely trigger a split.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def _detect_heading(chunk_text: str) -> str | None:
    """
    Return the heading text if the chunk's first meaningful line is a Markdown
    ATX heading, otherwise return None.
    """
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING_RE.match(line):
            # Strip leading hashes and whitespace to get the bare heading text
            return re.sub(r"^#{1,4}\s+", "", stripped).strip()
        # First non-empty line is not a heading → not a section boundary
        return None
    return None


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM output
# ---------------------------------------------------------------------------

class AgentAssignment(BaseModel):
    """One parallel agent's assignment."""
    section_indices: List[int] = Field(
        description="0-based indices of consecutive sections assigned to this agent"
    )
    slide_count: int = Field(
        description="Exact number of slides this agent may produce (>= 1)"
    )
    rationale: str = Field(
        description="One sentence explaining why these sections were grouped together"
    )


class PartitionPlan(BaseModel):
    """Full partition plan returned by the LLM."""
    assignments: List[AgentAssignment] = Field(
        description="Ordered list of agent assignments covering all sections"
    )
    overall_reasoning: str = Field(
        description="2-4 sentences summarising the partitioning strategy"
    )


# ---------------------------------------------------------------------------
# Section data (local, not sent to LLM directly)
# ---------------------------------------------------------------------------

class _Section:
    """Internal representation of one detected section."""
    __slots__ = ("index", "heading", "chunk_ids", "word_count")

    def __init__(self, index: int, heading: str, chunk_ids: list[str], word_count: int):
        self.index = index
        self.heading = heading
        self.chunk_ids = chunk_ids
        self.word_count = word_count


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ParseSupervisorAgent(BaseLLMAgent):
    """
    Analyses the section structure of a research paper and fans out chunk
    ranges to parallel research_to_slide agents via LangGraph's Send API.
    """

    def __init__(self) -> None:
        super().__init__("parse_supervisor")
        self._logger = AgentLogger()

    # ------------------------------------------------------------------
    # Public entry point — called by the graph node
    # ------------------------------------------------------------------

    def run(self, state: ResearchState) -> Command[Literal["research_to_slide"]]:
        doc_id     = state["doc_id"]
        max_slides = state.get("max_slides", 12)
        session_id = state.get("session_id", "")

        self._logger.log(
            f"[ParseSupervisor] Starting — doc_id={doc_id} max_slides={max_slides}"
        )

        # 1. Load ordered chunks cheaply (no images/tables/embeddings)
        with ResearchDatabase() as db:
            raw_chunks = db.get_chunks_for_dispatch(doc_id)

        if not raw_chunks:
            self._logger.log(
                "[ParseSupervisor] No chunks found in research.db — nothing to dispatch"
            )
            return Command(goto=[])

        # 2. Detect section boundaries and group chunks into sections
        sections = self._detect_sections(raw_chunks)

        self._logger.log(
            f"[ParseSupervisor] Detected {len(sections)} sections "
            f"from {len(raw_chunks)} total chunks"
        )

        # 3. Build compact outline for the LLM
        outline_text = self._build_outline(sections, max_slides)

        # 4. Ask the LLM for a partition plan
        plan = self._call_llm(outline_text, len(sections), max_slides)

        self._logger.log(
            f"[ParseSupervisor] LLM plan: {len(plan.assignments)} agents — "
            f"{plan.overall_reasoning}"
        )

        # 5. Validate & repair the plan (ensure slide counts sum to max_slides)
        plan = self._repair_plan(plan, sections, max_slides)

        # 6. Convert assignments → Send objects and fan out.
        # Command(goto=[Send(...)]) is the idiomatic LangGraph pattern for
        # dynamic parallel dispatch — no conditional edge or state key needed.
        sends = self._build_sends(plan, sections, session_id)

        self._logger.log(
            f"[ParseSupervisor] Dispatching {len(sends)} research_to_slide agents"
        )
        return Command(goto=sends)

    # ------------------------------------------------------------------
    # Step 2 — Section detection
    # ------------------------------------------------------------------

    def _detect_sections(self, raw_chunks: list[dict]) -> list[_Section]:
        """
        Group consecutive chunks into sections.  A new section begins when
        the first non-empty line of a chunk is an ATX Markdown heading.
        The very first chunk always starts a section regardless.
        """
        sections: list[_Section] = []
        current_ids: list[str] = []
        current_heading = "(no heading)"
        current_words = 0

        for i, chunk in enumerate(raw_chunks):
            text = chunk["text"] or ""
            heading = _detect_heading(text)

            is_boundary = (i == 0) or (heading is not None)

            if is_boundary and current_ids:
                sections.append(_Section(
                    index=len(sections),
                    heading=current_heading,
                    chunk_ids=current_ids,
                    word_count=current_words,
                ))
                current_ids = []
                current_words = 0

            if is_boundary and heading:
                current_heading = heading
            elif is_boundary and i == 0:
                current_heading = heading or "(no heading)"

            current_ids.append(chunk["id"])
            current_words += len(text.split())

        # Flush the last section
        if current_ids:
            sections.append(_Section(
                index=len(sections),
                heading=current_heading,
                chunk_ids=current_ids,
                word_count=current_words,
            ))

        return sections

    # ------------------------------------------------------------------
    # Step 3 — Outline formatting
    # ------------------------------------------------------------------

    def _build_outline(self, sections: list[_Section], max_slides: int) -> str:
        total_chunks = sum(len(s.chunk_ids) for s in sections)
        total_words  = sum(s.word_count for s in sections)

        lines = [
            f"Paper outline ({len(sections)} sections, "
            f"{total_chunks} total chunks, ~{total_words} words):",
            f"Target slide budget: {max_slides} slides total",
            "",
            "Index | Heading                              | Chunks | ~Words",
            "------|--------------------------------------|--------|-------",
        ]
        for s in sections:
            heading_col = s.heading[:36].ljust(36)
            lines.append(
                f"  {s.index:3d} | {heading_col} |   {len(s.chunk_ids):3d} | {s.word_count:6d}"
            )

        lines += [
            "",
            "Assign ALL sections to agents. Section indices must be consecutive "
            "within each assignment and cover every section exactly once.",
            f"The sum of all slide_count values must equal exactly {max_slides}.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Step 4 — LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self, outline_text: str, num_sections: int, max_slides: int
    ) -> PartitionPlan:
        user_msg = (
            f"{outline_text}\n\n"
            f"Produce a PartitionPlan that groups these {num_sections} sections "
            f"into parallel agent assignments with a total slide budget of {max_slides}."
        )
        turns = [{"role": "user", "content": user_msg}]
        return self._call(turns, schema=PartitionPlan, model="slides")

    # ------------------------------------------------------------------
    # Step 5 — Plan repair
    # ------------------------------------------------------------------

    def _repair_plan(
        self, plan: PartitionPlan, sections: list[_Section], max_slides: int
    ) -> PartitionPlan:
        """
        Guard against LLM mistakes:
        1. Ensure every section index appears in exactly one assignment.
        2. Ensure slide_count >= 1 for every assignment.
        3. Normalise slide_counts so they sum to exactly max_slides.
        """
        all_indices = set(range(len(sections)))
        covered: set[int] = set()
        repaired_assignments: list[AgentAssignment] = []

        for asgn in plan.assignments:
            # Remove duplicates, keep valid indices
            valid = sorted(set(asgn.section_indices) - covered)
            valid = [i for i in valid if i in all_indices]
            if not valid:
                continue
            covered.update(valid)
            repaired_assignments.append(AgentAssignment(
                section_indices=valid,
                slide_count=max(1, asgn.slide_count),
                rationale=asgn.rationale,
            ))

        # Any sections the LLM forgot → append as a catch-all bucket
        missed = sorted(all_indices - covered)
        if missed:
            self._logger.log(
                f"[ParseSupervisor] Repair: re-adding {len(missed)} missed sections"
            )
            repaired_assignments.append(AgentAssignment(
                section_indices=missed,
                slide_count=1,
                rationale="Catch-all for sections not assigned by the LLM.",
            ))

        if not repaired_assignments:
            # Fallback: one agent gets everything
            repaired_assignments = [AgentAssignment(
                section_indices=list(all_indices),
                slide_count=max_slides,
                rationale="Single-agent fallback.",
            )]

        # Normalise slide counts to sum exactly to max_slides
        total = sum(a.slide_count for a in repaired_assignments)
        if total != max_slides:
            # Scale proportionally then fix rounding error on the largest bucket
            scaled = [
                max(1, round(a.slide_count * max_slides / total))
                for a in repaired_assignments
            ]
            diff = max_slides - sum(scaled)
            if diff != 0:
                # Add/subtract the rounding error from the assignment with the
                # most slides (least impact on proportion)
                largest_idx = scaled.index(max(scaled))
                scaled[largest_idx] = max(1, scaled[largest_idx] + diff)
            # Reconstruct as new objects — Pydantic v2 models are immutable
            repaired_assignments = [
                AgentAssignment(
                    section_indices=a.section_indices,
                    slide_count=s,
                    rationale=a.rationale,
                )
                for a, s in zip(repaired_assignments, scaled)
            ]

        return PartitionPlan(
            assignments=repaired_assignments,
            overall_reasoning=plan.overall_reasoning,
        )

    # ------------------------------------------------------------------
    # Step 6 — Build Send objects
    # ------------------------------------------------------------------

    def _build_sends(
        self,
        plan: PartitionPlan,
        sections: list[_Section],
        session_id: str,
    ) -> list[Send]:
        sends: list[Send] = []
        slide_cursor = 1

        for asgn in plan.assignments:
            # Collect chunk IDs in document order
            chunk_ids: list[str] = []
            for idx in sorted(asgn.section_indices):
                chunk_ids.extend(sections[idx].chunk_ids)

            if not chunk_ids:
                continue

            start_slide = slide_cursor
            end_slide   = slide_cursor + asgn.slide_count - 1
            slide_cursor = end_slide + 1

            sends.append(Send(
                "research_to_slide",
                {
                    "chunk_ids":          chunk_ids,
                    "slide_number_range": [start_slide, end_slide],
                    "session_id":         session_id,
                },
            ))

        return sends


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def parse_supervisor_node(state: ResearchState) -> Command[Literal["research_to_slide"]]:
    return ParseSupervisorAgent().run(state)
