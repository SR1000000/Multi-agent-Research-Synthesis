"""
Slide writer agents.

This module now separates initial generation from revision behavior while
sharing the same prompt-building, LLM, and persistence machinery. The active
graph still calls `slide_writer_node`, which dispatches to the correct thin
agent wrapper based on whether rewrite instructions are present.
"""
from __future__ import annotations

import json
from typing import List, TypedDict

from langgraph.types import Command

from src.agents.base import BaseLLMAgent, SLIDE_REWRITER_ROLE
from src.memory.research.schema import ProtoSlide, SlideContent, make_slide_batch_model, slide_output_prompt_contract
from src.memory.research.database import ResearchDatabase
from src.state import group_allows_empty_chunks


# ---------------------------------------------------------------------------
# Dispatch state (sent via LangGraph Send API)
# ---------------------------------------------------------------------------

class SlideWriterDispatch(TypedDict):
    """State payload delivered to each slide_writer node via Send()."""
    chunk_ids:             List[str]   # group-wide union passed to the writer as shared working context
    slide_blueprints:      List[dict]  # serialized SlideBlueprint dicts
    group_idx:             int         # index into PresentationPlan.slide_groups
    session_id:            str
    rewrite_instructions:  str         # empty for initial gen; Critic populates for rewrites


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------

def _group_log_label(state: SlideWriterDispatch) -> str:
    blueprints = state.get("slide_blueprints", [])
    if not blueprints:
        return "SlideWriter[empty]"
    nums = [bp.get("slide_number", "?") for bp in blueprints]
    return f"SlideWriter[slides {nums[0]}-{nums[-1]}, group {state.get('group_idx', '?')}]"


def _ordered_chunk_texts(rows: list, chunk_ids: list[str]) -> list[str]:
    """Return chunk text blocks in the caller-provided chunk order."""
    rows_by_id = {row["id"]: row for row in rows}
    ordered: list[str] = []
    for chunk_id in chunk_ids:
        row = rows_by_id.get(chunk_id)
        if row is None:
            continue
        text = row["contextualized_text"] if row["contextualized_text"] else row["text"]
        ordered.append(f"--- Chunk ID: {row['id']} ---\n{text}")
    return ordered


def _format_existing_slide(slide: ProtoSlide) -> str:
    """Render an existing slide draft into a compact JSON block for revision mode."""
    return json.dumps(
        {
            "slide_number": slide.slide_number,
            "content": slide.content.model_dump(mode="json"),
            "chunk_references": slide.chunk_references,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _BaseSlideWorkerAgent(BaseLLMAgent):
    def __init__(self, *, log_display: str | None = None):
        super().__init__("slide_writer", log_display=log_display)

    def _failure(self, error: Exception, *, group_idx: int, tag: str) -> Command:
        err_str = f"{type(error).__name__}: {error}"
        self._logger.log(f"[{tag}] ERROR: {err_str}", level="error")
        err_msg = f"[{tag}] FAILED: {err_str}"
        return Command(update={
            "slides_written": [{"group_idx": group_idx, "count": 0}],
            "messages":       [err_msg],
            "errors":         [{"node": tag, "error": err_str}],
        })

    def _load_context(
        self,
        *,
        chunk_ids: list[str],
        slide_blueprints: list[dict],
        include_existing_slides: bool,
    ) -> tuple[str, list[ProtoSlide]]:
        with ResearchDatabase() as research_db:
            rows = []
            if chunk_ids:
                placeholders = ",".join(["?"] * len(chunk_ids))
                rows = research_db.connection.execute(
                    f"SELECT id, text, contextualized_text FROM text_chunks "
                    f"WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
            existing_slides: list[ProtoSlide] = []
            if include_existing_slides:
                for bp_dict in slide_blueprints:
                    slide_num = bp_dict.get("slide_number")
                    if slide_num is None:
                        continue
                    existing_slide = research_db.load_slide(slide_num)
                    if existing_slide is not None:
                        existing_slides.append(existing_slide)

        chunk_texts = _ordered_chunk_texts(rows, chunk_ids)
        return "\n\n".join(chunk_texts), existing_slides

    def _build_assignment_block(self, slide_blueprints: list[dict]) -> str:
        return "\n".join(
            f"Slide {bp.get('slide_number', i + 1)}: "
            f"[{bp.get('narrative_role', 'evidence')}] "
            f'"{bp.get("working_title", "")}" — {bp.get("intent", "")}'
            for i, bp in enumerate(slide_blueprints)
        )

    def _base_prompt_parts(self, *, slide_count: int, combined_text: str) -> list[str]:
        return [
            "",
            slide_output_prompt_contract(slide_count),
            "",
            "### SEMANTIC GUIDANCE:",
            "- Keep each slide focused on one primary takeaway.",
            "- Choose the layout that best supports the evidence and intended narrative role.",
            "- Speaker notes should be professional, conversational, and cover the core bullets with added context.",
            "- Avoid academic jargon unless the original terminology is important to preserve.",
            "",
            "SOURCE MATERIAL (research chunks):",
            combined_text,
        ]

    def _build_user_prompt(
        self,
        *,
        slide_count: int,
        slide_blueprints: list[dict],
        combined_text: str,
        rewrite_instructions: str,
        existing_slides: list[ProtoSlide],
    ) -> str:
        raise NotImplementedError

    def _system_prompt_override(self) -> str | None:
        return None

    def run(self, state: SlideWriterDispatch) -> Command:
        self._set_session_id(state)
        chunk_ids            = state.get("chunk_ids", [])
        slide_blueprints     = state.get("slide_blueprints", [])
        group_idx            = state.get("group_idx", 0)
        rewrite_instructions = state.get("rewrite_instructions", "")
        tag                  = self._log_display

        try:
            if not chunk_ids and not group_allows_empty_chunks(slide_blueprints):
                self._logger.log(f"[{tag}] No chunk_ids — skipping", level="warning")
                return Command(update={
                    "slides_written": [{"group_idx": group_idx, "count": 0}],
                    "messages":       [f"[{tag}] Skipped (no chunks)"],
                })

            if not slide_blueprints:
                self._logger.log(f"[{tag}] No blueprints — skipping", level="warning")
                return Command(update={
                    "slides_written": [{"group_idx": group_idx, "count": 0}],
                    "messages":       [f"[{tag}] Skipped (no blueprints)"],
                })

            prebuilt_slides = [
                ProtoSlide(
                    slide_number=bp_dict["slide_number"],
                    content=SlideContent.model_validate(bp_dict["prebuilt_content"]),
                    chunk_references=bp_dict.get("source_chunk_ids", []),
                )
                for bp_dict in slide_blueprints
                if bp_dict.get("prebuilt_content") is not None
            ]
            if len(prebuilt_slides) == len(slide_blueprints):
                saved_count = 0
                with ResearchDatabase() as research_db:
                    for proto in prebuilt_slides:
                        research_db.save_slide(proto)
                        saved_count += 1
                msg = f"[{tag}] Wrote {saved_count}/{len(slide_blueprints)} planner-authored slide(s) without Slide Writer generation"
                self._logger.log(msg)
                return Command(update={
                    "slides_written": [{"group_idx": group_idx, "count": saved_count}],
                    "messages":       [msg],
                })

            slide_count = len(slide_blueprints)
            combined_text, existing_slides = self._load_context(
                chunk_ids=chunk_ids,
                slide_blueprints=slide_blueprints,
                include_existing_slides=bool(rewrite_instructions.strip()),
            )
            user_prompt = self._build_user_prompt(
                slide_count=slide_count,
                slide_blueprints=slide_blueprints,
                combined_text=combined_text,
                rewrite_instructions=rewrite_instructions,
                existing_slides=existing_slides,
            )
            turns = [{"role": "user", "content": user_prompt}]

            # ------------------------------------------------------------------
            # 3. Call LLM
            # ------------------------------------------------------------------
            output_schema = make_slide_batch_model(slide_count)
            result = self._call_structured(
                turns,
                output_schema,
                model="slides",
                system_prompt_override=self._system_prompt_override(),
                runtime_validator=lambda parsed: [] if len(parsed.slides) == slide_count else [
                    f"Expected exactly {slide_count} slides but received {len(parsed.slides)}."
                ],
            )
            slides = result.parsed.slides

            if len(slides) != slide_count:
                raise ValueError(
                    f"Expected exactly {slide_count} slides but received {len(slides)}."
                )

            # ------------------------------------------------------------------
            # 4. Save proto-slides to research.db
            # ------------------------------------------------------------------
            saved_count = 0
            with ResearchDatabase() as research_db:
                for idx, bp_dict in enumerate(slide_blueprints):
                    slide_content = slides[idx]
                    slide_num = bp_dict.get("slide_number", saved_count + 1)
                    slide_chunk_ids = bp_dict.get("source_chunk_ids", [])
                    proto = ProtoSlide(
                        slide_number=slide_num,
                        content=slide_content,
                        chunk_references=slide_chunk_ids,
                    )
                    research_db.save_slide(proto)
                    saved_count += 1

            if saved_count == 0:
                self._logger.log(f"[{tag}] LLM returned 0 slides", level="warning")

            msg = f"[{tag}] Wrote {saved_count}/{slide_count} slide(s)"
            self._logger.log(msg)

            return Command(update={
                "slides_written": [{"group_idx": group_idx, "count": saved_count}],
                "messages":       [msg],
            })

        except Exception as e:
            return self._failure(e, group_idx=group_idx, tag=tag)


# ---------------------------------------------------------------------------
# Thin agent wrappers
# ---------------------------------------------------------------------------

class InitialSlideWriterAgent(_BaseSlideWorkerAgent):
    def _build_user_prompt(
        self,
        *,
        slide_count: int,
        slide_blueprints: list[dict],
        combined_text: str,
        rewrite_instructions: str,
        existing_slides: list[ProtoSlide],
    ) -> str:
        del rewrite_instructions, existing_slides
        blueprint_block = self._build_assignment_block(slide_blueprints)
        user_prompt_parts = [
            f"Return exactly ONE JSON object whose `slides` array contains exactly {slide_count} slide(s).\n",
            "",
            "SLIDE ASSIGNMENTS:",
            blueprint_block,
            "",
            "For each slide, follow the intent directive precisely. "
            "Do not add extra slides or skip any.",
        ]
        user_prompt_parts += self._base_prompt_parts(
            slide_count=slide_count,
            combined_text=combined_text,
        )
        return "\n".join(user_prompt_parts)


class SlideRewriterAgent(_BaseSlideWorkerAgent):
    def _system_prompt_override(self) -> str | None:
        return SLIDE_REWRITER_ROLE

    def _build_user_prompt(
        self,
        *,
        slide_count: int,
        slide_blueprints: list[dict],
        combined_text: str,
        rewrite_instructions: str,
        existing_slides: list[ProtoSlide],
    ) -> str:
        blueprint_block = self._build_assignment_block(slide_blueprints)
        existing_slides_block = "\n\n".join(
            _format_existing_slide(slide) for slide in existing_slides
        )
        user_prompt_parts = [
            f"Return exactly ONE JSON object whose `slides` array contains exactly {slide_count} slide(s).\n",
            "",
            "REVISION MODE:",
            "- Rewrite the assigned slides to satisfy the reviewer feedback.",
            "- Reviewer feedback overrides the prior blueprint intent when they conflict.",
            "- Preserve any parts of the current slides that already work.",
            "- Keep the slide count fixed and maintain slide order.",
            "",
            "INSTRUCTION PRIORITY:",
            "1. Reviewer rewrite instructions",
            "2. Factual grounding in the provided research chunks",
            "3. Existing slide drafts as the material to revise",
            "4. Slide assignments and narrative roles as supporting context",
            "",
            "REWRITE INSTRUCTIONS (from reviewer):",
            rewrite_instructions,
            "",
            "CURRENT SLIDE DRAFTS TO REVISE:",
            existing_slides_block or "No existing drafts were found. Regenerate the assigned slides from scratch while following the reviewer feedback.",
            "",
            "SLIDE ASSIGNMENTS (context only unless reviewer feedback conflicts):",
            blueprint_block,
        ]
        user_prompt_parts += self._base_prompt_parts(
            slide_count=slide_count,
            combined_text=combined_text,
        )
        return "\n".join(user_prompt_parts)


class SlideWriterAgent(InitialSlideWriterAgent):
    """Backward-compatible alias for the initial slide writer."""


def slide_writer_node(state: SlideWriterDispatch) -> Command:
    rewrite_instructions = state.get("rewrite_instructions", "")
    agent_cls = SlideRewriterAgent if rewrite_instructions.strip() else InitialSlideWriterAgent
    return agent_cls(log_display=_group_log_label(state)).run(state)
