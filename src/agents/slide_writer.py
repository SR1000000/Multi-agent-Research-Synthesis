"""
SlideWriterAgent
================
Receives a batch of slide blueprints (with chunk IDs and per-slide intents
from the PresentationPlan) and writes the initial set of proto-slides.

The same agent is used for Critic-driven rewrites in the future: callers
populate `rewrite_instructions` instead of leaving it empty.

Error contract: NEVER raise an unhandled exception. Catch everything, log it
loudly, and return count=0 so the PlanExecutor can retry the group.
"""
from __future__ import annotations

from typing import List, TypedDict

from langgraph.types import Command

from src.agents.base import BaseLLMAgent
from src.memory.research.schema import ProtoSlide, make_slide_batch_model, slide_output_prompt_contract
from src.memory.research.database import ResearchDatabase


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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SlideWriterAgent(BaseLLMAgent):
    def __init__(self, *, log_display: str | None = None):
        super().__init__("slide_writer", log_display=log_display)

    def run(self, state: SlideWriterDispatch) -> Command:
        self._set_session_id(state)

        chunk_ids            = state.get("chunk_ids", [])
        slide_blueprints     = state.get("slide_blueprints", [])
        group_idx            = state.get("group_idx", 0)
        rewrite_instructions = state.get("rewrite_instructions", "")
        tag                  = self._log_display

        # Result payload — always returned, even on error
        def _failure(error: Exception) -> Command:
            err_str = f"{type(error).__name__}: {error}"
            self._logger.log(f"[{tag}] ERROR: {err_str}", level="error")
            err_msg = f"[{tag}] FAILED: {err_str}"
            return Command(update={
                "slides_written": [{"group_idx": group_idx, "count": 0}],
                "messages":       [err_msg],
                "errors":         [{"node": tag, "error": err_str}],
            })

        try:
            if not chunk_ids:
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

            # ------------------------------------------------------------------
            # 1. Fetch chunk text from research.db
            # ------------------------------------------------------------------
            with ResearchDatabase() as research_db:
                placeholders = ",".join(["?"] * len(chunk_ids))
                rows = research_db.connection.execute(
                    f"SELECT id, text, contextualized_text FROM text_chunks "
                    f"WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()

            chunk_texts = _ordered_chunk_texts(rows, chunk_ids)
            combined_text = "\n\n".join(chunk_texts)

            # ------------------------------------------------------------------
            # 2. Build the user prompt from blueprints
            # ------------------------------------------------------------------
            slide_count = len(slide_blueprints)
            blueprint_block = "\n".join(
                f"Slide {bp.get('slide_number', i + 1)}: "
                f"[{bp.get('narrative_role', 'evidence')}] "
                f'"{bp.get("working_title", "")}" — {bp.get("intent", "")}'
                for i, bp in enumerate(slide_blueprints)
            )

            user_prompt_parts = [
                f"Return exactly ONE JSON object whose `slides` array contains exactly {slide_count} slide(s).\n",
                "",
                "SLIDE ASSIGNMENTS:",
                blueprint_block,
                "",
                "For each slide, follow the intent directive precisely. "
                "Do not add extra slides or skip any.",
            ]

            if rewrite_instructions:
                user_prompt_parts += [
                    "",
                    "REWRITE INSTRUCTIONS (from reviewer):",
                    rewrite_instructions,
                ]

            user_prompt_parts += [
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

            user_prompt = "\n".join(user_prompt_parts)
            turns = [{"role": "user", "content": user_prompt}]

            # ------------------------------------------------------------------
            # 3. Call LLM
            # ------------------------------------------------------------------
            output_schema = make_slide_batch_model(slide_count)
            result = self._call_structured(
                turns,
                output_schema,
                model="slides",
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
            return _failure(e)


def slide_writer_node(state: SlideWriterDispatch) -> Command:
    return SlideWriterAgent(log_display=_group_log_label(state)).run(state)
