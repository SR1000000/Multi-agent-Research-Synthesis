"""
Slide writer agents.

This module now separates initial generation from revision behavior while
sharing the same prompt-building, LLM, and persistence machinery. The active
graph still calls `slide_writer_node`, which dispatches to the correct thin
agent wrapper based on whether rewrite instructions are present.
"""
from __future__ import annotations

import json
from typing import Any, List, TypedDict

from langgraph.types import Command

from src.agents._image_utils import format_image_assets_block
from src.agents.base import BaseLLMAgent, SLIDE_REWRITER_ROLE
from src.memory.research.schema import ImageMetadata, ProtoSlide, make_slide_batch_model, slide_output_prompt_contract
from src.memory.research.database import ResearchDatabase


# ---------------------------------------------------------------------------
# Dispatch state (sent via LangGraph Send API)
# ---------------------------------------------------------------------------

class SlideWriterDispatch(TypedDict):
    """State payload delivered to each slide_writer node via Send()."""
    dispatch_id:            str
    assignment_id:          str
    chunk_ids:             List[str]   # group-wide union passed to the writer as shared working context
    slide_blueprints:      List[dict]  # serialized SlideBlueprint dicts
    group_idx:             int         # index into PresentationPlan.slide_groups
    session_id:            str
    rewrite_instructions:  str         # empty for initial gen; Critic populates for rewrites
    target_slide_numbers:  List[int]


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
    def __init__(
        self,
        *,
        log_display: str | None = None,
        tools_for_agent: dict[str, dict[str, Any]] | None = None,
    ):
        super().__init__("slide_writer", log_display=log_display, tools_for_agent=tools_for_agent)

    def _failure(
        self,
        error: Exception,
        *,
        dispatch_id: str,
        assignment_id: str,
        group_idx: int,
        target_slide_numbers: list[int],
        tag: str,
    ) -> Command:
        err_str = f"{type(error).__name__}: {error}"
        self._logger.log(f"[{tag}] ERROR: {err_str}", level="error")
        err_msg = f"[{tag}] FAILED: {err_str}"
        return Command(update={
            "slides_written": [{
                "dispatch_id": dispatch_id,
                "assignment_id": assignment_id,
                "group_idx": group_idx,
                "count": 0,
                "target_slide_numbers": target_slide_numbers,
            }],
            "messages":       [err_msg],
            "errors":         [{"node": tag, "error": err_str}],
        })

    def _load_context(
        self,
        *,
        chunk_ids: list[str],
        slide_blueprints: list[dict],
        include_existing_slides: bool,
    ) -> tuple[str, list[ProtoSlide], list[ImageMetadata]]:
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
            image_metadatas = research_db.get_images_for_chunks(chunk_ids) if chunk_ids else []

        chunk_texts = _ordered_chunk_texts(rows, chunk_ids)
        return "\n\n".join(chunk_texts), existing_slides, image_metadatas

    def _build_assignment_block(self, slide_blueprints: list[dict]) -> str:
        return "\n".join(
            f"Slide {bp.get('slide_number', i + 1)}: "
            f"[{bp.get('narrative_role', 'evidence')}] "
            f'"{bp.get("working_title", "")}" — {bp.get("intent", "")}'
            for i, bp in enumerate(slide_blueprints)
        )

    def _base_prompt_parts(
        self,
        *,
        slide_count: int,
        combined_text: str,
        image_metadatas: list[ImageMetadata],
    ) -> list[str]:
        parts: list[str] = [
            "",
            slide_output_prompt_contract(slide_count),
            "",
            "### SEMANTIC GUIDANCE:",
            "- Keep each slide focused on one primary takeaway.",
            "- Do not default to a basic slide with exactly three flat bullets unless that structure is genuinely the clearest fit for the material.",
            "- Vary slide density based on the content: some slides should use 2 strong bullets, others 4-5 concise bullets, and others a few top-level bullets with supporting sub-bullets.",
            "- Use sub-bullets when they help unpack evidence, examples, caveats, or stepwise logic under a main claim.",
            "- Speaker notes should be professional, conversational, and cover the core bullets with added context.",
            "- Avoid academic jargon unless the original terminology is important to preserve.",
            "- Only populate `media_id` when an image genuinely reinforces the slide's narrative intent. Do not force image inclusion; omit `media_id` (leave it null) if no available asset meaningfully supports the slide.",
            "- Choose the layout that best supports the evidence and intended narrative role.",
        ]
        image_block = format_image_assets_block(image_metadatas)
        if image_block:
            parts += ["", image_block]
        parts += ["", "SOURCE MATERIAL (research chunks):", combined_text]
        return parts

    def _build_retrieval_turns(
        self,
        *,
        slide_blueprints: list[dict],
        rewrite_instructions: str,
        existing_slides: list[ProtoSlide],
    ) -> list[dict]:
        # Tool prompt snippets describe tool policy at the agent/system level.
        # This separate user turn exists to make the first LLM call an explicit
        # evidence-gathering step for the current slide assignment before the
        # second structured slide-generation call.
        assignment_block = self._build_assignment_block(slide_blueprints)
        existing_slides_block = "\n\n".join(
            _format_existing_slide(slide) for slide in existing_slides
        )
        prompt_parts = [
            "Use the available retrieval tool to gather evidence for the assigned slides.",
            "You must call `retrieve_artifacts` at least once before answering.",
            "After finishing tool use, return a short evidence brief summarizing only the retrieved evidence that is most relevant to these slides.",
            "",
            "SLIDE ASSIGNMENTS:",
            assignment_block,
        ]
        if rewrite_instructions.strip():
            prompt_parts.extend(
                [
                    "",
                    "REVISION CONTEXT:",
                    "Reviewer rewrite instructions:",
                    rewrite_instructions,
                    "",
                    "Current slide drafts:",
                    existing_slides_block or "No existing drafts were found.",
                ]
            )
        return [{"role": "user", "content": "\n".join(prompt_parts)}]

    def _tool_payload_text(self, tool_results: list[dict[str, Any]]) -> str:
        payloads: list[str] = []
        for result in tool_results:
            payload = result.get("payload")
            if not isinstance(payload, dict):
                continue
            payloads.append(json.dumps(payload, ensure_ascii=True))
        return "\n\n".join(payloads)

    def _build_user_prompt(
        self,
        *,
        slide_count: int,
        slide_blueprints: list[dict],
        combined_text: str,
        rewrite_instructions: str,
        existing_slides: list[ProtoSlide],
        image_metadatas: list[ImageMetadata],
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
        target_slide_numbers = state.get("target_slide_numbers", [])
        dispatch_id          = state.get("dispatch_id", "")
        assignment_id        = state.get("assignment_id", f"group-{group_idx}")
        session_id           = state.get("session_id", "")
        tag                  = self._log_display

        try:
            if not slide_blueprints:
                self._logger.log(f"[{tag}] No blueprints — skipping", level="warning")
                return Command(update={
                    "slides_written": [{
                        "dispatch_id": dispatch_id,
                        "assignment_id": assignment_id,
                        "group_idx": group_idx,
                        "count": 0,
                        "target_slide_numbers": target_slide_numbers,
                    }],
                    "messages":       [f"[{tag}] Skipped (no blueprints)"],
                })

            if target_slide_numbers:
                target_set = set(target_slide_numbers)
                slide_blueprints = [
                    bp_dict for bp_dict in slide_blueprints
                    if bp_dict.get("slide_number") in target_set
                ]
                if not slide_blueprints:
                    self._logger.log(f"[{tag}] No matching target slides — skipping", level="warning")
                    return Command(update={
                        "slides_written": [{
                            "dispatch_id": dispatch_id,
                            "assignment_id": assignment_id,
                            "group_idx": group_idx,
                            "count": 0,
                            "target_slide_numbers": target_slide_numbers,
                        }],
                        "messages": [f"[{tag}] Skipped (no target slides)"],
                    })

            slide_count = len(slide_blueprints)
            combined_text, existing_slides, image_metadatas = self._load_context(
                chunk_ids=chunk_ids,
                slide_blueprints=slide_blueprints,
                include_existing_slides=bool(rewrite_instructions.strip()),
            )
            retrieval_turns = self._build_retrieval_turns(
                slide_blueprints=slide_blueprints,
                rewrite_instructions=rewrite_instructions,
                existing_slides=existing_slides,
            )
            tool_call_out = self._call(
                retrieval_turns,
                use_tools=True,
                session_id=session_id,
                max_tool_calls=4,
                model="slides",
                system_prompt_override=self._system_prompt_override(),
            )
            # The tool loop above already gave the model the tool outputs once.
            # We pass the full returned payloads into the second structured call
            # because tool-use mode and schema mode are still separate paths.
            combined_text = "\n\n".join(
                part
                for part in [self._tool_payload_text(tool_call_out["tool_results"]), tool_call_out["content"]]
                if isinstance(part, str) and part.strip()
            )
            if not combined_text:
                self._logger.log(f"[{tag}] No retrieved artifacts available for writing", level="warning")
                return Command(update={
                    "slides_written": [{
                        "dispatch_id": dispatch_id,
                        "assignment_id": assignment_id,
                        "group_idx": group_idx,
                        "count": 0,
                        "target_slide_numbers": target_slide_numbers,
                    }],
                    "messages": [f"[{tag}] Skipped (no retrieved evidence)"],
                    "tool_calls": tool_call_out["tool_calls"],
                    "tool_results": tool_call_out["tool_results"],
                    "retrieval_queries": tool_call_out["retrieval_queries"],
                })
            user_prompt = self._build_user_prompt(
                slide_count=slide_count,
                slide_blueprints=slide_blueprints,
                combined_text=combined_text,
                rewrite_instructions=rewrite_instructions,
                existing_slides=existing_slides,
                image_metadatas=image_metadatas,
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
                "slides_written": [{
                    "dispatch_id": dispatch_id,
                    "assignment_id": assignment_id,
                    "group_idx": group_idx,
                    "count": saved_count,
                    "target_slide_numbers": target_slide_numbers,
                }],
                "messages":       [msg],
                "retrieval_queries": tool_call_out["retrieval_queries"],
                "tool_calls": tool_call_out["tool_calls"],
                "tool_results": tool_call_out["tool_results"],
            })

        except Exception as e:
            return self._failure(
                e,
                dispatch_id=dispatch_id,
                assignment_id=assignment_id,
                group_idx=group_idx,
                target_slide_numbers=target_slide_numbers,
                tag=tag,
            )


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
        image_metadatas: list[ImageMetadata],
    ) -> str:
        del rewrite_instructions
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
            image_metadatas=image_metadatas,
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
        image_metadatas: list[ImageMetadata],
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
            image_metadatas=image_metadatas,
        )
        return "\n".join(user_prompt_parts)


class SlideWriterAgent(InitialSlideWriterAgent):
    """Backward-compatible alias for the initial slide writer."""


def slide_writer_node(
    state: SlideWriterDispatch,
    *,
    tools_for_agent: dict[str, dict[str, Any]] | None = None,
) -> Command:
    rewrite_instructions = state.get("rewrite_instructions", "")
    agent_cls = SlideRewriterAgent if rewrite_instructions.strip() else InitialSlideWriterAgent
    return agent_cls(
        log_display=_group_log_label(state),
        tools_for_agent=tools_for_agent,
    ).run(state)
