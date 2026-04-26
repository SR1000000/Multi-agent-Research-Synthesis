"""
Slide writer agents.

This module now separates initial generation from revision behavior while
sharing the same prompt-building, LLM, and persistence machinery. The active
graph still calls `slide_writer_node`, which dispatches to the correct thin
agent wrapper based on whether rewrite instructions are present.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.types import Command

from src.agents.base import BaseLLMAgent
from src.agents.prompts.common import ordered_chunk_texts
from src.agents.prompts.writer_prompts import (
    SLIDE_REWRITER_ROLE,
    SLIDE_WRITER_ROLE,
    build_initial_slide_user_prompt,
    build_slide_retrieval_turns,
    build_slide_rewrite_user_prompt,
)
from src.memory.research.schema import ImageMetadata, ProtoSlide, make_slide_batch_model
from src.memory.research.database import ResearchDatabase


# ---------------------------------------------------------------------------
# Dispatch state (sent via LangGraph Send API)
# ---------------------------------------------------------------------------

class SlideWriterDispatch(TypedDict):
    """State payload delivered to each slide_writer node via Send()."""
    plan_number: int
    dispatch_id:            str
    assignment_id:          str
    chunk_ids:             list[str]   # group-wide union passed to the writer as shared working context
    slide_blueprints:      list[dict]  # serialized SlideBlueprint dicts
    group_idx:             int         # index into PresentationPlan.slide_groups
    session_id:            str
    rewrite_instructions:  str         # empty for initial gen; Critic populates for rewrites
    target_slide_numbers:  list[int]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _BaseSlideWorkerAgent(BaseLLMAgent):
    """Abstract base for slide-writing agents that share a two-phase retrieval-then-generation loop.

    Subclasses override ``_build_user_prompt()`` to supply a different prompt contract
    (initial generation vs. critic-driven rewrite).  The shared ``run()`` method handles
    retrieval tool calls, context loading, structured LLM generation, slide count validation,
    and persistence to the research database.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        log_display: str | None = None,
        tools_for_agent: dict[str, dict[str, Any]] | None = None,
    ):
        """Initialise the base worker, forwarding the role-specific system prompt to BaseLLMAgent."""
        super().__init__(
            "slide_writer",
            system_prompt=system_prompt,
            log_display=log_display,
            tools_for_agent=tools_for_agent,
        )

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
        """Build a zero-count error Command so the graph can make progress after an unhandled exception.

        Emits a slides_written record with count=0 so PlanExecutor's completion checks do not
        wait indefinitely for a result that will never arrive.  The exception is also surfaced
        in the ``errors`` state key for downstream inspection.
        """
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
        """Load source chunks, optionally existing slide drafts, and image metadata from the database.

        Returns a 3-tuple of ``(combined_chunk_text, existing_slides, image_metadatas)``.
        Existing slides are only fetched when ``include_existing_slides`` is True (i.e. during
        rewrites) so initial generation cannot accidentally see stale drafts from a prior pass.
        """
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
            # Load existing slides only for rewrites; initial generation should not see stale drafts.
            if include_existing_slides:
                for bp_dict in slide_blueprints:
                    slide_num = bp_dict.get("slide_number")
                    if slide_num is None:
                        continue
                    existing_slide = research_db.load_slide(slide_num)
                    if existing_slide is not None:
                        existing_slides.append(existing_slide)
            image_metadatas = research_db.get_images_for_chunks(chunk_ids) if chunk_ids else []

        chunk_texts = ordered_chunk_texts(rows, chunk_ids)
        return "\n\n".join(chunk_texts), existing_slides, image_metadatas

    def _tool_payload_text(self, tool_results: list[dict[str, Any]]) -> str:
        """Serialize tool-call payloads to a single JSON text block.

        Tool-use mode and structured-schema mode are separate LLM call paths; converting
        the retrieved payloads to plain text bridges the gap so evidence from the retrieval
        phase reaches the structured generation call as part of the user prompt.
        """
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
        """Build the user-turn prompt for the structured slide generation call.

        Must be overridden by each concrete subclass with its own prompt contract.
        The shared ``run()`` loop calls this after retrieval is complete, passing in
        all available context; subclasses may ignore inputs irrelevant to their role.
        """
        raise NotImplementedError

    def run(self, state: SlideWriterDispatch) -> Command:
        """Orchestrate the two-phase write loop for one slide group assignment.

        Phase 1 — Retrieval: issues a tool-calling LLM pass to fetch relevant evidence
            from the research database against the slide blueprints.
        Phase 2 — Generation: sends the retrieved context to a structured LLM call that
            produces exactly one typed slide object per blueprint.

        Validates the output slide count, persists each ProtoSlide to research.db, and
        returns a Command carrying a slides_written record for the plan executor.
        All exceptions are caught and forwarded via ``_failure()`` so the graph never stalls.
        """
        self._set_session_id(state)
        self._set_plan_number(state)
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
            retrieval_turns = build_slide_retrieval_turns(
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
    """Writes first-draft slides from source chunks and plan blueprints.

    Uses the initial-writer system prompt and ignores any rewrite instructions or existing
    slide drafts, ensuring first drafts are generated cleanly from source evidence only.
    """

    def __init__(
        self,
        *,
        log_display: str | None = None,
        tools_for_agent: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialise with the initial-writer role prompt."""
        super().__init__(
            system_prompt=SLIDE_WRITER_ROLE,
            log_display=log_display,
            tools_for_agent=tools_for_agent,
        )

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
        """Build the initial-generation prompt, discarding rewrite-only inputs.

        Explicitly deletes ``rewrite_instructions`` and ``existing_slides`` to prevent
        accidental state leakage into first-draft generation.
        """
        del rewrite_instructions, existing_slides
        return build_initial_slide_user_prompt(
            slide_count=slide_count,
            slide_blueprints=slide_blueprints,
            combined_text=combined_text,
            image_metadatas=image_metadatas,
        )


class SlideRewriterAgent(_BaseSlideWorkerAgent):
    """Revises existing slide drafts in response to structured critic feedback.

    Uses the rewriter system prompt and supplies both the current slide content and the
    rewrite instructions so the LLM can apply targeted fixes while preserving valid
    content from the prior draft.
    """

    def __init__(
        self,
        *,
        log_display: str | None = None,
        tools_for_agent: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialise with the rewriter role prompt."""
        super().__init__(
            system_prompt=SLIDE_REWRITER_ROLE,
            log_display=log_display,
            tools_for_agent=tools_for_agent,
        )

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
        """Build the rewrite prompt, supplying both the existing draft and critic instructions.

        Passing the current slide content alongside the critique lets the LLM preserve
        valid material and make only the targeted changes the critic requested.
        """
        return build_slide_rewrite_user_prompt(
            slide_count=slide_count,
            slide_blueprints=slide_blueprints,
            combined_text=combined_text,
            image_metadatas=image_metadatas,
            rewrite_instructions=rewrite_instructions,
            existing_slides=existing_slides,
        )

def slide_writer_node(
    state: SlideWriterDispatch,
    *,
    tools_for_agent: dict[str, dict[str, Any]] | None = None,
) -> Command:
    """LangGraph node entry point shared by initial write and rewrite passes.

    Inspects ``rewrite_instructions`` to decide whether to delegate to
    ``SlideRewriterAgent`` (non-empty) or ``InitialSlideWriterAgent`` (empty/absent),
    then runs the selected agent against the dispatched state.
    """
    blueprints = state.get("slide_blueprints", [])

    # Modify log prefixes so parallel writers are easy to tell apart in logs.
    if not blueprints:
        log_display = "SlideWriter[empty]"
    else:
        nums = [bp.get("slide_number", "?") for bp in blueprints]
        log_display = f"SlideWriter[slides {nums[0]}-{nums[-1]}, group {state.get('group_idx', '?')}]"
    
    rewrite_instructions = state.get("rewrite_instructions", "")
    agent_cls = SlideRewriterAgent if rewrite_instructions.strip() else InitialSlideWriterAgent
    return agent_cls(
        log_display=log_display,
        tools_for_agent=tools_for_agent,
    ).run(state)
