"""System roles and user-prompt text for `SlideCriticAgent` (structured CriticOutput)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.agents.prompts.common import schema_prompt_contract

CRITIC_ROLE = """
You are a Slide Deck Critic. Your job is to review the assigned scope and surface only
meaningful issues that require correction. You must evaluate and report issues ONLY within
the assigned review criteria for this pass; do not raise issues from criteria that were not assigned.

Core Directives:
1. Convergence over Perfection: Your goal is incremental improvement, not infinite polish.
2. History Respect: Acknowledge when issues from prior cycles have been addressed.
3. Scope Discipline: Review only the assigned scope.
4. Criteria Discipline: If you notice a potential issue outside your assigned criteria, ignore it for this run.

For each issue found:
- Assign a unique ID (ISS_001, ISS_002, ...)
- Severity: critical (Blocks publication) | major (Significantly degrades quality) | minor (Polish)
- Location: pinpoint what to change (e.g. slide number and bullet or heading).
- Provide a precise rewrite instruction that would fix the issue.
"""

GROUNDING_REVIEW_CRITERIA = """
Review Criteria:
2. Grounding over speculation: Do not require citations for claims that are clearly supported by the provided chunks, but flag hallucinations, unsupported claims, contradictions, and misleading framing.
4. Sufficiency Check: If the assigned slides are adequately grounded and understandable, return no actionable issues.

For each issue found:
- Classify: factual_inaccuracy | hallucination | unsupported_claim | logical_gap | structural | clarity | contradiction

How to use evidence (the user message below includes baseline source chunks, in-session retrieval log, and image assets when applicable):
- Review slide claims against the BASELINE SOURCE MATERIAL and IN-SESSION RETRIEVAL LOG together as the source of truth for grounding. Treat the combined evidence as what the writer was allowed to use.
- If the combined evidence does not support a concrete claim on a slide, that is a grounding issue.
- Identify only significant issues that require correction under these criteria.
"""

NARRATIVE_REVIEW_CRITERIA = """
Review Criteria:
1. Narrative coherence: The deck should read as one story—logical order, clear transitions, and a thesis that the slides support end-to-end.
2. Clarity: Each slide’s role in the arc should be obvious; flag confusion, redundancy, or missing connective tissue between slides.
3. Pacing: Flag abrupt jumps, repeated beats, or a weak opening/closing relative to the rest of the deck.

You are not given source research text—judge only how slide content and plan intent work together.  
Ignore the key message of each slide when reviewing for narrative coherence.  Focus only on the titles, bullet points, and speaker notes.
"""


def build_critic_output_format(output_model: type[BaseModel]) -> str:
    """JSON schema contract for critic structured output; `output_model` is typically `CriticOutput`."""
    return schema_prompt_contract(
        output_model,
        extra_rules=[
            "Top-level keys MUST be exactly `summary`, `actionable`, and `issues` - do not wrap the payload in another key.",
            "If no meaningful issues exist, set actionable=false and issues=[].",
            "If one or more issues exist, set actionable=true and include every required field on each issue "
            "(issue_code, severity, issue_type, location, affected_slide_numbers, rewrite_instruction).",
            "On every issue, affected_slide_numbers must list every slide the issue applies to (non-empty for actionable issues); "
            "for whole-deck concerns, list all target slide numbers.",
            "issue_code values must be unique within this response (e.g. ISS_001, ISS_002).",
            "Use the exact field names issue_code and issue_type - not `id`, `classification`, or other synonyms.",
            "location must pinpoint what to change (e.g. slide number and bullet or heading).",
            "rewrite_instruction must be one concrete edit directive per issue, not only a restatement of the problem.",
            "If no changes are needed, set actionable=false and issues=[].",
        ],
    )


def format_retrieved_artifact_row(row: dict[str, Any]) -> str:
    """Format one normalized artifact row for the IN-SESSION RETRIEVAL LOG in the critic user prompt."""
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


def build_grounding_critic_user_prompt(
    *,
    cycle_number: int,
    check_type: str,
    scope_type: str,
    scope_id: str,
    target_slide_numbers: list[int],
    blueprint_block: str,
    slides_block: str,
    baseline_chunks_block: str,
    retrieval_log: str,
    image_block: str,
    output_model: type[BaseModel],
) -> str:
    """Assemble the user message for a grounding / evidence-aware critic run."""
    return "\n".join(
        [
            f"Cycle: {cycle_number}",
            f"Check type: {check_type}",
            f"Scope: {scope_type}::{scope_id}",
            f"Target slides: {target_slide_numbers}",
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
            build_critic_output_format(output_model),
        ]
    )


def build_narrative_critic_user_prompt(
    *,
    cycle_number: int,
    check_type: str,
    scope_type: str,
    scope_id: str,
    target_slide_numbers: list[int],
    blueprint_block: str,
    slides_block: str,
    output_model: type[BaseModel],
) -> str:
    """Assemble the user message for a narrative / deck-flow critic (no source chunks or RAG)."""
    return "\n".join(
        [
            f"Cycle: {cycle_number}",
            f"Check type: {check_type}",
            f"Scope: {scope_type}::{scope_id}",
            f"Target slides: {target_slide_numbers}",
            "",
            "SLIDE ASSIGNMENTS (plan intent for slides in scope):",
            blueprint_block or "(none)",
            "",
            "CURRENT SLIDES (draft content, presentation order):",
            slides_block,
            build_critic_output_format(output_model),
        ]
    )


def format_rewrite_instruction(issue: dict) -> str:
    """Format one issue dict into a line for `rewrite_instructions` sent to the slide writer."""
    slide_nums = issue.get("affected_slide_numbers") or []
    loc = issue.get("location", "").strip()
    context_parts: list[str] = []
    if slide_nums:
        context_parts.append(f"Slide(s) {', '.join(str(n) for n in slide_nums)}")
    if loc and loc.lower() not in ("none", "n/a", "general", "all"):
        context_parts.append(f"Location: {loc}")
    prefix = f"[{' | '.join(context_parts)}] " if context_parts else ""
    return f"- {prefix}{issue['rewrite_instruction']}"
