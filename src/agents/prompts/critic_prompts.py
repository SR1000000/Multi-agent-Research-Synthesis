"""System roles and user-prompt text for `SlideCriticAgent` (structured CriticOutput)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.agents.prompts.common import schema_prompt_contract

CRITIC_ROLE = """
You are a Slide Deck Critic. Your job is to review assigned slides against the
source research chunks and identify only meaningful issues that require correction.
You must evaluate and report issues ONLY within the assigned review criteria for
this pass; do not raise issues from criteria that were not assigned.

Core Directives:
1. Convergence over Perfection: Your goal is incremental improvement, not infinite polish.
An issue is only an "issue" if it triggers your assigned review criteria.
2. Grounding over speculation: Do not require citations for claims that are clearly supported by the provided chunks, but flag hallucinations, unsupported claims, contradictions, and misleading framing.
3. History Respect: Acknowledge when issues from prior cycles have been addressed.
4. Sufficiency Check: If the assigned slides are adequately grounded and understandable, return no actionable issues.
5. Scope Discipline: Review only the assigned scope. If the title slide is assigned for a grounding check, trivially pass it unless the instructions explicitly say otherwise.
6. Criteria Discipline: If you notice a potential issue outside assigned criteria, ignore it for this run.

For each issue found:
- Assign a unique ID (ISS_001, ISS_002, ...)
- Classify: factual_inaccuracy | hallucination | unsupported_claim | logical_gap | structural | clarity | contradiction
- Severity: critical (Blocks publication) | major (Significantly degrades quality) | minor (Polish)
- Provide a precise rewrite instruction that would fix the issue.
"""

# Reserved for a future layout-focused critic pass; not wired in the current graph.
CRITIC_LAYOUT_ROLE = """
Layout and image-placement review criteria (modular block):
Review the assigned slides against the IMAGE ASSETS listed in the prompt.

Image Placement Assessment:
1. If `media_id` is set, verify the referenced image is relevant to the slide's content.
2. If `media_id` is set, verify the layout choice matches the image's aspect ratio:
   - landscape images -> media_top or media_bottom
   - portrait images -> media_left or media_right
   - square images -> media_left or media_right preferred
3. If no image is used but a clearly relevant image asset exists for an evidence or
   insight slide, raise a minor issue suggesting image inclusion with the specific Image ID.
4. Do NOT penalize omission of images when no relevant image is available.
"""


def build_critic_output_format(output_model: type[BaseModel]) -> str:
    """JSON schema contract for critic structured output; `output_model` is typically `CriticOutput`."""
    return schema_prompt_contract(
        output_model,
        extra_rules=[
            "Top-level keys MUST be exactly `summary`, `actionable`, and `issues` - do not wrap the payload in another key.",
            "If no meaningful issues exist, set actionable=false and issues=[].",
            "If one or more issues exist, set actionable=true and include every required field on each issue "
            "(issue_code, severity, issue_type, location, rewrite_instruction).",
            "issue_code values must be unique within this response (e.g. ISS_001, ISS_002).",
            "Use the exact field names issue_code and issue_type - not `id`, `classification`, or other synonyms.",
            "location must pinpoint what to change (e.g. slide number and bullet or heading).",
            "rewrite_instruction must be one concrete edit directive per issue, not only a restatement of the problem.",
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


def build_critic_user_prompt(
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
    """Assemble the full user message for a critic run (ingested by `SlideCriticAgent._call`)."""
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
            "",
            "Identify only significant issues that break grounding, clarity, coherence, or the review criteria. "
            "If no changes are needed, set actionable=false and issues=[].",
            "Review the slides against the BASELINE SOURCE MATERIAL and IN-SESSION RETRIEVAL LOG. Treat this combined evidence as the source of truth for grounding checks.",
            "If the combined evidence is missing support for a concrete claim on a slide, treat that as a grounding issue.",
            "Review the slides against the BASELINE SOURCE MATERIAL and IN-SESSION RETRIEVAL LOG. Treat this combined evidence as the source of truth for grounding checks.",
            "",
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
