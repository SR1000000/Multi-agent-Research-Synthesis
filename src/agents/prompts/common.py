"""Shared prompt text and string builders for multiple agents (schema contracts, image blocks, retries)."""

from __future__ import annotations

import json

from pydantic import BaseModel

from src.memory.research.schema import ImageMetadata, ProtoSlide


def schema_prompt_contract(
    schema: type[BaseModel],
    *,
    root_key: str | None = None,
    extra_rules: list[str] | None = None,
) -> str:
    """Build a concise output format prompt contract from a Pydantic schema (used in planner, critic, etc.).
    Used mainly for json_object response format, when direct json_schema response format is not supported.
    Though still useful for guiding output even with json_schema response format, especially with extra rules."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    lines = [
        "### REQUIRED ROOT JSON SHAPE:",
        "- Return exactly ONE top-level JSON object matching the schema below.",
    ]
    if root_key:
        lines.append(f"- The top-level key MUST be `{root_key}`.")
    lines.extend(
        [
            "- Do NOT return multiple top-level objects.",
            "- Do NOT return newline-delimited JSON.",
            "- Do NOT return a top-level array.",
            "- Do NOT include any text before or after the JSON object.",
        ]
    )
    if extra_rules:
        lines.append("")
        lines.append("### ADDITIONAL RULES:")
        lines.extend(f"- {rule}" for rule in extra_rules)
    lines.extend(
        [
            "",
            "### EXACT JSON SCHEMA:",
            schema_json,
        ]
    )
    return "\n".join(lines)


def build_structured_retry_turns(
    current_turns: list[dict],
    clean_response: str,
    error_summary: str,
    schema: type[BaseModel],
) -> list[dict]:
    """User turn appended after a failed structured LLM output (BaseLLMAgent _call_structured)."""
    return [
        *current_turns,
        {"role": "assistant", "content": clean_response},
        {
            "role": "user",
            "content": (
                f"Your previous response failed validation:\n{error_summary}\n\n"
                f"Required JSON schema:\n{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
                "Respond with ONLY a valid JSON object that matches the schema above. "
                "Do NOT wrap in markdown fences, add explanations, or include any text outside the JSON."
            ),
        },
    ]


def format_image_assets_block(images: list[ImageMetadata]) -> str:
    """Compact IMAGE ASSETS block for slide writer and critic user prompts (so they can work with images)."""
    if not images:
        return ""
    lines = [
        "### IMAGE ASSETS",
        "Set `media_id` to one of these IDs when an image supports a slide. Prefer contextualized description when available; otherwise use the raw caption.",
        "Each line includes `bbox` (region on the source PDF page) when available.",
    ]
    for img in images:
        desc = img.contextualized_text or img.caption or "(no description)"
        bbox_s = json.dumps(img.bbox, separators=(",", ":")) if img.bbox else "null"
        lines.append(
            f"- `{img.id}` — aspect={img.aspect_ratio} — bbox={bbox_s} — {desc}"
        )
    return "\n".join(lines)


def format_slide_for_prompt(slide: ProtoSlide) -> str:
    """Serialize a proto-slide to JSON for critic review and writer revision prompts.
    Reused enough that it's worth a function."""
    return json.dumps(
        {
            "slide_number": slide.slide_number,
            "content": slide.content.model_dump(mode="json"),
            "chunk_references": slide.chunk_references,
        },
        indent=2,
    )
