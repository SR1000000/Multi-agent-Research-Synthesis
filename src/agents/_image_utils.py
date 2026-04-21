"""Shared helpers for image metadata in agent prompts."""

from __future__ import annotations

import json

from src.memory.research.schema import ImageMetadata


def format_image_assets_block(images: list[ImageMetadata]) -> str:
    """Compact IMAGE ASSETS block for slide writer and critic prompts."""
    if not images:
        return ""
    lines = [
        "### IMAGE ASSETS",
        "Set `media_id` to one of these IDs when an image supports a slide. Prefer the VLM line; use Caption if VLM is empty.",
        "Each line includes `bbox` (region on the source PDF page) when available.",
    ]
    for img in images:
        desc = img.vlm_caption or img.caption or "(no description)"
        bbox_s = json.dumps(img.bbox, separators=(",", ":")) if img.bbox else "null"
        lines.append(
            f"- `{img.id}` — aspect={img.aspect_ratio} — bbox={bbox_s} — {desc}"
        )
    return "\n".join(lines)
