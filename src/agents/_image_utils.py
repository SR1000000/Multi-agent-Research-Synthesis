"""Shared helpers for image metadata in agent prompts."""

from __future__ import annotations

from src.memory.research.schema import ImageMetadata


def format_image_assets_block(images: list[ImageMetadata]) -> str:
    """Compact IMAGE ASSETS block for slide writer and critic prompts."""
    if not images:
        return ""
    lines = [
        "### IMAGE ASSETS",
        "Set `media_id` to one of these IDs when an image supports a slide. Prefer the VLM line; use Caption if VLM is empty.",
    ]
    for img in images:
        desc = img.vlm_caption or img.caption or "(no description)"
        lines.append(f"- `{img.id}` — aspect={img.aspect_ratio} — {desc}")
    return "\n".join(lines)
