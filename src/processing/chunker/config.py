from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticTextSplitterConfig:
    markdown_capacity: tuple[int, int] = (1200, 2200)
    text_capacity: tuple[int, int] = (1000, 2000)
