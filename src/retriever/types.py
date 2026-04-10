from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RetrievedItem:
    kind: Literal["chunk", "table", "equation", "image"]
    id: str
    document_id: str
    text: str
    score: float | None = None
    # For images, text stores the URL
    # For equations, text stores the LaTeX representation
