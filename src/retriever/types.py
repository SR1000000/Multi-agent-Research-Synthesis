from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RetrievedItem:
    kind: Literal["chunk", "table"]
    id: str
    document_id: str
    text: str
    score: float | None = None
