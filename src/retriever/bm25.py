from __future__ import annotations

import html
import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_TAG_RE = re.compile(r"<[^>]+>")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def strip_html_for_index(raw: str) -> str:
    if not raw:
        return ""
    t = _TAG_RE.sub(" ", raw)
    t = html.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def chunk_display_text(text: str, contextualized: str | None) -> str:
    if contextualized and contextualized.strip():
        return contextualized.strip()
    return text or ""


def chunk_bm25_text(text: str, contextualized: str | None) -> str:
    return chunk_display_text(text, contextualized)


def table_bm25_text(contextualized: str | None, content_html: str) -> str:
    if contextualized and contextualized.strip():
        return contextualized.strip()
    return strip_html_for_index(content_html or "")
