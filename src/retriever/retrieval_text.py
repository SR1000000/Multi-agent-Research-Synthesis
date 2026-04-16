from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")


def preferred_text(text: str, contextualized_text: str | None) -> str:
    base = (text or "").strip()
    contextualized = (contextualized_text or "").strip()
    if contextualized and base:
        return f"{contextualized}\n\n{base}"
    return contextualized or base


def strip_html_text(raw: str) -> str:
    if not raw:
        return ""
    t = _TAG_RE.sub(" ", raw)
    t = html.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def table_fallback_text(contextualized_text: str | None, content_html: str) -> str:
    base = strip_html_text(content_html or "")
    contextualized = (contextualized_text or "").strip()
    if contextualized and base:
        return f"{contextualized}\n\n{base}"
    return contextualized or base
