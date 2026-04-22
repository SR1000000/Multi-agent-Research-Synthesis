from __future__ import annotations

import json
from dataclasses import dataclass

from src.llm import LLMConfig, _heal_json, _strip_code_fence, _strip_think_block, get_llm
from src.logging.logger import AgentLogger
from src.processing.document.schema import DocumentContext, ExtractionResult

from .prompts import DOCUMENT_CONTEXT_PROMPT


@dataclass
class DocumentContextConfig:
    model: str = "context"


DEFAULT_DOCUMENT_CONTEXT_CONFIG = DocumentContextConfig()


def document_context_to_json_text(document_context: DocumentContext | None) -> str | None:
    if document_context is None:
        return None
    return document_context.model_dump_json()


def document_context_from_json_text(raw: str | None) -> DocumentContext | None:
    if not raw or not raw.strip():
        return None
    return DocumentContext.model_validate_json(raw)


class DocumentContextualizer:
    def __init__(
        self,
        config: DocumentContextConfig = DEFAULT_DOCUMENT_CONTEXT_CONFIG,
        logger: AgentLogger | None = None,
    ) -> None:
        self.config = config
        self._logger = logger or AgentLogger()
        self._llm = get_llm(config=LLMConfig(model=config.model))

    def contextualize(self, result: ExtractionResult) -> ExtractionResult:
        if result.document_context is not None and result.document_context.sections:
            return result

        section_outline = self._build_section_outline(result)
        prompt = DOCUMENT_CONTEXT_PROMPT.format(
            section_outline=section_outline,
            document_markdown=result.markdown or "",
        )

        raw = self._llm.complete(
            [{"role": "user", "content": prompt}],
            schema=DocumentContext,
        )
        cleaned = _heal_json(_strip_code_fence(_strip_think_block(raw)), DocumentContext)
        result.document_context = DocumentContext.model_validate_json(cleaned)
        return result

    def _build_section_outline(self, result: ExtractionResult) -> str:
        if result.paper_metadata and result.paper_metadata.sections:
            lines: list[str] = []
            for header in result.paper_metadata.sections:
                lines.append(f"- {header}")
            return "\n".join(lines)
        return self._extract_heading_outline(result.markdown or "")

    def _extract_heading_outline(self, markdown: str) -> str:
        lines: list[str] = []
        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if not line.startswith("#"):
                continue
            level = len(line) - len(line.lstrip("#"))
            if level > 2:
                continue
            header = line[level:].strip()
            if not header:
                continue
            indent = "  " if level == 2 else ""
            lines.append(f"{indent}- {header}")
        return "\n".join(lines)
