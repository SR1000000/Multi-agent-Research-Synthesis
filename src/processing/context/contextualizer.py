from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types

from .prompts import ARTIFACT_CONTEXT_PROMPT, CHUNK_CONTEXT_PROMPT

if TYPE_CHECKING:
    from ..schema import ExtractionResult, ExtractedChunk, ExtractedImage, ExtractedTable, ExtractedEquation

logger = logging.getLogger(__name__)

@dataclass
class ContextConfig:
    model: str = "gemini-2.5-flash"
    skip_chunk_token_threshold: int = 120
    api_key: str | None = None


DEFAULT_CONTEXT_CONFIG = ContextConfig()


class GeminiContextualizer:
    """
    Handles document contextualization using the Gemini API via implicit caching.
    """
    def __init__(self, config: ContextConfig = DEFAULT_CONTEXT_CONFIG) -> None:
        self.config = config
        self._client = genai.Client(api_key=config.api_key)

    def contextualize(self, result: ExtractionResult) -> ExtractionResult:
        """
        Main entry point. Iterates through chunks and artifacts to provide context.
        Uses implicit Gemini caching by prefixing every prompt with the document markdown.
        """
        markdown = result.markdown
        
        # Contextualize text chunks
        for chunk in result.source_chunks:
            if len(chunk.text) < self.config.skip_chunk_token_threshold and len(chunk.meta_data.get("headings", [])) >= 2:
                chunk.contextualized_text = chunk.text
                continue
            
            prompt = CHUNK_CONTEXT_PROMPT.format(
                document_markdown=markdown,
                chunk_text=chunk.text
            )
            chunk.contextualized_text = self._generate(prompt)

        # Contextualize multimodal artifacts
        artifacts = list(result.images) + list(result.tables) + list(result.equations)
        for artifact in artifacts:
            text_before, text_after = self._find_surrounding_chunks(artifact.page, result.source_chunks)
            artifact_content = self._get_artifact_content(artifact)
            
            prompt = ARTIFACT_CONTEXT_PROMPT.format(
                document_markdown=markdown,
                text_before=text_before,
                artifact_content=artifact_content,
                text_after=text_after
            )
            artifact.contextualized_text = self._generate(prompt)

        return result

    def _find_surrounding_chunks(
        self,
        artifact_page: int | None,
        chunks: list[ExtractedChunk],
    ) -> tuple[str, str]:
        """Returns (text_before, text_after) for the chunks surrounding an artifact based on page numbers."""
        if not chunks:
            return "", ""

        if artifact_page is None:
            # Fallback: first and last chunk
            return "", chunks[0].text

        # Find the last chunk at or before the artifact's page
        before_idx = -1
        for i, chunk in enumerate(chunks):
            page_nums = chunk.meta_data.get("page_numbers", [])
            if page_nums and max(page_nums) <= artifact_page:
                before_idx = i

        if before_idx == -1:
            # Artifact is before all chunks
            text_before = ""
            text_after = chunks[0].text if chunks else ""
        elif before_idx >= len(chunks) - 1:
            # Artifact is after all chunks
            text_before = chunks[before_idx].text
            text_after = ""
        else:
            text_before = chunks[before_idx].text
            text_after = chunks[before_idx + 1].text

        return text_before, text_after

    def _get_artifact_content(self, artifact: Any) -> str:
        """Returns the raw content of an artifact suitable for LLM prompting."""
        from ..schema import ExtractedImage, ExtractedTable, ExtractedEquation
        if isinstance(artifact, ExtractedImage): return artifact.caption
        if isinstance(artifact, ExtractedTable): return artifact.content
        if isinstance(artifact, ExtractedEquation): return artifact.latex_or_text
        return ""

    def _generate(self, prompt: str) -> str:
        """Generates content and logs implicit cache usage metadata."""
        res = self._client.models.generate_content(
            model=self.config.model,
            contents=prompt
        )
        self._log_cache_usage(res)
        return res.text.strip()

    def _log_cache_usage(self, response: Any) -> None:
        """Logs whether the response utilized the implicit context cache."""
        usage = getattr(response, "usage_metadata", None)
        if usage:
            cached_count = getattr(usage, "cached_content_token_count", 0)
            if cached_count > 0:
                logger.info(f"Implicit Cache HIT: {cached_count} tokens reused.")
            else:
                logger.info("Implicit Cache MISS: Full prefix processed.")
        else:
            logger.debug("No usage metadata found in response.")
