from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from src.llm import get_llm, LLMConfig
from src.logging.logger import AgentLogger

from .prompts import ARTIFACT_CONTEXT_PROMPT, CHUNK_CONTEXT_PROMPT
from src.memory.objectstore.image_uploader import ImageUploader

from src.processing.document.schema import ExtractionResult, ExtractedChunk, ExtractedImage, ExtractedTable, ExtractedEquation
from src.memory.objectstore.provider import ObjectStoreProvider


@dataclass
class ContextConfig:
    """Configuration for document contextualization.

    Attributes:
        model: Model identifier for contextualization (default: "context" alias from config.yaml)
               Uses light, cheap, multimodal models: gemini-2.0-flash, gemini-2.5-flash-lite
        skip_chunk_token_threshold: Minimum token count to skip contextualization for short chunks with sufficient headings
        max_concurrency: Reserved; batch requests currently run sequentially to avoid provider overload
        batch_size: Number of items to process in each batch
    """
    model: str = "context"
    skip_chunk_token_threshold: int = 120
    max_concurrency: int = 1
    batch_size: int = 20      # Items per batch


DEFAULT_CONTEXT_CONFIG = ContextConfig()


class Contextualizer:
    """
    Handles document contextualization using LiteLLM.

    Supports implicit caching via LiteLLM's cache_control for providers that support it
    (Anthropic, Gemini). Add cache_control to system message parts in _generate() if needed.
    """
    def __init__(
        self,
        config: ContextConfig = DEFAULT_CONTEXT_CONFIG,
        object_store: "ObjectStoreProvider | None" = None,
        logger: AgentLogger | None = None,
    ) -> None:
        self.config = config
        self._logger = logger or AgentLogger()
        self._llm_config = LLMConfig(
            model=config.model,
        )
        self._object_store = object_store
        self._llm = None
        try:
            self._llm = get_llm(config=self._llm_config)
        except Exception as e:
            self._logger.log(
                f"Failed to initialize contextualizer LLM for model alias '{config.model}': {e}. "
                "Contextualizer will return original text/captions.",
                level="warning",
            )
        self._image_uploader: ImageUploader | None = None
        if object_store is not None:
            self._image_uploader = ImageUploader(object_store)
        self._multimodal_disabled = False

    async def contextualize(self, result: ExtractionResult) -> ExtractionResult:
        try:
            markdown = result.markdown

            if self._llm is None:
                for chunk in result.source_chunks:
                    chunk.contextualized_text = chunk.text
                    self._logger.log(
                        f"chunk_contextualization chunk_id={chunk.id} chunk_type=text_chunk status=passed",
                        level="info",
                    )
                for image in result.images:
                    image.contextualized_text = image.caption
                for table in result.tables:
                    table.contextualized_text = table.content
                for equation in result.equations:
                    equation.contextualized_text = equation.latex_or_text
                return result

            chunks_todo = [c for c in result.source_chunks if not c.contextualized_text]
            artifacts_todo = [
                a for a in list(result.images) + list(result.tables) + list(result.equations)
                if not a.contextualized_text
            ]

            await self._process_items_in_batches(
                chunks_todo,
                artifacts_todo,
                markdown,
                result.source_chunks,
            )
            return result
        except KeyboardInterrupt:
            self._logger.log(
                f"Contextualization interrupted (KeyboardInterrupt); returning partial ExtractionResult doc_id={result.doc_id}",
                level="warning",
            )
            return result

    async def _process_items_in_batches(
        self,
        chunks_todo: list[ExtractedChunk],
        artifacts_todo: list[Any],
        markdown: str | None,
        source_chunks: list[ExtractedChunk],
    ) -> None:
        """Process chunks and artifacts using batched LLM calls without concurrent batch requests."""

        # Separate text and multimodal items
        text_items = chunks_todo[:]
        multimodal_items = [a for a in artifacts_todo if isinstance(a, ExtractedImage)]
        other_artifacts = [a for a in artifacts_todo if not isinstance(a, ExtractedImage)]

        # Process text items (chunks, tables, equations) in batches
        if text_items or other_artifacts:
            all_text_items = text_items + other_artifacts
            text_batches = self._create_batches(all_text_items, self.config.batch_size)

            async def process_text_batch(batch):
                # Skip items with deterministic fallbacks instead of sending empty requests.
                batch_requests: list[tuple[Any, list[dict]]] = []
                for item in batch:
                    if isinstance(item, ExtractedChunk):
                        payload = self._build_chunk_payload(item, markdown)
                    else:  # Table or Equation
                        text_before, text_after = self._find_surrounding_chunks(item.page, source_chunks)
                        payload = self._build_artifact_payload(
                            markdown,
                            text_before,
                            self._get_artifact_content(item),
                            text_after,
                        )
                    if not payload:
                        if isinstance(item, ExtractedChunk):
                            item.contextualized_text = item.text
                        else:
                            item.contextualized_text = self._get_artifact_content(item)
                        continue
                    batch_requests.append((item, payload))
                if not batch_requests:
                    return

                results = await self._llm.batch_complete(
                    [payload for _, payload in batch_requests],
                )

                for (item, _payload), result_str in zip(batch_requests, results):
                    if isinstance(result_str, Exception):
                        self._logger.log(f"Failed to contextualize {item.id}: {result_str}", level="error")
                        if isinstance(item, ExtractedChunk):
                            item.contextualized_text = item.text
                        else:
                            item.contextualized_text = self._get_artifact_content(item)
                    else:
                        validated = self._validate_contextualized_text(result_str)
                        item.contextualized_text = validated or (item.text if isinstance(item, ExtractedChunk) else self._get_artifact_content(item))

            for i, batch in enumerate(text_batches):
                try:
                    await process_text_batch(batch)
                except Exception as exc:
                    self._logger.log(f"Text batch contextualization failed: {exc}", level="error")

        # Process multimodal items (images) in batches
        if multimodal_items and not self._multimodal_disabled:
            multimodal_batches = self._create_batches(multimodal_items, self.config.batch_size)

            async def process_multimodal_batch(batch):
                payloads = []
                for item in batch:
                    text_before, text_after = self._find_surrounding_chunks(item.page, source_chunks)
                    payload = self._build_multimodal_payload(item, markdown, text_before, text_after)
                    payloads.append(payload)

                results = await self._llm.batch_complete(
                    payloads
                )

                for i, result_str in enumerate(results):
                    item = batch[i]
                    if isinstance(result_str, Exception):
                        self._logger.log(f"Failed to contextualize image {item.id}: {result_str}", level="error")
                        item.contextualized_text = item.caption
                    else:
                        validated = self._validate_contextualized_text(result_str)
                        item.contextualized_text = validated or item.caption

            for i, batch in enumerate(multimodal_batches):
                try:
                    await process_multimodal_batch(batch)
                except Exception as exc:
                    self._logger.log(f"Image batch contextualization failed: {exc}", level="error")

    def _create_batches(self, items: list, batch_size: int) -> list[list]:
        """Split items into batches of specified size."""
        if not items:
            return []
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def _build_chunk_payload(self, chunk: ExtractedChunk, markdown: str | None) -> list[dict]:
        """Build the message payload for a text chunk."""
        if len(chunk.text) < self.config.skip_chunk_token_threshold and len(chunk.meta_data.get("headings", [])) >= 2:
            self._logger.log(
                f"chunk_contextualization chunk_id={chunk.id} chunk_type=text_chunk status=skipped_short_chunk_with_headings",
                level="info",
            )
            return []

        if not markdown:
            # Fallback to simple prompt without caching
            prompt = CHUNK_CONTEXT_PROMPT.format(
                document_markdown="",
                chunk_text=chunk.text
            )
            return [{'role': 'user', 'content': prompt}]

        system_text = f"<document>\n{markdown}\n</document>"
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
            {
                "role": "user",
                "content": (
                    "Here is the chunk we want to situate within the whole document.\n"
                    f"<chunk>\n{chunk.text}\n</chunk>\n\n"
                    "Provide a succinct context in 1 paragraph, no more than 5 sentences, "
                    "describing this chunk's specific position and role within the document "
                    "for the purposes of improving search retrieval. Answer only with the "
                    "succinct context and nothing else. \nExample: original_chunk = "
                    "\"The company's revenue grew by 3% over the previous quarter.\" "
                    "contextualized_chunk = \"This chunk is from an SEC filing on ACME corp's "
                    "performance in Q2 2023; the previous quarter's revenue was $314 million. "
                    "The company's revenue grew by 3% over the previous quarter.\""
                )
            }
        ]

        try:
            messages[0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        except (IndexError, KeyError):
            pass  # Fallback if cache_control is not supported

        return messages

    def _build_artifact_payload(self, document_markdown: str | None, text_before: str, artifact_content: str, text_after: str) -> list[dict]:
        """Build the message payload for a non-image artifact."""
        if not document_markdown:
            # Fallback to simple prompt without caching
            prompt = ARTIFACT_CONTEXT_PROMPT.format(
                document_markdown="",
                text_before=text_before,
                artifact_content=artifact_content,
                text_after=text_after
            )
            return [{'role': 'user', 'content': prompt}]

        system_text = f"<document>\n{document_markdown}\n</document>"
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
            {
                "role": "user",
                "content": f"Here is the artifact we want to situate, along with surrounding text.\n<text_before>\n{text_before}\n</text_before>\n<artifact>\n{artifact_content}\n</artifact>\n<text_after>\n{text_after}\n</text_after>\n\nProvide a succinct context in 1 paragraph, no more than 5 sentences, describing this artifact's role and specific position within the document for the purposes of improving search retrieval. Focus on how the surrounding narrative distinguishes this artifact. Answer only with the succinct context and nothing else."
            }
        ]

        try:
            messages[0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        except (IndexError, KeyError):
            pass  # Fallback if cache_control is not supported

        return messages

    def _resolve_image_url(self, image: ExtractedImage) -> str:
        """Return a fresh fetchable image URL for multimodal requests when possible."""
        if self._object_store is not None and image.storage_path:
            presign = getattr(self._object_store, "generate_presigned_url_for_location", None)
            if callable(presign):
                try:
                    return presign(image.storage_path)
                except Exception as exc:
                    self._logger.log(
                        f"Failed to generate presigned URL from storage_path for {image.id}: {exc}",
                        level="warning",
                    )

        if self._image_uploader is not None and image.base64_data:
            upload_url = self._image_uploader.get_or_upload_url(
                doc_id=image.source_filename or "document",
                image_id=image.id,
                base64_data=image.base64_data,
            )
            if upload_url:
                return upload_url

        return image.storage_path or ""

    def _build_multimodal_payload(self, image: ExtractedImage, document_markdown: str | None, text_before: str, text_after: str) -> list[dict]:
        """Build the message payload for an image."""
        image_url = self._resolve_image_url(image)

        if image_url:
            # Format multimodal message with image URL
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{document_markdown}\n</document>\n\n"
                                f"Surrounding text:\n<before>{text_before}</before>\n"
                                f"<after>{text_after}</after>\n\n"
                                f"[IMAGE]: The image shows: {image.caption}\n\n"
                                "Provide a succinct context in 1 paragraph, no more than 5 sentences describing this image's role and specific position within the document.\n"
                                "Focus on how the surrounding narrative frames this image and what information it conveys.\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }]
            return messages

        # If no storage_path, fall back to text-only
        return self._build_artifact_payload(document_markdown, text_before, f"[IMAGE]: {image.caption}", text_after)

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
        if isinstance(artifact, ExtractedImage): return artifact.caption
        if isinstance(artifact, ExtractedTable): return artifact.content
        if isinstance(artifact, ExtractedEquation): return artifact.latex_or_text
        return ""

    def _validate_contextualized_text(self, text: str | None) -> str:
        """ 
        The LLM call can succeed at the transport/provider level but still return unusable
        text (empty output, truncated output, or a surfaced self-referential failure message).
        This lightweight guard is only for post-response quality screening; 
        native call failures should still be handled through the LLM exception path.
        """
        if not text or len(text.strip()) < 10:
            return ""
        if any(err in text.lower() for err in ["failed to", "unable to", "exception occurred"]):
            return ""
        return text.strip()
