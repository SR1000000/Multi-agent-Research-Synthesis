from __future__ import annotations

import asyncio
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from threading import Thread
from typing import Any, Literal

from src.llm import get_llm, LLMConfig

from .prompts import ARTIFACT_CONTEXT_PROMPT, CHUNK_CONTEXT_PROMPT
from .image_uploader import ImageUploader

from src.processing.document.schema import ExtractionResult, ExtractedChunk, ExtractedImage, ExtractedTable, ExtractedEquation
from src.memory.objectstore.provider import ObjectStoreProvider

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for document contextualization.

    Attributes:
        model: Model identifier for contextualization (default: "context" alias from config.yaml)
               Uses light, cheap, multimodal models: gemini-2.0-flash, gemini-2.5-flash-lite
        skip_chunk_token_threshold: Minimum token count to skip contextualization for short chunks with sufficient headings
        max_concurrency: Maximum number of concurrent batch requests
        batch_size: Number of items to process in each batch
    """
    model: str = "context"
    skip_chunk_token_threshold: int = 120
    max_concurrency: int = 2  # Reduced for rate limit control
    batch_size: int = 20      # Items per batch


DEFAULT_CONTEXT_CONFIG = ContextConfig()


class Contextualizer:
    """
    Handles document contextualization using LiteLLM.

    Supports implicit caching via LiteLLM's cache_control for providers that support it
    (Anthropic, Gemini). Add cache_control to system message parts in _generate() if needed.
    """
    def __init__(self, config: ContextConfig = DEFAULT_CONTEXT_CONFIG, object_store: "ObjectStoreProvider | None" = None) -> None:
        self.config = config
        self._llm_config = LLMConfig(
            model=config.model,
        )
        self._llm = None
        try:
            self._llm = get_llm(config=self._llm_config)
        except Exception as e:
            logger.error(
                "Failed to initialize contextualizer LLM for model alias '%s': %s. "
                "Contextualizer will return original text/captions.",
                config.model,
                e,
            )
        self._image_uploader: ImageUploader | None = None
        if object_store is not None:
            self._image_uploader = ImageUploader(object_store)
        self._multimodal_disabled = False

    def contextualize(self, result: ExtractionResult) -> ExtractionResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(self.contextualize_async(result))
            except KeyboardInterrupt:
                logger.warning(
                    "Contextualization interrupted (KeyboardInterrupt); returning partial ExtractionResult doc_id=%s",
                    result.doc_id,
                )
                return result

        payload: dict[str, ExtractionResult] = {"result": result}
        error: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                payload["result"] = asyncio.run(self.contextualize_async(result))
            except BaseException as exc:  # pragma: no cover - defensive thread boundary
                error["exc"] = exc

        thread = Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "exc" in error:
            exc = error["exc"]
            if isinstance(exc, KeyboardInterrupt):
                logger.warning(
                    "Contextualization interrupted (KeyboardInterrupt); returning partial ExtractionResult doc_id=%s",
                    result.doc_id,
                )
                return result
            raise exc
        return payload["result"]

    @staticmethod
    def _llm_rejects_vision_input(exc: BaseException) -> bool:
        """Heuristic: router / provider cannot serve image_url multimodal requests."""
        text = f"{type(exc).__name__}: {exc}".lower()
        if "no endpoints found" in text and "image" in text:
            return True
        if "does not support" in text and "image" in text:
            return True
        if "image input" in text or "image_url" in text and "not" in text:
            return True
        if "vision" in text and ("not" in text or "unsupported" in text or "no " in text):
            return True
        return False

    @staticmethod
    def _probe_image_url(url: str, timeout: float = 12.0) -> tuple[bool, str]:
        """Best-effort HEAD check so logs show fetchability before LLM multimodal call."""
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
                return True, f"HEAD status={code}"
        except urllib.error.HTTPError as e:
            return False, f"HEAD HTTP {e.code}: {e.reason}"
        except Exception as e:
            return False, str(e)

    async def contextualize_async(self, result: ExtractionResult) -> ExtractionResult:
        """
        Main entry point. Processes chunks and artifacts using batched LLM calls with concurrency.

        Uses prompt caching for the document markdown (system message) to reduce
        token costs when contextualizing multiple chunks/artifacts from the same document.
        """
        markdown = result.markdown

        if self._llm is None:
            for chunk in result.source_chunks:
                chunk.contextualized_text = chunk.text
                logger.info(
                    "chunk_contextualization chunk_id=%s chunk_type=%s status=%s",
                    chunk.id,
                    "text_chunk",
                    "passed",
                )
            for image in result.images:
                image.contextualized_text = image.caption
            for table in result.tables:
                table.contextualized_text = table.content
            for equation in result.equations:
                equation.contextualized_text = equation.latex_or_text
            return result

        # Collect items that need contextualization
        chunks_todo = [c for c in result.source_chunks if not c.contextualized_text]
        artifacts_todo = [
            a for a in list(result.images) + list(result.tables) + list(result.equations)
            if not a.contextualized_text
        ]

        # Process items in batches
        await self._process_items_in_batches(chunks_todo, artifacts_todo, markdown)

        return result

    def _collect_items(self, result: ExtractionResult) -> list[Any]:
        """Collects all chunks and artifacts that need contextualization."""
        items = []
        for chunk in result.source_chunks:
            if not chunk.contextualized_text and not (len(chunk.text) < self.config.skip_chunk_token_threshold and len(chunk.meta_data.get("headings", [])) >= 2):
                items.append(chunk)
        for artifact in list(result.images) + list(result.tables) + list(result.equations):
            if not artifact.contextualized_text:
                items.append(artifact)
        return items

    async def _process_items_in_batches(
        self,
        chunks_todo: list[ExtractedChunk],
        artifacts_todo: list[Any],
        markdown: str | None
    ) -> None:
        """Process chunks and artifacts using batched LLM calls with concurrency."""

        # Separate text and multimodal items
        text_items = chunks_todo[:]
        multimodal_items = [a for a in artifacts_todo if isinstance(a, ExtractedImage)]
        other_artifacts = [a for a in artifacts_todo if not isinstance(a, ExtractedImage)]

        # Process text items (chunks, tables, equations) in batches
        if text_items or other_artifacts:
            all_text_items = text_items + other_artifacts
            text_batches = self._create_batches(all_text_items, self.config.batch_size)
            semaphore = asyncio.Semaphore(self.config.max_concurrency)

            async def process_text_batch(batch, batch_idx):
                async with semaphore:
                    # Build message payloads for this batch
                    payloads = []
                    for item in batch:
                        if isinstance(item, ExtractedChunk):
                            payload = self._build_chunk_payload(item, markdown)
                        else:  # Table or Equation
                            text_before, text_after = self._find_surrounding_chunks(item.page, [])
                            payload = self._build_artifact_payload(markdown, text_before, self._get_artifact_content(item), text_after)
                        payloads.append(payload)

                    # Execute batch
                    results = await asyncio.to_thread(
                        self._llm.batch_complete,
                        payloads
                    )

                    # Apply results back to items
                    for i, result_str in enumerate(results):
                        item = batch[i]
                        if isinstance(result_str, Exception):
                            logger.error(f"Failed to contextualize {item.id}: {result_str}")
                            if isinstance(item, ExtractedChunk):
                                item.contextualized_text = item.text
                            else:
                                item.contextualized_text = self._get_artifact_content(item)
                        else:
                            validated = self._validate_contextualized_text(result_str)
                            item.contextualized_text = validated or (item.text if isinstance(item, ExtractedChunk) else self._get_artifact_content(item))

            text_tasks = [process_text_batch(batch, i) for i, batch in enumerate(text_batches)]
            await asyncio.gather(*text_tasks, return_exceptions=True)

        # Process multimodal items (images) in batches
        if multimodal_items and not self._multimodal_disabled:
            multimodal_batches = self._create_batches(multimodal_items, self.config.batch_size)
            semaphore = asyncio.Semaphore(self.config.max_concurrency)

            async def process_multimodal_batch(batch, batch_idx):
                async with semaphore:
                    # Build message payloads for this batch
                    payloads = []
                    for item in batch:
                        text_before, text_after = self._find_surrounding_chunks(item.page, [])
                        payload = self._build_multimodal_payload(item, markdown, text_before, text_after)
                        payloads.append(payload)

                    # Execute batch
                    results = await asyncio.to_thread(
                        self._llm.batch_complete,
                        payloads
                    )

                    # Apply results back to items
                    for i, result_str in enumerate(results):
                        item = batch[i]
                        if isinstance(result_str, Exception):
                            logger.error(f"Failed to contextualize image {item.id}: {result_str}")
                            item.contextualized_text = item.caption
                        else:
                            validated = self._validate_contextualized_text(result_str)
                            item.contextualized_text = validated or item.caption

            multimodal_tasks = [process_multimodal_batch(batch, i) for i, batch in enumerate(multimodal_batches)]
            await asyncio.gather(*multimodal_tasks, return_exceptions=True)

    def _create_batches(self, items: list, batch_size: int) -> list[list]:
        """Split items into batches of specified size."""
        if not items:
            return []
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def _build_chunk_payload(self, chunk: ExtractedChunk, markdown: str | None) -> list[dict]:
        """Build the message payload for a text chunk."""
        if len(chunk.text) < self.config.skip_chunk_token_threshold and len(chunk.meta_data.get("headings", [])) >= 2:
            logger.info(
                "chunk_contextualization chunk_id=%s chunk_type=%s status=%s",
                chunk.id,
                "text_chunk",
                "skipped_short_chunk_with_headings",
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
                "content": f"Here is the chunk we want to situate within the whole document.\n<chunk>\n{chunk.text}\n</chunk>\n\nProvide a succinct context in 1 paragraph, no more than 5 sentences, describing this chunk's specific position and role within the document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else."
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

    def _build_multimodal_payload(self, image: ExtractedImage, document_markdown: str | None, text_before: str, text_after: str) -> list[dict]:
        """Build the message payload for an image."""
        # If we have a storage path (R2 URL), use it directly
        if image.storage_path:
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
                        "image_url": {"url": image.storage_path}
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
        """Validate contextualization output quality."""
        if not text or len(text.strip()) < 10:
            return ""
        if any(err in text.lower() for err in ["failed to", "unable to", "exception occurred"]):
            return ""
        return text.strip()
