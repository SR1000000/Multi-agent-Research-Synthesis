from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional
import requests
from io import BytesIO
from PIL import Image
import litellm

from src.retriever import Retriever
from src.memory.research.database import ResearchDatabase
from src.memory.objectstore.image_uploader import ImageUploader
from src.memory.objectstore.r2_store import R2ObjectStore
from src.memory.objectstore.config import ObjectStoreConfig

DEFAULT_CONTEXT_K = 8


class RetrievalContextTool:
    def __init__(
        self,
        retriever: Retriever,
        research_db: ResearchDatabase,
        default_k: int = DEFAULT_CONTEXT_K,
    ) -> None:
        self._retriever = retriever
        self._research_db = research_db
        self._default_k = default_k
        # Initialize image uploader if R2 is configured
        try:
            self._object_store = R2ObjectStore(ObjectStoreConfig())
            self._image_uploader = ImageUploader(self._object_store)
        except ValueError:
            # If R2 is not configured, set to None
            self._object_store = None
            self._image_uploader = None

    def build_context(
        self,
        query: str,
        k: int | None = None,
        *,
        session_id: str | None = None,
        agent_type: str = "retrieval_context",
    ) -> str:
        n = k if k is not None else self._default_k
        items = self._retriever.fusion_retrieve(query, n)
        if not items:
            return ""

        if session_id:
            for it in items:
                self._research_db.save_retrieved_chunk(it, session_id, agent_type, query)

        parts: list[str] = []
        for it in items:
            if it.kind == "chunk":
                parts.append(
                    f"### chunk id={it.id} document_id={it.document_id}\n{it.text}\n"
                )
            elif it.kind == "table":
                parts.append(
                    f"### table id={it.id} document_id={it.document_id}\n{it.text}\n"
                )
            elif it.kind == "equation":
                parts.append(
                    f"### equation id={it.id} document_id={it.document_id}\nLaTeX: {it.text}\n"
                )
            elif it.kind == "image":
                # For images, we need to fetch the image data from the database and potentially embed it
                image_content = self._handle_image_retrieval(it.id, it.document_id, it.text)
                parts.append(
                    f"### image id={it.id} document_id={it.document_id}\n{image_content}\n"
                )
        return "\n".join(parts).strip()

    def _check_vision_support(self) -> bool:
        """Placeholder to check if the current LLM model supports vision.
        In a real implementation, this would check the active model's capabilities.
        """
        # This is a placeholder. In a real scenario, you would check the model configuration
        # or capabilities provided by the LLM API client.
        # For now, we assume vision is supported if litellm is configured.
        # A simple check: if the model name contains known vision-capable models.
        # We would need access to the current model name, which we don't have here.
        # For the purpose of this task, we'll return True to enable the logic.
        return True

    def _handle_image_retrieval(self, image_id: str, document_id: str, caption_or_path: str) -> str:
        """
        Handle image retrieval by fetching image data from DB and preparing it for LLM context.
        """
        # Fetch the image data from the research database
        image_data = self._research_db.get_image(image_id)

        if not image_data:
            return f"Image not found: {image_id}. Caption: {caption_or_path}"

        # Check if the current model supports vision
        supports_vision = self._check_vision_support()

        if supports_vision and self._image_uploader and image_data.base64_data:
            try:
                public_url = self._image_uploader.get_or_upload_url(document_id, image_id, image_data.base64_data)
                # For multimodal models, embed the URL
                return f"Image URL: {public_url}\nCaption: {image_data.caption}"
            except Exception as e:
                return f"Failed to upload image {image_id}: {str(e)}\nCaption: {image_data.caption}"
        else:
            # If vision not supported or image upload failed, return caption and path
            return f"Image ID: {image_id}\nCaption: {image_data.caption}\nStorage Path: {image_data.storage_path}"
