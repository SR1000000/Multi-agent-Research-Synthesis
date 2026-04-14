"""Utility for uploading base64 images to R2 and getting public URLs."""

import hashlib
import mimetypes
from base64 import b64decode
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.objectstore.provider import ObjectStoreProvider

class ImageUploader:
    """Upload base64 images to R2 and return public URLs."""

    def __init__(self, object_store: "ObjectStoreProvider"):
        self.object_store = object_store
        self._upload_cache: dict[str, str] = {}

    def get_or_upload_url(self, doc_id: str, image_id: str, base64_data: str) -> str:
        """
        Get R2 URL for image, uploading if necessary.

        Args:
            doc_id: Document identifier
            image_id: Image identifier
            base64_data: Base64 encoded image data

        Returns:
            Public URL for multimodal LLM input
        """
        # Create cache key from base64 data
        cache_key = hashlib.md5(base64_data.encode()).hexdigest()

        # Check cache first
        if cache_key in self._upload_cache:
            return self._upload_cache[cache_key]

        try:
            # Decode base64 data
            image_bytes = b64decode(base64_data)

            # Determine file extension from the data
            # First 10 bytes should contain magic number for image format
            ext = self._guess_image_extension(image_bytes[:10])
            if not ext:
                ext = "png"  # default fallback

            # Upload to R2 with proper key
            storage_key = f"{doc_id}/images/{image_id}.{ext}"
            public_url = self.object_store.write(storage_key, image_bytes)

            # If the object store is R2, generate a presigned URL to avoid 401 errors
            # Check if the object store has the generate_presigned_url method
            if hasattr(self.object_store, 'generate_presigned_url'):
                try:
                    presigned_url = self.object_store.generate_presigned_url(storage_key)
                    public_url = presigned_url
                except Exception as e:
                    # If generating presigned URL fails, continue with the original public URL
                    print(f"[ImageUploader] Failed to generate presigned URL for {image_id}: {e}")

            # Cache the result
            self._upload_cache[cache_key] = public_url
            return public_url

        except Exception as e:
            # Log error but don't fail the entire pipeline
            print(f"[ImageUploader] Failed to upload image {image_id}: {e}")
            # Return empty string to indicate failure
            return ""

    def _guess_image_extension(self, header_bytes: bytes) -> str | None:
        """Guess image extension from file header bytes."""
        # PNG signature
        if header_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            return "png"
        # JPEG signature
        elif header_bytes.startswith(b'\xff\xd8\xff'):
            return "jpg"
        # GIF signature
        elif header_bytes.startswith(b'GIF87a') or header_bytes.startswith(b'GIF89a'):
            return "gif"
        # WebP signature
        elif header_bytes.startswith(b'RIFF') and b'WEBP' in header_bytes:
            return "webp"
        # BMP signature
        elif header_bytes.startswith(b'BM'):
            return "bmp"
        return None