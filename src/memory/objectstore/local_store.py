from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_OBJECT_STORE_CONFIG, ObjectStoreConfig
from .provider import ObjectStoreProvider


class LocalObjectStore(ObjectStoreProvider):
    """Local filesystem-backed object store."""

    def __init__(self, config: ObjectStoreConfig = DEFAULT_OBJECT_STORE_CONFIG) -> None:
        self._config = config
        if self._config.auto_create_dirs:
            self._config.root_path.mkdir(parents=True, exist_ok=True)

    def _resolve_key_path(self, key: str) -> Path:
        cleaned = key.strip().replace("\\", "/").lstrip("/")
        if not cleaned:
            raise ValueError("Object store key cannot be empty.")

        candidate = (self._config.root_path / cleaned).resolve()
        root = self._config.root_path.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Unsafe object store key path: {key}")
        return candidate

    def write(self, key: str, data: bytes) -> str:
        path = self._resolve_key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def read(self, key: str) -> bytes:
        path = self._resolve_key_path(key)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Object not found for key: {key}")
        return path.read_bytes()
