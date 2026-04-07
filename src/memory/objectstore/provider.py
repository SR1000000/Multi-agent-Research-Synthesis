from __future__ import annotations

from abc import ABC, abstractmethod


class ObjectStoreProvider(ABC):
    """Interface for binary object storage providers."""

    @abstractmethod
    def write(self, key: str, data: bytes) -> str:
        """Write bytes to storage and return a provider-specific location."""
        pass

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read bytes from storage for the provided key."""
        pass
