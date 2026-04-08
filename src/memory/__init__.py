from __future__ import annotations

from .provider.provider import DatabaseProvider
from .objectstore import (
    DEFAULT_OBJECT_STORE_CONFIG,
    LocalObjectStore,
    ObjectStoreConfig,
    ObjectStoreProvider,
)
from .research.database import ResearchDatabase
from .research.config import StorageConfig, DEFAULT_CONFIG

def get_database() -> DatabaseProvider:
    """Helper factory to obtain the active database provider."""
    return ResearchDatabase()


def get_object_store() -> ObjectStoreProvider:
    """Helper factory to obtain the active object store provider."""
    return LocalObjectStore()


__all__ = [
    "DatabaseProvider",
    "ResearchDatabase",
    "StorageConfig",
    "DEFAULT_CONFIG",
    "ObjectStoreProvider",
    "ObjectStoreConfig",
    "DEFAULT_OBJECT_STORE_CONFIG",
    "LocalObjectStore",
    "get_database",
    "get_object_store",
]
