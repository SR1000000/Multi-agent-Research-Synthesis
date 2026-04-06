from __future__ import annotations

from .provider.provider import DatabaseProvider
from .objectstore import (
    DEFAULT_OBJECT_STORE_CONFIG,
    LocalObjectStore,
    ObjectStoreConfig,
    ObjectStoreProvider,
)
from .sqlite.database import SQLiteDatabase
from .sqlite.config import StorageConfig, DEFAULT_CONFIG

def get_database() -> DatabaseProvider:
    """Helper factory to obtain the active database provider."""
    return SQLiteDatabase()


def get_object_store() -> ObjectStoreProvider:
    """Helper factory to obtain the active object store provider."""
    return LocalObjectStore()


__all__ = [
    "DatabaseProvider",
    "SQLiteDatabase",
    "StorageConfig",
    "DEFAULT_CONFIG",
    "ObjectStoreProvider",
    "ObjectStoreConfig",
    "DEFAULT_OBJECT_STORE_CONFIG",
    "LocalObjectStore",
    "get_database",
    "get_object_store",
]
