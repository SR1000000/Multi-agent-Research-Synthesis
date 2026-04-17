from __future__ import annotations

from .provider.provider import DatabaseProvider
from .research.database import ResearchDatabase
from .research.config import StorageConfig, DEFAULT_CONFIG

def get_database() -> ResearchDatabase:
    """Helper factory to obtain the active database provider."""
    return ResearchDatabase()


__all__ = [
    "DatabaseProvider",
    "ResearchDatabase",
    "StorageConfig",
    "DEFAULT_CONFIG",
    "get_database",
]
