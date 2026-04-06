from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ObjectStoreConfig:
    root_path: Path = Path("data/objectstore")
    auto_create_dirs: bool = True


DEFAULT_OBJECT_STORE_CONFIG = ObjectStoreConfig()
