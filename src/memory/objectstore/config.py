from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class ObjectStoreConfig:
    root_path: Path = Path("data/objectstore")
    auto_create_dirs: bool = True
    r2_bucket_name: str = ""


DEFAULT_OBJECT_STORE_CONFIG = ObjectStoreConfig()
