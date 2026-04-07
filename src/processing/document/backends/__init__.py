try:
    from .docling_backend import DoclingBackend
except ImportError:
    DoclingBackend = None

try:
    from .chandra_backend import ChandraOCRBackend
except ImportError:
    ChandraOCRBackend = None

try:
    from .glm_backend import GLMOCRBackend
except ImportError:
    GLMOCRBackend = None

try:
    from .lighton_backend import LightOnOCRBackend
except ImportError:
    LightOnOCRBackend = None

try:
    from .marker_backend import MarkerBackend, MarkerConfig
except ImportError:
    MarkerBackend = None
    MarkerConfig = None

__all__ = [
    "DoclingBackend",
    "ChandraOCRBackend",
    "LightOnOCRBackend",
    "GLMOCRBackend",
    "MarkerBackend",
    "MarkerConfig",
]
