"""General-purpose utilities."""

from .env import environment_info, set_global_seed
from .exceptions import (
    DataValidationError,
    GeoExplainError,
    NotSupportedError,
    OptionalDependencyError,
    SpatialAlignmentError,
)
from .optional import is_available, require
from .serialization import read_json, to_jsonable, write_json

__all__ = [
    "DataValidationError",
    "GeoExplainError",
    "NotSupportedError",
    "OptionalDependencyError",
    "SpatialAlignmentError",
    "environment_info",
    "is_available",
    "read_json",
    "require",
    "set_global_seed",
    "to_jsonable",
    "write_json",
]
