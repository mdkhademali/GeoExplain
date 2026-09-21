"""Exception hierarchy used throughout GeoExplain."""

from __future__ import annotations


class GeoExplainError(Exception):
    """Base class for all GeoExplain-specific errors."""


class OptionalDependencyError(GeoExplainError, ImportError):
    """Raised when an optional third-party package is required but missing."""


class DataValidationError(GeoExplainError, ValueError):
    """Raised when input data fail validation (missing columns, NaN, bad types...)."""


class SpatialAlignmentError(GeoExplainError, ValueError):
    """Raised when rasters do not share the same CRS, transform or shape."""


class NotSupportedError(GeoExplainError, NotImplementedError):
    """Raised when a feature is not available for a given model or task."""
