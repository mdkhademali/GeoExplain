"""Synthetic example datasets (clearly labelled as artificial)."""

from .synthetic import (
    FEATURE_INFO,
    FEATURE_LABELS,
    FEATURES,
    GENERATING_COEFFICIENTS,
    SyntheticGeoData,
    make_synthetic_geodata,
    sample_points,
    write_synthetic_dataset,
)

__all__ = [
    "FEATURES",
    "FEATURE_INFO",
    "FEATURE_LABELS",
    "GENERATING_COEFFICIENTS",
    "SyntheticGeoData",
    "make_synthetic_geodata",
    "sample_points",
    "write_synthetic_dataset",
]
