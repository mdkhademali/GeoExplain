"""Environment capture and seeding utilities for reproducibility records."""

from __future__ import annotations

import platform
import random
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "shap",
    "xgboost",
    "lightgbm",
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
    "joblib",
    "openpyxl",
)


def package_version(dist_name: str) -> str | None:
    """Return the installed version of a distribution or ``None`` if not installed."""
    try:
        from importlib import metadata

        return metadata.version(dist_name)
    except Exception:  # noqa: BLE001
        return None


def environment_info() -> dict[str, Any]:
    """Collect Python, OS and package versions for reproducibility records."""
    from .._version import __version__

    versions = {name: package_version(name) for name in _TRACKED_PACKAGES}
    return {
        "geoxplain": __version__,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def set_global_seed(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs (models also receive explicit seeds)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed for third-party code
