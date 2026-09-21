"""Helpers shared by the example scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURES = ["ndvi", "ndbi", "lst", "elevation", "rainfall", "distance_to_water", "population"]


def require_data() -> None:
    """Exit with a helpful message when the synthetic dataset has not been generated yet."""
    if not (DATA / "samples.csv").exists():
        sys.exit("Synthetic data not found. Run:  geoxplain generate-data --out-dir data   (or: python reproduce.py)")


def out_dir(name: str) -> Path:
    path = ROOT / "outputs" / "examples" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
