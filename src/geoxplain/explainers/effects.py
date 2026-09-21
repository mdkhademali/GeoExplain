"""Partial dependence / ICE curves computed directly from the fitted model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..models.model import GeoExplainModel
from ..utils.exceptions import DataValidationError


@dataclass
class PartialDependenceResult:
    """Partial dependence of the model output on one feature."""

    feature: str
    grid: np.ndarray
    average: np.ndarray
    ice: np.ndarray  # shape (n_samples, n_grid)
    output_label: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"feature_value": self.grid, "partial_dependence": self.average})


def partial_dependence(
    model: GeoExplainModel,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 30,
    percentiles: tuple[float, float] = (0.05, 0.95),
    class_label: Any = None,
    max_samples: int = 500,
    random_state: int = 42,
) -> PartialDependenceResult:
    """Average model response when ``feature`` is set to each grid value (Friedman, 2001).

    The output is the predicted probability of ``class_label`` (classification) or the
    predicted value (regression). Partial dependence assumes the feature can be varied
    independently of the others, which may be unrealistic for correlated predictors.
    """
    frame = model._check_X(X)
    if feature not in frame.columns:
        raise DataValidationError(f"Unknown feature '{feature}'")
    if len(frame) > max_samples:
        frame = frame.sample(max_samples, random_state=random_state).reset_index(drop=True)
    lo, hi = np.quantile(frame[feature], percentiles)
    grid = np.linspace(lo, hi, grid_resolution)
    k = model.class_index(class_label) if model.task == "classification" else None
    ice = np.empty((len(frame), len(grid)))
    for j, value in enumerate(grid):
        modified = frame.copy()
        modified[feature] = value
        out = model.predict_proba(modified)[:, k] if k is not None else model.predict(modified)
        ice[:, j] = out
    label = "predicted probability" if k is not None else "predicted value"
    return PartialDependenceResult(feature, grid, ice.mean(axis=0), ice, label)
