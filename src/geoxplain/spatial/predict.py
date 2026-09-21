"""Raster prediction with fitted models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..io.raster import RasterStack
from ..models.model import GeoExplainModel
from ..utils.exceptions import DataValidationError


@dataclass
class RasterPrediction:
    """Prediction grids (NaN / -1 outside the valid area)."""

    prediction: np.ndarray  # probability of `class_label` (classification) or value (regression)
    predicted_class: np.ndarray | None  # int16 class index, -1 = nodata
    class_label: Any
    classes: list[str] | None


def model_stack(model: GeoExplainModel, stack: RasterStack) -> RasterStack:
    """Return ``stack`` restricted to the model's features, in training order."""
    missing = [n for n in model.feature_names_ if n not in stack.names]
    if missing:
        raise DataValidationError(
            f"Raster stack lacks bands required by the model: {missing}. Available: {stack.names}"
        )
    return stack.subset_bands(model.feature_names_)


def predict_raster(model: GeoExplainModel, stack: RasterStack, class_label: Any = None) -> RasterPrediction:
    """Predict every valid cell of ``stack`` and return grids aligned with it."""
    ms = model_stack(model, stack)
    table = ms.to_dataframe()
    feats = table[model.feature_names_]
    if model.task == "classification":
        k = model.class_index(class_label)
        proba = model.predict_proba(feats)
        pred = ms.scatter(proba[:, k])
        cls = ms.scatter(np.argmax(proba, axis=1), fill=-1, dtype=np.int16)
        return RasterPrediction(pred, cls, model.classes_[k], [str(c) for c in model.classes_])
    return RasterPrediction(ms.scatter(model.predict(feats)), None, None, None)
