"""Model-based predictive uncertainty.

.. important::
   These are *model-internal* dispersion measures (tree disagreement or class-probability
   entropy). They are **not** calibrated confidence intervals and do not account for
   extrapolation, spatial dependence or data error. Interpret them as relative indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..models.model import GeoExplainModel
from ..utils.exceptions import DataValidationError, NotSupportedError


@dataclass
class UncertaintyResult:
    """Uncertainty values with a description of how they were computed."""

    values: np.ndarray
    method: str
    description: str


def predictive_uncertainty(
    model: GeoExplainModel, X: pd.DataFrame, method: str = "auto", class_label: Any = None
) -> UncertaintyResult:
    """Estimate per-row predictive uncertainty.

    Methods
    -------
    ``"entropy"``
        Classification (any model with ``predict_proba``): Shannon entropy of the
        predicted class distribution divided by ``log(K)`` -> range ``[0, 1]``.
    ``"tree_std"``
        Random Forest only: standard deviation across trees of the predicted value
        (regression) or of the predicted probability of ``class_label`` (classification).
    ``"auto"``
        Classification -> ``"entropy"``; regression -> ``"tree_std"`` for Random Forest.

    Raises
    ------
    NotSupportedError
        If no uncertainty estimate is available for the model/task combination
        (e.g. regression with XGBoost, LightGBM, MLP, ridge or SVM).
    """
    frame = model._check_X(X)
    if method == "auto":
        method = "entropy" if model.task == "classification" else "tree_std"
    if method == "entropy":
        if model.task != "classification":
            raise NotSupportedError("Entropy uncertainty requires a classifier.")
        proba = np.clip(model.predict_proba(frame), 1e-12, 1.0)
        ent = -(proba * np.log(proba)).sum(axis=1) / np.log(proba.shape[1])
        return UncertaintyResult(
            ent.astype(np.float32), "entropy",
            "Normalised Shannon entropy of the predicted class probabilities (0 = certain, 1 = uniform).",
        )
    if method == "tree_std":
        if model.algorithm != "random_forest":
            raise NotSupportedError(
                f"tree_std uncertainty is only available for Random Forest, not {model.label}. "
                "Use a Random Forest or a classifier (entropy)."
            )
        arr = frame.to_numpy(dtype=np.float32)
        trees = model.final_estimator_.estimators_
        if model.task == "classification":
            k = model.class_index(class_label)
            preds = np.stack([t.predict_proba(arr)[:, k] for t in trees])
            desc = "Standard deviation across trees of the predicted class probability."
        else:
            preds = np.stack([t.predict(arr) for t in trees])
            desc = "Standard deviation across trees of the predicted value (target units)."
        return UncertaintyResult(preds.std(axis=0).astype(np.float32), "tree_std", desc)
    raise DataValidationError(f"Unknown uncertainty method '{method}'")
