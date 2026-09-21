"""Built-in and permutation feature importance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..metrics.classification import classification_metrics
from ..metrics.regression import regression_metrics
from ..models.model import GeoExplainModel
from ..utils.exceptions import DataValidationError, NotSupportedError

_LOWER_IS_BETTER = {"mae", "mse", "rmse", "mape_percent"}


def builtin_importance(model: GeoExplainModel) -> pd.DataFrame:
    """Model-native importance, normalised to sum to one.

    * Random Forest: mean decrease in impurity (biased towards high-cardinality features).
    * XGBoost / LightGBM: total gain of the splits on each feature.
    * Logistic/ridge regression: absolute coefficients of *standardised* features.
    * MLP / SVM: not available (use permutation importance or SHAP).
    """
    est = model.final_estimator_
    names = model.feature_names_
    if model.algorithm == "lightgbm":
        raw = np.asarray(est.booster_.feature_importance(importance_type="gain"), dtype=float)
        kind = "gain"
    elif model.algorithm == "xgboost":
        gain = est.get_booster().get_score(importance_type="gain")
        raw = np.array([gain.get(n, gain.get(f"f{i}", 0.0)) for i, n in enumerate(names)], dtype=float)
        kind = "gain"
    elif model.algorithm == "random_forest":
        raw, kind = np.asarray(est.feature_importances_, dtype=float), "impurity decrease"
    elif model.algorithm == "logistic_regression":
        coef = np.abs(np.asarray(est.coef_, dtype=float))
        raw = coef.mean(axis=0) if coef.ndim == 2 else coef
        kind = "|standardised coefficient|"
    else:
        raise NotSupportedError(f"Built-in importance is not available for {model.label}.")
    total = raw.sum()
    imp = raw / total if total > 0 else raw
    table = pd.DataFrame({"feature": names, "importance": imp, "kind": kind})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def _score(model: GeoExplainModel, X: pd.DataFrame, y: Sequence[Any], scoring: str) -> float:
    pred = model.predict(X)
    if model.task == "classification":
        metrics = classification_metrics(y, pred, model.predict_proba(X), classes=model.classes_)
    else:
        metrics = regression_metrics(y, pred)
    if scoring not in metrics:
        raise DataValidationError(f"Unknown scoring '{scoring}'. Choose from {sorted(metrics)}.")
    return metrics[scoring]


def permutation_importance(
    model: GeoExplainModel,
    X: pd.DataFrame,
    y: Sequence[Any],
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on held-out data.

    Each feature is shuffled ``n_repeats`` times and the resulting *drop in performance*
    is recorded. ``scoring`` is any metric key from :mod:`geoxplain.metrics`
    (default ``roc_auc`` for classification, ``r2`` for regression). Lower-is-better
    metrics (e.g. ``rmse``) are handled so that importance is always "how much worse".

    Caveats: permuting one of several correlated features can create unrealistic
    combinations and may under- or over-state importance; use held-out (ideally
    spatially held-out) data.
    """
    frame = model._check_X(X)
    y = np.asarray(y)
    scoring = scoring or ("roc_auc" if model.task == "classification" else "r2")
    sign = -1.0 if scoring in _LOWER_IS_BETTER else 1.0
    base = _score(model, frame, y, scoring)
    if not np.isfinite(base) and model.task == "classification":
        scoring, base = "accuracy", _score(model, frame, y, "accuracy")
        sign = 1.0
    rng = np.random.default_rng(random_state)
    rows = []
    for col in frame.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(sign * (base - _score(model, shuffled, y, scoring)))
        rows.append(
            {"feature": col, "importance_mean": float(np.mean(drops)), "importance_std": float(np.std(drops))}
        )
    table = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    table["scoring"] = scoring
    table["baseline_score"] = base
    return table
