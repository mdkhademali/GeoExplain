"""Regression metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float], mape_eps: float = 1e-8) -> dict[str, float]:
    """MAE, MSE, RMSE, R^2 and MAPE.

    MAPE is reported in percent and is only computed when every ``|y_true|`` exceeds
    ``mape_eps`` (it is undefined for zero targets); otherwise it is ``NaN``.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(yt, yp))
    mape = float("nan")
    if np.all(np.abs(yt) > mape_eps):
        mape = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"),
        "mape_percent": mape,
    }
