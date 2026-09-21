"""Metric evaluation of fitted models and export of metric tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..io.tabular import save_table, save_tables_excel
from .classification import classification_metrics
from .regression import regression_metrics


def evaluate_model(model: Any, X: pd.DataFrame, y: Sequence[Any]) -> dict[str, float]:
    """Evaluate a fitted :class:`~geoxplain.models.GeoExplainModel` on ``(X, y)``."""
    pred = model.predict(X)
    if model.task == "classification":
        return classification_metrics(y, pred, model.predict_proba(X), classes=model.classes_)
    return regression_metrics(y, pred)


def metrics_table(results: Mapping[str, Mapping[str, float]], index_name: str = "model") -> pd.DataFrame:
    """Turn ``{model_name: {metric: value}}`` into a tidy DataFrame (one row per model)."""
    frame = pd.DataFrame.from_dict({k: dict(v) for k, v in results.items()}, orient="index")
    frame.index.name = index_name
    return frame


def export_metrics(
    table: pd.DataFrame, out_dir: str | Path, stem: str = "metrics", excel: bool = True
) -> dict[str, Path]:
    """Write ``table`` to ``<stem>.csv`` (and ``<stem>.xlsx``) in ``out_dir``."""
    out_dir = Path(out_dir)
    paths = {"csv": save_table(table, out_dir / f"{stem}.csv", index=True)}
    if excel:
        paths["xlsx"] = save_tables_excel({stem: table}, out_dir / f"{stem}.xlsx", index=True)
    return paths
