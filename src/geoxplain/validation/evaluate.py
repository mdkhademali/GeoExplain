"""Compare validation strategies (random vs. spatial) with the same model."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ..io.tabular import GeoDataset
from ..models.model import GeoExplainModel
from ..utils.exceptions import DataValidationError, GeoExplainError
from .splits import SpatialBlockKFold, random_split, spatial_split

STRATEGIES = ("random_split", "spatial_split", "random_kfold", "spatial_block_cv")
STRATEGY_LABELS = {
    "random_split": "Random hold-out",
    "spatial_split": "Spatial block hold-out",
    "random_kfold": "Random k-fold CV",
    "spatial_block_cv": "Spatial block k-fold CV",
}
HIGHER_IS_BETTER = {
    "accuracy": True, "balanced_accuracy": True, "precision": True, "recall": True, "f1": True,
    "roc_auc": True, "pr_auc": True, "r2": True,
    "mae": False, "mse": False, "rmse": False, "mape_percent": False,
}


@dataclass
class ValidationResult:
    """Fold-level metrics and their summary for one or more strategies."""

    folds: pd.DataFrame
    summary: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)

    def optimism(self, reference: str = "spatial_block_cv", optimistic: str = "random_kfold") -> pd.DataFrame:
        """Difference in mean score between two strategies for every metric.

        ``difference = mean(optimistic) - mean(reference)``. For higher-is-better metrics a
        positive value means the ``optimistic`` strategy reports better performance.
        """
        means = self.summary.xs("mean", axis=1, level=1)
        if reference not in means.index or optimistic not in means.index:
            raise DataValidationError("Both strategies must be present in the result.")
        out = pd.DataFrame(
            {
                optimistic: means.loc[optimistic],
                reference: means.loc[reference],
            }
        )
        out["difference"] = out[optimistic] - out[reference]
        out.index.name = "metric"
        return out


def iter_validation_splits(
    dataset: GeoDataset,
    strategy: str,
    n_splits: int = 5,
    test_size: float = 0.25,
    block_size: float | None = None,
    buffer: float = 0.0,
    seed: int = 42,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold, train_idx, test_idx)`` for a validation strategy."""
    if strategy not in STRATEGIES:
        raise DataValidationError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")
    n = len(dataset)
    stratify = dataset.y.to_numpy() if dataset.task == "classification" else None
    if strategy == "random_split":
        yield 0, *random_split(n, test_size, seed, stratify)
    elif strategy == "spatial_split":
        yield 0, *spatial_split(dataset.require_coords(), test_size, block_size, seed, buffer)
    elif strategy == "random_kfold":
        y = dataset.y.to_numpy()
        use_strat = dataset.task == "classification" and pd.Series(y).value_counts().min() >= n_splits
        cv = (StratifiedKFold if use_strat else KFold)(n_splits=n_splits, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(cv.split(np.zeros(n), y if use_strat else None)):
            yield f, tr, te
    else:
        cv = SpatialBlockKFold(n_splits, block_size, True, seed, buffer)
        for f, (tr, te) in enumerate(cv.split(coords=dataset.require_coords())):
            yield f, tr, te


def cross_validate(
    model: GeoExplainModel,
    dataset: GeoDataset,
    strategy: str,
    n_splits: int = 5,
    test_size: float = 0.25,
    block_size: float | None = None,
    buffer: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit a fresh clone of ``model`` on every split and return per-fold test metrics."""
    rows = []
    for fold, tr, te in iter_validation_splits(dataset, strategy, n_splits, test_size, block_size, buffer, seed):
        train, test = dataset.subset(tr), dataset.subset(te)
        row: dict[str, Any] = {"strategy": strategy, "fold": fold, "n_train": len(tr), "n_test": len(te)}
        try:
            fitted = model.clone().fit(train.X, train.y)
            row.update(fitted.evaluate(test.X, test.y))
        except GeoExplainError as exc:
            warnings.warn(f"{strategy} fold {fold} failed: {exc}", stacklevel=2)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_validation_strategies(
    model: GeoExplainModel,
    dataset: GeoDataset,
    strategies: tuple[str, ...] = ("random_kfold", "spatial_block_cv"),
    n_splits: int = 5,
    test_size: float = 0.25,
    block_size: float | None = None,
    buffer: float = 0.0,
    seed: int = 42,
) -> ValidationResult:
    """Evaluate ``model`` under several validation strategies.

    The returned ``summary`` has a row per strategy and a two-level column index
    ``(metric, mean|std)``. Nothing here is pre-computed or hard-coded: every number comes
    from fitting and scoring the model on the actual splits.
    """
    frames = [
        cross_validate(model, dataset, s, n_splits, test_size, block_size, buffer, seed)
        for s in strategies
    ]
    folds = pd.concat(frames, ignore_index=True)
    metric_cols = [c for c in folds.columns if c not in {"strategy", "fold", "n_train", "n_test"}]
    grouped = folds.groupby("strategy", sort=False)[metric_cols].agg(["mean", "std"])
    summary = grouped.loc[list(strategies)]
    config = {
        "strategies": list(strategies), "n_splits": n_splits, "test_size": test_size,
        "block_size": block_size, "buffer": buffer, "seed": seed, "algorithm": model.algorithm,
    }
    return ValidationResult(folds=folds, summary=summary, config=config)
