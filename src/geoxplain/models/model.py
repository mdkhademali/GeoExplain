"""Unified model interface used by every other GeoExplain module."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from ..utils.exceptions import DataValidationError, NotSupportedError
from .factory import ALGORITHMS, build_estimator, resolve_algorithm


class GeoExplainModel:
    """Consistent wrapper around Random Forest, XGBoost, LightGBM, MLP, logistic/ridge and SVM.

    Parameters
    ----------
    algorithm:
        ``"random_forest"``, ``"xgboost"``, ``"lightgbm"``, ``"mlp"``,
        ``"logistic_regression"`` (Ridge for regression) or ``"svm"``. Common aliases
        such as ``"rf"`` or ``"lgbm"`` are accepted.
    task:
        ``"classification"`` or ``"regression"``.
    params:
        Hyper-parameters overriding the defaults (see :func:`geoxplain.models.default_params`).
    random_state:
        Seed passed to the underlying estimator.

    Notes
    -----
    * Inputs are DataFrames; columns are matched **by name** to the training columns.
    * Class labels of any type are supported; internally they are encoded ``0..K-1``.
    * Scale-sensitive models standardise features internally, so raw values can be passed.
    """

    def __init__(
        self,
        algorithm: str = "random_forest",
        task: str = "classification",
        params: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        if task not in ("classification", "regression"):
            raise DataValidationError("task must be 'classification' or 'regression'")
        self.algorithm = resolve_algorithm(algorithm)
        self.task = task
        self.user_params = dict(params or {})
        self.random_state = int(random_state)
        self.estimator_: Any = None
        self.resolved_params_: dict[str, Any] = {}
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray | None = None
        self._encoder: LabelEncoder | None = None

    # ---------------------------------------------------------------- metadata
    @property
    def spec(self):
        """The :class:`~geoxplain.models.factory.AlgorithmSpec` of this model."""
        return ALGORITHMS[self.algorithm]

    @property
    def label(self) -> str:
        """Human-readable algorithm name."""
        return self.spec.label

    @property
    def family(self) -> str:
        """``'tree'``, ``'linear'``, ``'neural'`` or ``'kernel'``."""
        return self.spec.family

    @property
    def is_fitted(self) -> bool:
        return self.estimator_ is not None

    @property
    def final_estimator_(self) -> Any:
        """The bare estimator (unwrapped from the scaling pipeline if present)."""
        self._check_fitted()
        if isinstance(self.estimator_, Pipeline):
            return self.estimator_.named_steps["model"]
        return self.estimator_

    @property
    def scaler_(self) -> Any | None:
        """Fitted ``StandardScaler`` for scale-sensitive models, else ``None``."""
        self._check_fitted()
        if isinstance(self.estimator_, Pipeline):
            return self.estimator_.named_steps["scaler"]
        return None

    def get_config(self) -> dict[str, Any]:
        """JSON-friendly description of the model configuration."""
        return {
            "algorithm": self.algorithm,
            "label": self.label,
            "family": self.family,
            "task": self.task,
            "random_state": self.random_state,
            "params": _jsonable_params(self.resolved_params_ or self.user_params),
            "feature_names": list(self.feature_names_),
            "classes": None if self.classes_ is None else [str(c) for c in self.classes_],
        }

    def clone(self) -> "GeoExplainModel":
        """Return an unfitted copy with identical configuration."""
        return GeoExplainModel(self.algorithm, self.task, dict(self.user_params), self.random_state)

    # --------------------------------------------------------------- validation
    def _check_fitted(self) -> None:
        if self.estimator_ is None:
            raise DataValidationError("Model is not fitted yet; call fit() first.")

    def _check_X(self, X: Any, fit: bool = False) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim != 2:
                raise DataValidationError("X must be two-dimensional")
            if fit:
                frame = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])
            else:
                if arr.shape[1] != len(self.feature_names_):
                    raise DataValidationError(
                        f"Expected {len(self.feature_names_)} columns, got {arr.shape[1]}"
                    )
                frame = pd.DataFrame(arr, columns=self.feature_names_)
        if not fit:
            missing = [c for c in self.feature_names_ if c not in frame.columns]
            if missing:
                raise DataValidationError(f"Missing feature columns: {missing}")
            frame = frame[self.feature_names_]
        frame = frame.astype(float)
        if not np.isfinite(frame.to_numpy()).all():
            raise DataValidationError("X contains NaN or infinite values.")
        return frame.reset_index(drop=True)

    # ------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y: Sequence[Any] | pd.Series) -> "GeoExplainModel":
        """Fit the model on features ``X`` and target ``y``."""
        frame = self._check_X(X, fit=True)
        target = np.asarray(y)
        if len(target) != len(frame):
            raise DataValidationError("X and y have different lengths")
        est, resolved = build_estimator(self.algorithm, self.task, self.user_params, self.random_state)
        if self.task == "classification":
            self._encoder = LabelEncoder().fit(target)
            self.classes_ = self._encoder.classes_
            if len(self.classes_) < 2:
                raise DataValidationError("Training target contains a single class.")
            y_fit = self._encoder.transform(target)
        else:
            self._encoder, self.classes_ = None, None
            y_fit = target.astype(float)
        est.fit(frame, y_fit)
        self.estimator_ = est
        self.resolved_params_ = resolved
        self.feature_names_ = list(frame.columns)
        return self

    # --------------------------------------------------------------- prediction
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (classification) or values (regression)."""
        self._check_fitted()
        frame = self._check_X(X)
        pred = self.estimator_.predict(frame)
        if self.task == "classification":
            return self._encoder.inverse_transform(np.asarray(pred).astype(int))
        return np.asarray(pred, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities, shape ``(n, n_classes)``, columns ordered as ``classes_``."""
        self._check_fitted()
        if self.task != "classification":
            raise NotSupportedError("predict_proba is only available for classification.")
        frame = self._check_X(X)
        return np.asarray(self.estimator_.predict_proba(frame), dtype=float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the model's internal feature scaling (identity for tree models)."""
        frame = self._check_X(X)
        scaler = self.scaler_
        return frame if scaler is None else scaler.transform(frame)

    # --------------------------------------------------------------- class utils
    def class_index(self, class_label: Any | None = None) -> int:
        """Encoded index (column of ``predict_proba``) of ``class_label``.

        For binary problems ``None`` selects the positive class (the second label in
        sorted order). For multi-class problems an explicit label is required.
        """
        if self.task != "classification":
            raise NotSupportedError("Class labels only exist for classification.")
        classes = list(self.classes_)
        if class_label is None:
            if len(classes) == 2:
                return 1
            raise DataValidationError(
                f"Multi-class model: choose a class via class_label (one of {classes})."
            )
        for i, c in enumerate(classes):
            if c == class_label or str(c) == str(class_label):
                return i
        raise DataValidationError(f"Unknown class {class_label!r}; known classes: {classes}")

    # --------------------------------------------------------------- evaluation
    def evaluate(self, X: pd.DataFrame, y: Sequence[Any]) -> dict[str, float]:
        """Compute the standard metric set for the model's task (see :mod:`geoxplain.metrics`)."""
        from ..metrics import evaluate_model

        return evaluate_model(self, X, y)

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        """Serialise the model with :mod:`joblib`.

        Only load model files from sources you trust: joblib/pickle files can execute
        arbitrary code when loaded.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GeoExplainModel":
        """Load a model saved with :meth:`save` (trusted files only)."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise DataValidationError(f"{path} does not contain a GeoExplainModel")
        return obj

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"GeoExplainModel(algorithm={self.algorithm!r}, task={self.task!r}, {state})"


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in params.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out
