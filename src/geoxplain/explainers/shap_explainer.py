"""SHAP-based feature attribution for :class:`~geoxplain.models.GeoExplainModel`.

Background
----------
For a prediction ``f(x)`` the Shapley value of feature ``i`` is its average marginal
contribution over all feature coalitions. SHAP values are *additive*::

    f(x) = phi_0 + sum_i phi_i(x)

where ``phi_0`` is the base value (the expected model output on the reference
distribution). SHAP values quantify how the **fitted model** uses each feature for a
prediction - they describe the model, not causal effects in the real world.

The explainer picks an appropriate algorithm automatically:

* tree ensembles (Random Forest, XGBoost, LightGBM) -> exact TreeSHAP,
* linear models (logistic/ridge) -> exact linear SHAP on the standardised features,
* other models (MLP, SVM) -> model-agnostic *exact* enumeration for <= 8 features,
  otherwise permutation SHAP, using a background sample.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..models.model import GeoExplainModel
from ..utils.exceptions import DataValidationError, NotSupportedError
from ..utils.optional import require


@dataclass
class ShapValues:
    """SHAP values for a set of observations.

    Attributes
    ----------
    values:
        Array ``(n_observations, n_features)`` of SHAP values.
    base_value:
        Expected model output ``phi_0`` (same output space as ``values``).
    data:
        Feature values that were explained (same row order as ``values``).
    feature_names:
        Column names.
    output_space:
        Unit of the explained output: ``"probability"``, ``"log-odds"`` or ``"value"``.
    class_label:
        Class being explained (classification) or ``None`` (regression).
    method:
        Algorithm used (``"tree"``, ``"linear"``, ``"exact"``, ``"permutation"``).
    """

    values: np.ndarray
    base_value: float
    data: pd.DataFrame
    feature_names: list[str]
    output_space: str
    class_label: Any = None
    method: str = "tree"
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def reconstructed_output(self) -> np.ndarray:
        """``base_value + sum(values)`` for each observation."""
        return self.base_value + self.values.sum(axis=1)

    def to_frame(self, include_data: bool = False) -> pd.DataFrame:
        """SHAP values as a DataFrame (optionally with the feature values, suffixed ``__value``)."""
        frame = pd.DataFrame(self.values, columns=self.feature_names)
        if include_data:
            frame = pd.concat([frame, self.data.add_suffix("__value").reset_index(drop=True)], axis=1)
        return frame

    def mean_abs(self) -> pd.DataFrame:
        """Global importance: mean absolute SHAP value per feature, ranked."""
        mabs = np.abs(self.values).mean(axis=0)
        table = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mabs})
        total = table["mean_abs_shap"].sum()
        table["share"] = table["mean_abs_shap"] / total if total > 0 else 0.0
        table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        table["rank"] = np.arange(1, len(table) + 1)
        return table

    def local(self, index: int) -> pd.DataFrame:
        """Feature values and SHAP values of observation ``index`` sorted by ``|SHAP|``."""
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} observations")
        table = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": self.data.iloc[index].to_numpy(),
                "shap": self.values[index],
            }
        )
        table["abs_shap"] = table["shap"].abs()
        return table.sort_values("abs_shap", ascending=False).drop(columns="abs_shap").reset_index(drop=True)

    def subset(self, index: np.ndarray | list[int]) -> "ShapValues":
        """Return the SHAP values of the observations at integer positions ``index``."""
        idx = np.asarray(index, dtype=int)
        return ShapValues(
            self.values[idx], self.base_value, self.data.iloc[idx].reset_index(drop=True),
            list(self.feature_names), self.output_space, self.class_label, self.method,
            dict(self.extra),
        )


def _to_float(base: Any) -> float:
   return float(np.ravel(np.asarray(base.replace('[', '').replace(']', '') if isinstance(base, str) else base, dtype=float))[0])


class GeoShapExplainer:
    """Compute SHAP values for any :class:`GeoExplainModel`.

    Parameters
    ----------
    model:
        A fitted model.
    background:
        Reference data (typically the training features). Required for linear and
        model-agnostic explainers, and for ``feature_perturbation='interventional'``.
    method:
        ``"auto"`` (default), ``"tree"``, ``"linear"``, ``"exact"`` or ``"permutation"``.
    class_label:
        Class to explain for classification. ``None`` selects the positive class of a
        binary problem; multi-class problems need an explicit label.
    max_background:
        Maximum number of background rows (random sub-sample, seeded).
    feature_perturbation:
        For tree models: ``"tree_path_dependent"`` (default; no background needed) or
        ``"interventional"`` (uses the background sample).
    max_evals:
        Evaluation budget per row for permutation SHAP.
    """

    def __init__(
        self,
        model: GeoExplainModel,
        background: pd.DataFrame | None = None,
        method: str = "auto",
        class_label: Any = None,
        max_background: int = 50,
        random_state: int = 42,
        feature_perturbation: str = "tree_path_dependent",
        max_evals: int | None = None,
    ) -> None:
        if not model.is_fitted:
            raise DataValidationError("The model must be fitted before it can be explained.")
        self.model = model
        self.shap = require("shap", "SHAP explanations")
        self.feature_names = list(model.feature_names_)
        self.class_label = class_label
        self.class_idx = model.class_index(class_label) if model.task == "classification" else None
        self.max_evals = max_evals
        self.feature_perturbation = feature_perturbation
        self.random_state = random_state

        if background is not None:
            bg = model._check_X(background)
            if len(bg) > max_background:
                bg = bg.sample(max_background, random_state=random_state).reset_index(drop=True)
        else:
            bg = None
        self.background = bg

        if method == "auto":
            method = {"tree": "tree", "linear": "linear"}.get(
                model.family, "exact" if len(self.feature_names) <= 8 else "permutation"
            )
        if method not in {"tree", "linear", "exact", "permutation"}:
            raise DataValidationError(f"Unknown SHAP method '{method}'")
        if method == "linear" and model.family != "linear":
            raise NotSupportedError("method='linear' is only valid for linear models.")
        if method == "tree" and model.family != "tree":
            raise NotSupportedError("method='tree' is only valid for tree ensembles.")
        self.method = method
        if method != "tree" and bg is None:
            raise DataValidationError(
                f"method='{method}' needs background data: pass background=X_train "
                "(a sample is drawn automatically)."
            )
        if method == "tree" and feature_perturbation == "interventional" and bg is None:
            raise DataValidationError("feature_perturbation='interventional' needs background data.")
        self._explainer = self._build()

    # ---------------------------------------------------------------- building
    @property
    def output_space(self) -> str:
        """Unit of the SHAP values: ``'value'``, ``'probability'`` or ``'log-odds'``."""
        m = self.model
        if m.task == "regression":
            return "value"
        if m.algorithm == "random_forest" and self.method == "tree":
            return "probability"
        if self.method in {"exact", "permutation"}:
            return "probability"
        return "log-odds"

    def _build(self) -> Any:
        shap, m = self.shap, self.model
        if self.method == "tree":
            kwargs: dict[str, Any] = {"model_output": "raw", "feature_perturbation": self.feature_perturbation}
            if self.feature_perturbation == "interventional":
                kwargs["data"] = self.background
            return shap.TreeExplainer(m.final_estimator_, **kwargs)
        if self.method == "linear":
            bg_scaled = m.transform(self.background).to_numpy()
            return shap.LinearExplainer(m.final_estimator_, shap.maskers.Independent(bg_scaled))
        names = self.feature_names

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(np.asarray(arr, dtype=float), columns=names)
            if m.task == "classification":
                return m.predict_proba(frame)[:, self.class_idx]
            return m.predict(frame)

        masker = shap.maskers.Independent(self.background.to_numpy(), max_samples=len(self.background))
        algorithm = "exact" if self.method == "exact" else "permutation"
        return shap.Explainer(predict_fn, masker, algorithm=algorithm, feature_names=names)

    # ---------------------------------------------------------------- computing
    def _raw_shap(self, frame: pd.DataFrame) -> tuple[np.ndarray, Any]:
        """Return ``(values, base)`` in the explainer's native shape."""
        if self.method == "tree":
            with warnings.catch_warnings():
                # SHAP warns that LightGBM binary output changed shape; we normalise shapes ourselves.
                warnings.filterwarnings("ignore", message="LightGBM binary classifier")
                values = self._explainer.shap_values(frame, check_additivity=False)
            return values, self._explainer.expected_value
        if self.method == "linear":
            scaled = self.model.transform(frame).to_numpy()
            return self._explainer.shap_values(scaled), self._explainer.expected_value
        kwargs = {}
        if self.method == "permutation" and self.max_evals:
            kwargs["max_evals"] = self.max_evals
        expl = self._explainer(frame.to_numpy(), silent=True, **kwargs)
        return expl.values, float(np.mean(expl.base_values))

    def _select(self, values: Any, base: Any) -> tuple[np.ndarray, float]:
        """Normalise SHAP output shapes to ``(n, f)`` for the requested class."""
        if isinstance(values, list):  # older SHAP: list of per-class arrays
            values = np.stack(values, axis=-1)
        values = np.asarray(values, dtype=float)
        base_arr = np.asarray(base, dtype=float)
        if values.ndim == 3:
            k = self.class_idx if self.class_idx is not None else 0
            return values[:, :, k], float(np.ravel(base_arr)[k if base_arr.size > 1 else 0])
        b = _to_float(base_arr)
        if self.model.task == "classification" and self.method in {"tree", "linear"} and self.class_idx == 0:
            return -values, -b  # single-output margin models explain the positive class
        return values, b

    def explain(self, X: pd.DataFrame, chunk_size: int = 5000) -> ShapValues:
        """Compute SHAP values for the rows of ``X`` (processed in chunks)."""
        frame = self.model._check_X(X)
        parts, base = [], 0.0
        for start in range(0, len(frame), chunk_size):
            chunk = frame.iloc[start : start + chunk_size]
            raw, raw_base = self._raw_shap(chunk)
            vals, base = self._select(raw, raw_base)
            parts.append(vals)
        values = np.vstack(parts) if parts else np.empty((0, len(self.feature_names)))
        return ShapValues(
            values=values, base_value=base, data=frame, feature_names=self.feature_names,
            output_space=self.output_space, class_label=(
                None if self.class_idx is None else self.model.classes_[self.class_idx]
            ),
            method=self.method,
        )

    # ------------------------------------------------------------- verification
    def model_output(self, X: pd.DataFrame) -> np.ndarray:
        """The model output that SHAP values add up to (see :attr:`output_space`)."""
        m = self.model
        frame = m._check_X(X)
        k = self.class_idx
        if m.task == "regression":
            return m.predict(frame)
        est = m.final_estimator_
        if self.method == "tree" and m.algorithm == "random_forest":
            return m.predict_proba(frame)[:, k]
        if self.method == "tree" and m.algorithm == "xgboost":
            raw = np.asarray(est.predict(frame, output_margin=True))
        elif self.method == "tree" and m.algorithm == "lightgbm":
            raw = np.asarray(est.predict(frame, raw_score=True))
        elif self.method == "linear":
            raw = np.asarray(est.decision_function(m.transform(frame)))
        else:
            return m.predict_proba(frame)[:, k]
        if raw.ndim == 2:
            return raw[:, k]
        return raw if k == 1 else -raw

    def check_additivity(self, shap_values: ShapValues, atol: float = 1e-3) -> dict[str, float | bool]:
        """Verify ``base_value + sum(values) == model output`` (local accuracy property)."""
        expected = self.model_output(shap_values.data)
        err = np.abs(shap_values.reconstructed_output - expected)
        return {
            "max_abs_error": float(err.max()) if len(err) else 0.0,
            "mean_abs_error": float(err.mean()) if len(err) else 0.0,
            "ok": bool(len(err) == 0 or err.max() <= atol),
        }
