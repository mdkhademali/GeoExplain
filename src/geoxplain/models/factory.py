"""Algorithm registry and estimator factory.

All algorithms are exposed under short keys (``random_forest``, ``xgboost``, ``lightgbm``,
``mlp``, ``logistic_regression``/``linear``, ``svm``). Scale-sensitive models are wrapped in
a ``Pipeline`` with a ``StandardScaler`` so users can pass raw feature values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from ..utils.exceptions import DataValidationError
from ..utils.optional import is_available, require


@dataclass(frozen=True)
class AlgorithmSpec:
    """Static description of an algorithm."""

    key: str
    label: str
    family: str  # 'tree' | 'linear' | 'neural' | 'kernel'
    scaled: bool
    optional_dependency: str | None = None


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "random_forest": AlgorithmSpec("random_forest", "Random Forest", "tree", False),
    "xgboost": AlgorithmSpec("xgboost", "XGBoost", "tree", False, "xgboost"),
    "lightgbm": AlgorithmSpec("lightgbm", "LightGBM", "tree", False, "lightgbm"),
    "mlp": AlgorithmSpec("mlp", "MLP", "neural", True),
    "logistic_regression": AlgorithmSpec("logistic_regression", "Logistic Regression", "linear", True),
    "svm": AlgorithmSpec("svm", "SVM (RBF)", "kernel", True),
}

ALIASES = {
    "rf": "random_forest",
    "randomforest": "random_forest",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "nn": "mlp",
    "neural_network": "mlp",
    "logreg": "logistic_regression",
    "logistic": "logistic_regression",
    "linear": "logistic_regression",
    "linear_regression": "logistic_regression",
    "svc": "svm",
    "svr": "svm",
}


def resolve_algorithm(name: str) -> str:
    """Normalise an algorithm name/alias to a registry key."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in ALGORITHMS:
        raise DataValidationError(
            f"Unknown algorithm '{name}'. Choose from {sorted(ALGORITHMS)} "
            f"(aliases: {sorted(ALIASES)})."
        )
    return key


def available_algorithms() -> dict[str, bool]:
    """Return ``{algorithm: is_importable}`` for every registered algorithm."""
    out = {}
    for key, spec in ALGORITHMS.items():
        out[key] = True if spec.optional_dependency is None else is_available(spec.optional_dependency)
    return out


def default_params(algorithm: str, task: str) -> dict[str, Any]:
    """Default hyper-parameters (moderate, deterministic, small-data friendly)."""
    key = resolve_algorithm(algorithm)
    if key == "random_forest":
        return {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt", "max_depth": None}
    if key == "xgboost":
        return {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 3, "tree_method": "hist",
        }
    if key == "lightgbm":
        return {
            "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 20,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1,
        }
    if key == "mlp":
        return {
            "hidden_layer_sizes": (64, 32), "alpha": 1e-3, "max_iter": 800,
            "early_stopping": True, "n_iter_no_change": 20,
        }
    if key == "logistic_regression":
        return {"C": 1.0, "max_iter": 2000} if task == "classification" else {"alpha": 1.0}
    if key == "svm":
        return {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
    raise DataValidationError(key)  # pragma: no cover


def build_estimator(
    algorithm: str, task: str, params: dict[str, Any] | None = None, random_state: int = 42
) -> tuple[Any, dict[str, Any]]:
    """Instantiate an *unfitted* scikit-learn compatible estimator.

    Returns ``(estimator, resolved_params)`` where ``resolved_params`` are the defaults
    merged with the user's ``params``. Scale-sensitive algorithms are returned as
    ``Pipeline([("scaler", StandardScaler()), ("model", ...)])``.
    """
    key = resolve_algorithm(algorithm)
    if task not in ("classification", "regression"):
        raise DataValidationError(f"Unknown task '{task}'")
    merged = {**default_params(key, task), **(params or {})}
    clf = task == "classification"

    if key == "random_forest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        est = cls(random_state=random_state, **merged)
    elif key == "xgboost":
        xgb = require("xgboost", "the XGBoost model")
        cls = xgb.XGBClassifier if clf else xgb.XGBRegressor
        est = cls(random_state=random_state, **merged)
    elif key == "lightgbm":
        lgb = require("lightgbm", "the LightGBM model")
        cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
        est = cls(random_state=random_state, **merged)
    elif key == "mlp":
        cls = MLPClassifier if clf else MLPRegressor
        est = Pipeline([
            ("scaler", _scaler()),
            ("model", cls(random_state=random_state, **merged)),
        ])
    elif key == "logistic_regression":
        if clf:
            inner = LogisticRegression(random_state=random_state, **merged)
        else:
            inner = Ridge(random_state=random_state, **merged)
        est = Pipeline([("scaler", _scaler()), ("model", inner)])
    elif key == "svm":
        if clf:
            inner = SVC(probability=True, random_state=random_state, **merged)
        else:
            inner = SVR(**merged)
        est = Pipeline([("scaler", _scaler()), ("model", inner)])
    else:  # pragma: no cover
        raise DataValidationError(key)
    return est, merged


def _scaler() -> StandardScaler:
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler
