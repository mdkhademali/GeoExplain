## Models

`GeoExplainModel(algorithm, task, params=None, random_state=42)` gives every algorithm the same API: `fit`, `predict`, `predict_proba` (classification), `evaluate`, `save`/`load`, `get_config`, `clone`.

| Key (aliases) | Classification | Regression | Scaling | SHAP method |
|---|---|---|---|---|
| `random_forest` (`rf`) | `RandomForestClassifier` | `RandomForestRegressor` | none | Tree (probability / target units) |
| `xgboost` (`xgb`) | `XGBClassifier` | `XGBRegressor` | none | Tree (log-odds / target units) |
| `lightgbm` (`lgbm`) | `LGBMClassifier` | `LGBMRegressor` | none | Tree (log-odds / target units) |
| `mlp` (`nn`) | `MLPClassifier` | `MLPRegressor` | StandardScaler | Permutation (probability / target units) |
| `logistic_regression` (`logreg`, `linear`) | `LogisticRegression` | `Ridge` (linear model) | StandardScaler | Linear (log-odds / target units) |
| `svm` | `SVC(probability=True)` | `SVR` | StandardScaler | Permutation |

Notes:

* Defaults are moderate and deterministic given `random_state`; pass `params={...}` to override (`geoxplain train --params '{"n_estimators": 300}'`). The exact defaults are in `geoxplain.models.default_params(algorithm, task)` and recorded in every experiment.
* For regression, the `logistic_regression` key resolves to a ridge regression (a linear model) so that the same model list can be used for both tasks.
* Classification supports binary and multi-class targets. Multi-class explanations require choosing a class (`class_label=`).
* `SVC(probability=True)` uses internal cross-validation to calibrate probabilities and is slower; it is provided for completeness.
* Saved models are joblib files: **only load files you trust**.

```python
from geoxplain import GeoExplainModel
m = GeoExplainModel("xgboost", "classification", params={"n_estimators": 300, "max_depth": 4}).fit(X, y)
m.save("model.joblib"); m2 = GeoExplainModel.load("model.joblib")
```

## Metrics

Classification: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC (average precision), confusion matrix. Binary metrics refer to the positive class (the second class in sorted order); multi-class metrics use macro averaging. If a fold contains a single class, ROC/PR-AUC are reported as `NaN` instead of failing.

Regression: MAE, MSE, RMSE, R², MAPE. MAPE is computed only over targets with `|y| > 1e-8` and is `NaN` when none exist (it is undefined for zero targets).

`metrics_table` and `export_metrics` produce a table and write CSV and Excel (`.xlsx`).
