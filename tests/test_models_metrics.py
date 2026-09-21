import numpy as np
import pandas as pd
import pytest

from geoxplain.metrics import classification_metrics, confusion, export_metrics, metrics_table, regression_metrics
from geoxplain.models import ALIASES, GeoExplainModel, available_algorithms, resolve_algorithm
from geoxplain.utils import DataValidationError, NotSupportedError

FAST = {"random_forest": {"n_estimators": 20}, "xgboost": {"n_estimators": 20}, "lightgbm": {"n_estimators": 20},
        "mlp": {"max_iter": 200}, "logistic_regression": {}, "svm": {}}


@pytest.mark.parametrize("algo", list(FAST))
def test_classification_models_train_and_predict(algo, dataset):
    if not available_algorithms()[algo]:
        pytest.skip(f"{algo} not installed")
    m = GeoExplainModel(algo, "classification", params=FAST[algo]).fit(dataset.X, dataset.y)
    proba = m.predict_proba(dataset.X)
    assert proba.shape == (len(dataset.X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(m.predict(dataset.X))) <= {0, 1}
    res = m.evaluate(dataset.X, dataset.y)
    assert res["roc_auc"] > 0.6


@pytest.mark.parametrize("algo", list(FAST))
def test_regression_models_train_and_predict(algo, reg_dataset):
    if not available_algorithms()[algo]:
        pytest.skip(f"{algo} not installed")
    m = GeoExplainModel(algo, "regression", params=FAST[algo]).fit(reg_dataset.X, reg_dataset.y)
    pred = m.predict(reg_dataset.X)
    assert pred.shape == (len(reg_dataset.X),)
    assert m.evaluate(reg_dataset.X, reg_dataset.y)["r2"] > 0.2


def test_aliases_and_unknown_algorithm():
    assert resolve_algorithm("rf") == "random_forest"
    assert all(v in available_algorithms() for v in ALIASES.values())
    with pytest.raises(DataValidationError):
        resolve_algorithm("not-a-model")


def test_predict_before_fit_raises():
    with pytest.raises(DataValidationError, match="not fitted"):
        GeoExplainModel("random_forest").predict(pd.DataFrame({"a": [1.0]}))


def test_feature_order_is_enforced(rf_model, dataset):
    shuffled = dataset.X[list(reversed(dataset.X.columns))]
    np.testing.assert_allclose(rf_model.predict_proba(shuffled), rf_model.predict_proba(dataset.X))


def test_save_load_roundtrip(rf_model, dataset, tmp_path):
    path = rf_model.save(tmp_path / "m.joblib")
    back = GeoExplainModel.load(path)
    np.testing.assert_allclose(back.predict_proba(dataset.X), rf_model.predict_proba(dataset.X))
    assert back.feature_names_ == rf_model.feature_names_


def test_classification_metrics_match_hand_computation():
    y = np.array([0, 0, 1, 1, 1, 0])
    proba = np.array([[.9, .1], [.6, .4], [.3, .7], [.2, .8], [.55, .45], [.4, .6]])
    pred = proba.argmax(axis=1)
    m = classification_metrics(y, pred, proba, classes=[0, 1])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(8 / 9)
    cm = confusion(y, pred, [0, 1])
    assert cm.tolist() == [[2, 1], [1, 2]]


def test_regression_metrics_match_hand_computation():
    y, p = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 3.0])
    m = regression_metrics(y, p)
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["mse"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert m["r2"] == pytest.approx(1 - 2 / (14 / 3))
    assert m["mape_percent"] == pytest.approx(100 * np.mean([0, 0.5, 0.25]))


def test_mape_is_nan_when_targets_are_zero():
    m = regression_metrics(np.zeros(3), np.ones(3))
    assert np.isnan(m["mape_percent"])


def test_single_class_fold_gives_nan_auc_not_crash():
    m = classification_metrics([1, 1, 1], [1, 1, 0], np.array([[.1, .9], [.2, .8], [.6, .4]]), classes=[0, 1])
    assert np.isnan(m["roc_auc"])


def test_metric_tables_export_csv_and_excel(rf_model, dataset, tmp_path):
    table = metrics_table({"rf": rf_model.evaluate(dataset.X, dataset.y)})
    paths = export_metrics(table, tmp_path, stem="m")
    assert paths["csv"].exists() and paths["xlsx"].exists()
    back = pd.read_excel(paths["xlsx"], index_col=0)
    assert "roc_auc" in back.columns


def test_multiclass_classification(points):
    from geoxplain.datasets import FEATURES
    from geoxplain.io import dataframe_to_dataset

    ds = dataframe_to_dataset(points, target="landcover", features=FEATURES, task="classification")
    m = GeoExplainModel("random_forest", "classification", params={"n_estimators": 20}).fit(ds.X, ds.y)
    assert m.predict_proba(ds.X).shape[1] == len(m.classes_) > 2
    with pytest.raises(DataValidationError):
        m.class_index(None)
    with pytest.raises(NotSupportedError):
        GeoExplainModel("random_forest", "regression").class_index("a")
