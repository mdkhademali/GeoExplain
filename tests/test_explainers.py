import numpy as np
import pandas as pd
import pytest

from geoxplain.explainers import GeoShapExplainer, builtin_importance, partial_dependence, permutation_importance
from geoxplain.models import GeoExplainModel, available_algorithms
from geoxplain.utils import DataValidationError, NotSupportedError

PARAMS = {"random_forest": {"n_estimators": 25}, "xgboost": {"n_estimators": 25}, "lightgbm": {"n_estimators": 25},
          "logistic_regression": {}, "mlp": {"max_iter": 150}, "svm": {}}


@pytest.mark.parametrize("algo", ["random_forest", "xgboost", "lightgbm", "logistic_regression", "mlp"])
def test_shap_is_additive_for_every_model_family(algo, dataset):
    if not available_algorithms()[algo]:
        pytest.skip(f"{algo} not installed")
    m = GeoExplainModel(algo, "classification", params=PARAMS[algo]).fit(dataset.X, dataset.y)
    ex = GeoShapExplainer(m, background=dataset.X.sample(30, random_state=0))
    sv = ex.explain(dataset.X.iloc[:25])
    assert sv.values.shape == (25, dataset.X.shape[1])
    chk = ex.check_additivity(sv)
    assert chk["ok"], chk
    assert sv.output_space in {"probability", "log-odds"}


@pytest.mark.parametrize("algo", ["random_forest", "xgboost", "lightgbm"])
def test_shap_regression_additivity(algo, reg_dataset):
    if not available_algorithms()[algo]:
        pytest.skip(f"{algo} not installed")
    m = GeoExplainModel(algo, "regression", params=PARAMS[algo]).fit(reg_dataset.X, reg_dataset.y)
    ex = GeoShapExplainer(m, background=reg_dataset.X.sample(30, random_state=0))
    sv = ex.explain(reg_dataset.X.iloc[:20])
    assert ex.check_additivity(sv)["ok"]


def test_shap_multiclass_needs_class_label(points):
    from geoxplain.datasets import FEATURES
    from geoxplain.io import dataframe_to_dataset

    ds = dataframe_to_dataset(points, target="landcover", features=FEATURES, task="classification")
    m = GeoExplainModel("random_forest", "classification", params=PARAMS["random_forest"]).fit(ds.X, ds.y)
    label = m.classes_[0]
    ex = GeoShapExplainer(m, class_label=label)
    sv = ex.explain(ds.X.iloc[:15])
    assert ex.check_additivity(sv)["ok"] and sv.class_label == label
    with pytest.raises(DataValidationError):
        GeoShapExplainer(m)


def test_shap_result_helpers(rf_model, dataset):
    sv = GeoShapExplainer(rf_model, background=dataset.X.sample(30, random_state=0)).explain(dataset.X.iloc[:40])
    ma = sv.mean_abs()
    assert ma["share"].sum() == pytest.approx(1.0)
    assert ma["mean_abs_shap"].is_monotonic_decreasing
    loc = sv.local(3)
    assert set(loc.columns) >= {"feature", "value", "shap"}
    assert loc["shap"].abs().is_monotonic_decreasing
    assert sv.subset([0, 1, 2]).values.shape[0] == 3
    frame = sv.to_frame(include_data=True)
    assert len(frame) == 40


def test_shap_top_features_reflect_known_generating_process(rf_model, dataset):
    sv = GeoShapExplainer(rf_model, background=dataset.X.sample(50, random_state=0)).explain(dataset.X)
    top3 = sv.mean_abs()["feature"].head(3).tolist()
    assert {"elevation", "distance_to_water"} & set(top3)


def test_permutation_importance(rf_model, dataset):
    imp = permutation_importance(rf_model, dataset.X, dataset.y, n_repeats=3)
    assert set(imp.columns) >= {"feature", "importance_mean", "importance_std", "scoring"}
    assert imp["importance_mean"].iloc[0] >= imp["importance_mean"].iloc[-1]
    assert imp.iloc[0]["feature"] in {"elevation", "distance_to_water", "rainfall"}


def test_builtin_importance_support_matrix(rf_model, dataset):
    imp = builtin_importance(rf_model)
    assert imp["importance"].sum() == pytest.approx(1.0)
    mlp = GeoExplainModel("mlp", "classification", params={"max_iter": 100}).fit(dataset.X, dataset.y)
    with pytest.raises(NotSupportedError):
        builtin_importance(mlp)


def test_partial_dependence(rf_model, dataset):
    pdp = partial_dependence(rf_model, dataset.X, "elevation", grid_resolution=12)
    assert pdp.grid.shape == (12,) and pdp.average.shape == (12,) and pdp.ice.shape[1] == 12
    assert np.all((pdp.average >= 0) & (pdp.average <= 1))
    assert isinstance(pdp.to_frame(), pd.DataFrame)
