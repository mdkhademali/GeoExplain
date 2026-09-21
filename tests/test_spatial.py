import numpy as np
import pytest
import rasterio

from geoxplain.explainers import GeoShapExplainer
from geoxplain.io import RasterStack
from geoxplain.models import GeoExplainModel
from geoxplain.spatial import compute_spatial_shap, layer_stem, predict_raster, predictive_uncertainty
from geoxplain.utils import DataValidationError, NotSupportedError


@pytest.fixture(scope="module")
def result(rf_model, scene, dataset):
    return compute_spatial_shap(rf_model, scene.stack, background=dataset.X.sample(40, random_state=0),
                                feature_labels={"ndvi": "NDVI", "distance_to_water": "DistanceToWater"})


def test_layer_names_follow_convention():
    assert layer_stem("ndvi", {"ndvi": "NDVI"}) == "NDVI_SHAP"
    assert layer_stem("elevation") == "elevation_SHAP"


def test_spatial_shap_layers_are_grids_matching_input(result, scene):
    for name, layer in result.layers.items():
        assert layer.shape == scene.stack.shape, name
        assert np.isfinite(layer[scene.stack.valid_mask]).all()
    assert set(result.layers) == set(scene.stack.names)


def test_spatial_shap_is_additive_cellwise(result, scene):
    mask = scene.stack.valid_mask
    recon = result.shap.base_value + result.total
    np.testing.assert_allclose(recon[mask], result.prediction[mask], atol=1e-4)
    assert result.additivity["ok"]


def test_prediction_matches_predict_raster(result, rf_model, scene):
    pr = predict_raster(rf_model, scene.stack)
    np.testing.assert_allclose(pr.prediction[scene.stack.valid_mask], result.prediction[scene.stack.valid_mask], atol=1e-6)


def test_written_rasters_preserve_georeferencing(result, scene, tmp_path):
    written = result.write(tmp_path)
    for key in ["NDVI_SHAP", "elevation_SHAP", "prediction", "uncertainty", "dominant_driver"]:
        assert key in written, key
    for key, path in written.items():
        if not str(path).endswith(".tif"):
            continue
        with rasterio.open(path) as src:
            assert src.crs == scene.stack.crs, key
            assert src.transform == scene.stack.transform, key
            assert (src.height, src.width) == scene.stack.shape, key
            assert src.nodata is not None, key


def test_nodata_cells_are_propagated(rf_model, scene, dataset, tmp_path):
    data = scene.stack.data.copy()
    data[2, 5:9, 5:9] = np.nan
    st = RasterStack(data, list(scene.stack.names), scene.stack.crs, scene.stack.transform, scene.stack.nodata)
    res = compute_spatial_shap(rf_model, st, background=dataset.X.sample(30, random_state=0))
    assert np.isnan(res.layers["ndvi"][5:9, 5:9]).all()
    written = res.write(tmp_path)
    with rasterio.open(written["ndvi_SHAP"]) as src:
        arr = src.read(1)
        assert (arr[5:9, 5:9] == src.nodata).all()


def test_missing_feature_band_is_an_error(rf_model, scene):
    sub = scene.stack.subset_bands(["ndvi", "ndbi"])
    with pytest.raises(DataValidationError):
        compute_spatial_shap(rf_model, sub)


def test_uncertainty_support_matrix(dataset, reg_dataset):
    rf_c = GeoExplainModel("random_forest", "classification", params={"n_estimators": 15}).fit(dataset.X, dataset.y)
    unc = predictive_uncertainty(rf_c, dataset.X.iloc[:30])
    assert unc.method == "entropy" and (unc.values >= 0).all() and (unc.values <= 1.0001).all()
    rf_r = GeoExplainModel("random_forest", "regression", params={"n_estimators": 15}).fit(reg_dataset.X, reg_dataset.y)
    assert predictive_uncertainty(rf_r, reg_dataset.X.iloc[:30]).method == "tree_std"
    xgb = GeoExplainModel("xgboost", "regression", params={"n_estimators": 15}).fit(reg_dataset.X, reg_dataset.y)
    with pytest.raises(NotSupportedError):
        predictive_uncertainty(xgb, reg_dataset.X.iloc[:30])


def test_regression_spatial_shap_without_uncertainty_support(reg_dataset, scene):
    xgb = GeoExplainModel("xgboost", "regression", params={"n_estimators": 20}).fit(reg_dataset.X, reg_dataset.y)
    res = compute_spatial_shap(xgb, scene.stack, background=reg_dataset.X.sample(30, random_state=0))
    assert res.uncertainty is None
    assert any("ncertainty" in n for n in res.notes)


def test_points_and_raster_explanations_agree(rf_model, scene, dataset):
    bg = dataset.X.sample(40, random_state=0)
    res = compute_spatial_shap(rf_model, scene.stack, background=bg)
    df = scene.stack.to_dataframe()
    sv = GeoShapExplainer(rf_model, background=bg).explain(df[rf_model.feature_names_].iloc[:20])
    mask = scene.stack.valid_mask
    rows, cols = np.where(mask)
    for k in range(5):
        assert res.layers["elevation"][rows[k], cols[k]] == pytest.approx(sv.values[k, rf_model.feature_names_.index("elevation")], abs=1e-6)
