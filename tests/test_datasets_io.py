import numpy as np
import pandas as pd
import pytest
import rasterio

from geoxplain.datasets import FEATURES, make_synthetic_geodata
from geoxplain.io import RasterStack, dataframe_to_dataset, read_raster_stack, write_raster
from geoxplain.utils import DataValidationError, SpatialAlignmentError


def test_synthetic_is_deterministic_and_labelled():
    a = make_synthetic_geodata(size=20, seed=1)
    b = make_synthetic_geodata(size=20, seed=1)
    c = make_synthetic_geodata(size=20, seed=2)
    assert np.array_equal(a.rasters["ndvi"], b.rasters["ndvi"])
    assert not np.array_equal(a.rasters["ndvi"], c.rasters["ndvi"])
    assert set(FEATURES) <= set(a.rasters)


def test_sample_points_columns(points):
    for col in ["id", "x", "y", *FEATURES, "target"]:
        assert col in points.columns
    assert set(points["target"].unique()) <= {0, 1}
    assert not points[FEATURES].isna().any().any()


def test_dataset_validation_errors(points):
    with pytest.raises(DataValidationError):
        dataframe_to_dataset(points, target="nope", features=FEATURES)
    with pytest.raises(DataValidationError):
        dataframe_to_dataset(points, target="target", features=["ndvi", "not_a_column"])


def test_dataset_drops_missing_rows(points):
    df = points.copy()
    df.loc[df.index[:5], "ndvi"] = np.nan
    ds = dataframe_to_dataset(df, target="target", features=FEATURES, missing="drop")
    assert len(ds.X) == len(df) - 5
    assert ds.summary()["rows_dropped"] == 5


def test_infer_task(points):
    assert dataframe_to_dataset(points, target="target", features=FEATURES).task == "classification"
    assert dataframe_to_dataset(points, target="susceptibility_index", features=FEATURES).task == "regression"


def test_raster_roundtrip_preserves_georeferencing(scene, tmp_path):
    stack = scene.stack
    path = write_raster(tmp_path / "s.tif", stack.data, stack, descriptions=list(stack.names))
    back = read_raster_stack(path)
    assert back.names == stack.names
    assert back.shape == stack.shape
    assert back.transform == stack.transform
    assert back.crs == stack.crs
    np.testing.assert_allclose(back.data, stack.data, rtol=1e-5, atol=1e-5)
    with rasterio.open(path) as src:
        assert src.crs == stack.crs
        assert src.nodata == stack.nodata


def test_nodata_is_masked(scene, tmp_path):
    stack = scene.stack
    data = stack.data.copy()
    data[0, :3, :3] = np.nan
    st = RasterStack(data, list(stack.names), stack.crs, stack.transform, stack.nodata)
    path = write_raster(tmp_path / "n.tif", st.data, st, descriptions=list(st.names))
    back = read_raster_stack(path)
    assert not back.valid_mask[:3, :3].any()
    assert back.valid_mask.sum() == stack.valid_mask.sum() - 9


def test_misaligned_rasters_raise(scene, tmp_path):
    stack = scene.stack
    a = write_raster(tmp_path / "a.tif", stack.data[0], stack)
    shifted = RasterStack(stack.data[:1], ["b"], stack.crs, stack.transform @ stack.transform.translation(2, 0), stack.nodata)
    b = write_raster(tmp_path / "b.tif", shifted.data[0], shifted)
    with pytest.raises(SpatialAlignmentError):
        read_raster_stack({"a": a, "b": b})


def test_stack_to_dataframe_and_scatter_are_inverse(scene):
    stack = scene.stack
    df = stack.to_dataframe()
    assert len(df) == stack.valid_mask.sum()
    back = stack.scatter(df["elevation"].to_numpy())
    mask = stack.valid_mask
    np.testing.assert_allclose(back[mask], stack.data[stack.names.index("elevation")][mask], rtol=1e-5)
    assert np.isnan(back[~mask]).all()


def test_points_lie_inside_raster(scene, points):
    left, bottom, right, top = scene.stack.bounds
    assert points["x"].between(left, right).all() and points["y"].between(bottom, top).all()
    assert isinstance(points, pd.DataFrame)
