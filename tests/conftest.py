"""Shared fixtures: a small synthetic scene so the whole suite runs in well under a few minutes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from geoxplain.datasets import FEATURES, make_synthetic_geodata, sample_points  # noqa: E402
from geoxplain.io.tabular import dataframe_to_dataset  # noqa: E402


@pytest.fixture(scope="session")
def scene():
    return make_synthetic_geodata(size=40, seed=7)


@pytest.fixture(scope="session")
def points(scene):
    return sample_points(scene, n_samples=500, seed=7)


@pytest.fixture(scope="session")
def dataset(points, scene):
    return dataframe_to_dataset(points, target="target", features=FEATURES, crs=scene.crs)


@pytest.fixture(scope="session")
def reg_dataset(points, scene):
    return dataframe_to_dataset(points, target="susceptibility_index", features=FEATURES, crs=scene.crs)


@pytest.fixture(scope="session")
def rf_model(dataset):
    from geoxplain.models import GeoExplainModel

    return GeoExplainModel("random_forest", "classification", params={"n_estimators": 40}).fit(dataset.X, dataset.y)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, scene):
    """Files on disk (CSV points + multi-band stack) for CLI and workflow tests."""
    from geoxplain.datasets import write_synthetic_dataset

    d = tmp_path_factory.mktemp("data")
    write_synthetic_dataset(d, size=30, n_samples=350, seed=3)
    return d
