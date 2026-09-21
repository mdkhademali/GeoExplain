import numpy as np
import pytest

from geoxplain.models import GeoExplainModel
from geoxplain.validation import (
    SpatialBlockKFold,
    compare_validation_strategies,
    default_block_size,
    empirical_semivariogram,
    make_spatial_blocks,
    morans_i,
    random_split,
    spatial_split,
)


def test_random_split_partitions_and_is_reproducible():
    tr, te = random_split(100, 0.3, random_state=1)
    assert len(tr) + len(te) == 100 and not set(tr) & set(te)
    tr2, te2 = random_split(100, 0.3, random_state=1)
    assert np.array_equal(te, te2)


def test_random_split_stratification_keeps_prevalence():
    y = np.array([0] * 80 + [1] * 20)
    _, te = random_split(100, 0.25, random_state=0, stratify=y)
    assert y[te].mean() == pytest.approx(0.2, abs=0.05)


def test_blocks_are_spatially_compact():
    coords = np.column_stack([np.repeat(np.arange(10), 10), np.tile(np.arange(10), 10)]).astype(float)
    blocks = make_spatial_blocks(coords, block_size=5)
    assert len(np.unique(blocks)) == 4
    for b in np.unique(blocks):
        pts = coords[blocks == b]
        assert np.ptp(pts[:, 0]) < 5 and np.ptp(pts[:, 1]) < 5


def test_spatial_split_keeps_blocks_intact(dataset):
    coords = dataset.require_coords()
    size = default_block_size(coords, 5)
    tr, te = spatial_split(coords, 0.25, size, random_state=3)
    blocks = make_spatial_blocks(coords, size)
    assert not set(tr) & set(te)
    assert not set(blocks[tr]) & set(blocks[te])
    assert 0.1 < len(te) / len(coords) < 0.5


def test_buffer_removes_training_points_near_test(dataset):
    coords = dataset.require_coords()
    size = default_block_size(coords, 5)
    buf = size / 4
    tr, te = spatial_split(coords, 0.25, size, random_state=3, buffer=buf)
    tr0, _ = spatial_split(coords, 0.25, size, random_state=3, buffer=0.0)
    assert len(tr) < len(tr0)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(coords[te]).query(coords[tr])
    assert d.min() >= buf


def test_spatial_block_kfold_partitions_and_separates_blocks(dataset):
    coords = dataset.require_coords()
    cv = SpatialBlockKFold(n_splits=4, random_state=0)
    seen = []
    blocks = make_spatial_blocks(coords, default_block_size(coords, 4))
    for tr, te in cv.split(dataset.X, coords=coords):
        assert not set(tr) & set(te)
        assert not set(blocks[tr]) & set(blocks[te])
        seen.extend(te.tolist())
    assert sorted(seen) == list(range(len(coords)))
    assert cv.get_n_splits() == 4


def test_compare_validation_strategies_output(dataset):
    m = GeoExplainModel("random_forest", "classification", params={"n_estimators": 20})
    res = compare_validation_strategies(m, dataset, n_splits=3)
    assert set(res.folds["strategy"]) == {"random_kfold", "spatial_block_cv"}
    assert len(res.folds) == 6
    assert ("roc_auc", "mean") in res.summary.columns
    opt = res.optimism()
    assert {"random_kfold", "spatial_block_cv", "difference"} <= set(opt.columns)


def test_morans_i_detects_spatial_structure(scene, points):
    smooth = morans_i(points["elevation"].to_numpy(), points[["x", "y"]].to_numpy(), permutations=49)
    rng = np.random.default_rng(0)
    noise = morans_i(rng.normal(size=len(points)), points[["x", "y"]].to_numpy(), permutations=49)
    assert smooth["I"] > 0.5 and smooth["p_value"] < 0.05
    assert abs(noise["I"]) < 0.15


def test_semivariogram_increases_with_distance_for_smooth_field(points):
    sv = empirical_semivariogram(points[["x", "y"]].to_numpy(), points["elevation"].to_numpy(), n_bins=8)
    assert sv["semivariance"].iloc[-1] > sv["semivariance"].iloc[0]
