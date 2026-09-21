import matplotlib.pyplot as plt
import numpy as np
import pytest
from affine import Affine

from geoxplain import visualization as viz
from geoxplain.explainers import GeoShapExplainer, partial_dependence
from geoxplain.io import RasterStack
from geoxplain.spatial import compute_spatial_shap


@pytest.fixture(scope="module")
def spatial(rf_model, scene, dataset):
    return compute_spatial_shap(rf_model, scene.stack, background=dataset.X.sample(30, random_state=0))


@pytest.fixture(scope="module")
def sv(rf_model, dataset):
    return GeoShapExplainer(rf_model, background=dataset.X.sample(30, random_state=0)).explain(dataset.X.iloc[:60])


def _nonempty(path):
    return path.exists() and path.stat().st_size > 1000


def test_save_figure_writes_all_formats(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    files = viz.save_figure(fig, tmp_path / "f.png", formats=("png", "pdf", "svg"))
    assert {p.suffix for p in files} == {".png", ".pdf", ".svg"} and all(_nonempty(p) for p in files)


def test_spatial_maps_are_written(spatial, tmp_path):
    figs = [
        viz.plot_shap_map(spatial, "elevation", path=tmp_path / "a"),
        viz.plot_shap_panels(spatial, path=tmp_path / "b"),
        viz.plot_prediction_uncertainty(spatial, path=tmp_path / "c"),
        viz.plot_feature_and_shap(spatial, "elevation", path=tmp_path / "d"),
        viz.plot_dominant_driver(spatial, path=tmp_path / "e"),
    ]
    plt.close("all")
    assert len(figs) == 5
    for stem in "abcde":
        assert _nonempty(tmp_path / f"{stem}.png")


def test_map_contains_required_cartographic_elements(spatial):
    fig = viz.plot_shap_map(spatial, "elevation")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "N" in texts                                  # north arrow
    assert any(t.endswith(" m") or t.endswith(" km") for t in texts)   # scale bar
    assert "Spatial contribution of elevation" in fig.axes[0].get_title(loc="left")
    assert fig.axes[0].get_xlabel().startswith("Easting") and len(fig.axes) == 2  # map + colour bar
    assert any("EPSG:32645" in t.get_text() for t in fig.texts)  # CRS note
    plt.close(fig)


def test_shap_map_colour_scale_is_symmetric_about_zero(spatial):
    fig = viz.plot_shap_map(spatial, "distance_to_water")
    im = fig.axes[0].images[0]
    assert im.norm.vmin == pytest.approx(-im.norm.vmax) and im.norm.vcenter == 0
    plt.close(fig)


def test_no_north_arrow_for_rotated_grid(spatial):
    st = spatial.stack
    rotated = RasterStack(st.data, list(st.names), st.crs, Affine.rotation(20) @ st.transform, st.nodata)
    assert not rotated.is_north_up
    spatial_rot = type(spatial)(**{**spatial.__dict__, "stack": rotated})
    fig = viz.plot_shap_map(spatial_rot, "elevation")
    assert "N" not in [t.get_text() for ax in fig.axes for t in ax.texts]
    plt.close(fig)


def test_scale_bar_for_geographic_crs(spatial):
    st = spatial.stack
    geo = RasterStack(st.data, list(st.names), "EPSG:4326", Affine(0.001, 0, 90.0, 0, -0.001, 24.0), st.nodata)
    fig, ax = plt.subplots()
    label = viz.add_scale_bar(ax, geo)
    assert label and "approximate" in label
    plt.close(fig)


def test_statistical_plots_are_written(sv, rf_model, dataset, tmp_path):
    from geoxplain.explainers import permutation_importance

    perm = permutation_importance(rf_model, dataset.X, dataset.y, n_repeats=2)
    pdps = [partial_dependence(rf_model, dataset.X, f, grid_resolution=8) for f in ("ndvi", "elevation")]
    makers = {
        "summary": lambda p: viz.plot_shap_summary(sv, path=p),
        "bar": lambda p: viz.plot_shap_bar(sv, path=p),
        "waterfall": lambda p: viz.plot_waterfall(sv, 0, path=p),
        "force": lambda p: viz.plot_force(sv, 0, path=p),
        "dependence": lambda p: viz.plot_shap_dependence(sv, "elevation", color_by="ndvi", path=p),
        "perm": lambda p: viz.plot_importance(perm, path=p),
        "pdp": lambda p: viz.plot_partial_dependence(pdps, path=p),
    }
    for name, fn in makers.items():
        fn(tmp_path / name)
        plt.close("all")
        assert _nonempty(tmp_path / f"{name}.png"), name


def test_comparison_plots(rf_model, dataset, tmp_path):
    import pandas as pd

    score = rf_model.predict_proba(dataset.X)[:, 1]
    viz.plot_roc_pr_curves({"random_forest": (dataset.y.to_numpy(), score)}, path=tmp_path / "roc")
    viz.plot_confusion_matrices({"random_forest": np.array([[5, 1], [2, 4]])}, [0, 1], path=tmp_path / "cm")
    table = pd.DataFrame({"roc_auc": [0.8, 0.7]}, index=["random_forest", "mlp"])
    viz.plot_metric_comparison(table, ["roc_auc"], path=tmp_path / "metrics")
    folds = pd.DataFrame({"model": ["a"] * 4, "strategy": ["random_kfold", "random_kfold", "spatial_block_cv", "spatial_block_cv"],
                          "roc_auc": [0.9, 0.88, 0.7, 0.75]})
    viz.plot_validation_comparison(folds, ["roc_auc"], path=tmp_path / "val")
    viz.plot_importance_comparison(pd.DataFrame({"a": [0.6, 0.4]}, index=["f1", "f2"]), path=tmp_path / "imp")
    plt.close("all")
    for n in ["roc", "cm", "metrics", "val", "imp"]:
        assert _nonempty(tmp_path / f"{n}.png"), n


def test_diagrams_and_logo(tmp_path):
    for fn, name in ((viz.draw_architecture, "arch"), (viz.draw_workflow, "wf")):
        files = fn(tmp_path / name, formats=("png", "svg"))
        assert all(_nonempty(p) for p in files)
    logo = viz.draw_logo(tmp_path / "logo", formats=("png", "svg"))
    assert all(_nonempty(p) for p in logo)
