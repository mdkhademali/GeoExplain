import json

import numpy as np
import pandas as pd
import pytest
import rasterio

from geoxplain.cli import main
from geoxplain.datasets import FEATURES
from geoxplain.workflows import ExperimentConfig, build_report, run_experiment

FAST = {"random_forest": {"n_estimators": 25}, "xgboost": {"n_estimators": 25}, "mlp": {"max_iter": 120}}
FEATS = list(FEATURES)


def _config(data_dir, out, **kw):
    base = dict(data=str(data_dir / "samples.csv"), raster=str(data_dir / "stack.tif"), features=FEATS,
                models=["random_forest", "logistic_regression"], primary_model="random_forest", out_dir=str(out),
                n_splits=3, shap_samples=60, model_params=FAST, crs="EPSG:32645",
                dataset_label="Synthetic test data (not real observations)")
    base.update(kw)
    return ExperimentConfig(**base)


@pytest.fixture(scope="module")
def experiment(data_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("exp")
    idx = run_experiment(_config(data_dir, out))
    return out, idx


def test_experiment_writes_expected_artifacts(experiment):
    out, idx = experiment
    for key in ["holdout_metrics", "cv_summary", "validation_optimism", "permutation_importance", "shap_global",
                "shap_values_points", "spatial_shap_summary", "local_explanations"]:
        assert (out / idx["tables"][key]).exists(), key
    for key in ["model_performance", "validation_comparison", "shap_summary", "spatial_shap_panels",
                "spatial_prediction", "spatial_uncertainty_explanation", "roc_pr_curves", "feature_importance"]:
        assert (out / idx["figures"][key]).exists(), key
    assert (out / "tables" / "all_results.xlsx").exists()
    assert (out / "reproducibility.json").exists() and (out / "experiment.json").exists()
    assert idx["shap_additivity"]["ok"]
    with rasterio.open(out / idx["rasters"]["prediction"]) as src:
        assert src.crs.to_string() == "EPSG:32645"


def test_experiment_compares_random_and_spatial(experiment):
    out, idx = experiment
    ho = pd.read_csv(out / idx["tables"]["holdout_metrics"])
    assert set(ho["strategy"]) == {"random_split", "spatial_split"} and set(ho["model"]) == {"random_forest", "logistic_regression"}
    opt = pd.read_csv(out / idx["tables"]["validation_optimism"])
    assert {"random_kfold", "spatial_block_cv", "optimism"} <= set(opt.columns)
    assert np.isfinite(opt["random_kfold"]).all()


def test_reproducibility_record(experiment):
    out, idx = experiment
    repro = json.loads((out / "reproducibility.json").read_text())
    assert repro["seed"] == 42 and repro["environment"]["python"] and "scikit-learn" in repro["environment"]["packages"]
    assert repro["model_configs"]["random_forest"]["params"]["n_estimators"] == 25


def test_same_seed_gives_identical_metrics(data_dir, tmp_path, experiment):
    out0, idx0 = experiment
    idx1 = run_experiment(_config(data_dir, tmp_path / "again", raster=None))
    a = pd.read_csv(out0 / idx0["tables"]["holdout_metrics"])
    b = pd.read_csv(tmp_path / "again" / idx1["tables"]["holdout_metrics"])
    pd.testing.assert_frame_equal(a, b)


def test_report_contains_all_sections_and_is_self_contained(experiment):
    out, _ = experiment
    path = build_report(out)
    html = path.read_text()
    for section in ["Dataset", "Model configuration", "Validation strategy", "Metrics", "Feature importance",
                    "SHAP explanations", "Spatial explanation maps", "Reproducibility", "Model comparison"]:
        assert section in html, section
    assert "data:image/png;base64" in html and "not real observations" in html and "not causal" in html.lower() or "causal" in html


def test_regression_experiment(data_dir, tmp_path):
    cfg = _config(data_dir, tmp_path / "reg", target="susceptibility_index", task="regression", models=["random_forest", "lightgbm"],
                  model_params={"random_forest": {"n_estimators": 20}, "lightgbm": {"n_estimators": 20}})
    idx = run_experiment(cfg)
    assert idx["task"] == "regression"
    assert "r2" in pd.read_csv(tmp_path / "reg" / idx["tables"]["holdout_metrics"]).columns
    assert (tmp_path / "reg" / idx["rasters"]["prediction"]).exists()


def test_config_json_roundtrip(tmp_path):
    cfg = ExperimentConfig(data="d.csv", features=["a"], out_dir="o")
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg.to_dict()))
    assert ExperimentConfig.from_json(p) == cfg


# ------------------------------------------------------------------ CLI
def test_cli_full_chain(data_dir, tmp_path, capsys):
    csv, stack = str(data_dir / "samples.csv"), str(data_dir / "stack.tif")
    common = ["--data", csv, "--target", "target", "--features", *FEATS, "--crs", "EPSG:32645"]
    model = tmp_path / "m.joblib"
    assert main(["train", *common, "--algorithm", "random_forest", "--params", '{"n_estimators": 25}', "--out", str(model)]) == 0
    train_out = json.loads(capsys.readouterr().out)
    assert model.exists() and train_out["validation"]["strategy"] == "spatial_split"

    assert main(["evaluate", "--model", str(model), "--data", csv, "--target", "target", "--out-dir", str(tmp_path / "ev")]) == 0
    assert (tmp_path / "ev" / "metrics.csv").exists() and (tmp_path / "ev" / "metrics.xlsx").exists()
    capsys.readouterr()

    assert main(["explain", "--model", str(model), "--data", csv, "--target", "target", "--out-dir", str(tmp_path / "ex"),
                 "--max-samples", "60", "--local", "0", "1", "--repeats", "2"]) == 0
    assert (tmp_path / "ex" / "figures" / "shap_summary.png").exists() and (tmp_path / "ex" / "figures" / "local_waterfall_1.png").exists()
    capsys.readouterr()

    assert main(["spatial-explain", "--model", str(model), "--raster", stack, "--out-dir", str(tmp_path / "sp")]) == 0
    rasters = list((tmp_path / "sp" / "rasters").glob("*_SHAP.tif"))
    assert len(rasters) >= len(FEATS) and (tmp_path / "sp" / "rasters" / "prediction.tif").exists()
    assert (tmp_path / "sp" / "maps" / "spatial_shap_panels.png").exists()
    capsys.readouterr()

    assert main(["spatial-explain", "--model", str(model), "--points", csv, "--crs", "EPSG:32645", "--out-dir", str(tmp_path / "pt")]) == 0
    pts = pd.read_csv(tmp_path / "pt" / "spatial_shap_points.csv")
    assert {"x", "y"} <= set(pts.columns) and (tmp_path / "pt" / "spatial_shap_points.gpkg").exists()


def test_cli_compare_then_report(data_dir, tmp_path, capsys):
    out = tmp_path / "cmp"
    rc = main(["compare", "--data", str(data_dir / "samples.csv"), "--target", "target", "--features", *FEATS,
               "--models", "random_forest", "logistic_regression", "--out-dir", str(out), "--n-splits", "3",
               "--crs", "EPSG:32645", "--raster", str(data_dir / "stack.tif")])
    assert rc == 0 and (out / "report.html").exists()
    capsys.readouterr()
    assert main(["report", "--experiment-dir", str(out), "--out", str(tmp_path / "r2.html")]) == 0
    assert (tmp_path / "r2.html").exists()


def test_cli_generate_data_info_and_errors(tmp_path, capsys):
    assert main(["generate-data", "--out-dir", str(tmp_path / "d"), "--size", "20", "--n-samples", "100"]) == 0
    assert (tmp_path / "d" / "samples.csv").exists() and (tmp_path / "d" / "stack.tif").exists()
    capsys.readouterr()
    assert main(["info"]) == 0
    assert "algorithms_available" in capsys.readouterr().out
    assert main(["report", "--experiment-dir", str(tmp_path / "nothing")]) == 2
    assert main(["train", "--data", str(tmp_path / "missing.csv"), "--features", "a"]) == 2
    with pytest.raises(SystemExit):
        main(["--version"])
