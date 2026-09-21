"""Command-line interface (``geoxplain``).

Sub-commands: ``generate-data``, ``train``, ``evaluate``, ``explain``, ``spatial-explain``,
``compare``, ``run``, ``report`` and ``info``. Run ``geoxplain <command> --help`` for details.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__
from .utils.exceptions import GeoExplainError

log = logging.getLogger("geoxplain")


# --------------------------------------------------------------------------- helpers
def _add_data_args(p: argparse.ArgumentParser, need_features: bool = True) -> None:
    p.add_argument("--data", required=True, help="CSV file with one row per observation.")
    p.add_argument("--target", default="target", help="Target column (default: target).")
    p.add_argument("--features", nargs="+", required=need_features, help="Feature columns (space separated).")
    p.add_argument("--x", default="x", help="X / longitude column (default: x).")
    p.add_argument("--y", default="y", help="Y / latitude column (default: y).")
    p.add_argument("--id-col", default="id", help="Identifier column (default: id; ignored if absent).")
    p.add_argument("--crs", default=None, help="CRS of the x/y columns, e.g. EPSG:32645.")
    p.add_argument("--task", choices=["classification", "regression"], default=None, help="Inferred if omitted.")


def _load(args: argparse.Namespace, features: Sequence[str] | None = None):
    from .io.tabular import load_dataset, load_table

    feats = features if features is not None else args.features
    id_col = args.id_col if args.id_col in load_table(args.data).columns else None
    return load_dataset(args.data, args.target, feats, x=args.x, y=args.y, id_col=id_col, task=args.task, crs=args.crs)


def _print_json(obj: Any) -> None:
    from .utils.serialization import to_jsonable

    print(json.dumps(to_jsonable(obj), indent=2))


def _formats(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(getattr(args, "formats", None) or ("png",))


# --------------------------------------------------------------------------- commands
def cmd_generate_data(args: argparse.Namespace) -> int:
    from .datasets import write_synthetic_dataset

    info = write_synthetic_dataset(args.out_dir, size=args.size, n_samples=args.n_samples, seed=args.seed,
                                   pixel_size=args.pixel_size)
    _print_json({k: v for k, v in info.items() if k != "data"})
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .metrics import evaluate_model
    from .models import GeoExplainModel
    from .validation import random_split, spatial_split

    ds = _load(args)
    params = json.loads(args.params) if args.params else None
    model = GeoExplainModel(args.algorithm, ds.task, params=params, random_state=args.seed)
    report: dict[str, Any] = {"algorithm": model.label, "task": ds.task, "n_observations": len(ds.X)}
    if args.validation != "none":
        if args.validation == "spatial_split":
            tr, te = spatial_split(ds.require_coords(), args.test_size, args.block_size, args.seed, args.buffer)
        else:
            tr, te = random_split(len(ds.X), args.test_size, args.seed,
                                  stratify=ds.y.to_numpy() if ds.task == "classification" else None)
        probe = GeoExplainModel(args.algorithm, ds.task, params=params, random_state=args.seed).fit(ds.X.iloc[tr], ds.y.iloc[tr])
        report["validation"] = {"strategy": args.validation, "n_train": int(len(tr)), "n_test": int(len(te)),
                                "metrics": evaluate_model(probe, ds.X.iloc[te], ds.y.iloc[te])}
    model.fit(ds.X, ds.y)
    path = model.save(args.out)
    report["model_file"] = str(path)
    report["note"] = "The saved model is refit on ALL observations; validation metrics come from the hold-out probe model."
    _print_json(report)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .metrics import confusion, evaluate_model, export_metrics, metrics_table
    from .models import GeoExplainModel

    model = GeoExplainModel.load(args.model)
    args.task = model.task
    ds = _load(args, features=model.feature_names_)
    res = evaluate_model(model, ds.X, ds.y)
    table = metrics_table({model.label: res})
    paths = export_metrics(table, args.out_dir, stem="metrics")
    if model.task == "classification":
        import pandas as pd

        labels = list(model.classes_)
        cm = pd.DataFrame(confusion(ds.y, model.predict(ds.X), labels), index=[f"true_{c}" for c in labels],
                          columns=[f"pred_{c}" for c in labels])
        cm.to_csv(Path(args.out_dir) / "confusion_matrix.csv")
    _print_json({"metrics": res, "files": {k: str(v) for k, v in paths.items()}})
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt
    import numpy as np

    from . import visualization as viz
    from .explainers import GeoShapExplainer, partial_dependence, permutation_importance
    from .io.tabular import save_table
    from .models import GeoExplainModel

    model = GeoExplainModel.load(args.model)
    args.task = model.task
    ds = _load(args, features=model.feature_names_)
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    fmts = _formats(args)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(ds.X), size=min(args.max_samples, len(ds.X)), replace=False))
    explainer = GeoShapExplainer(model, background=ds.X.sample(min(100, len(ds.X)), random_state=args.seed),
                                 class_label=args.class_label, random_state=args.seed)
    sv = explainer.explain(ds.X.iloc[idx])
    files: dict[str, str] = {}
    save_table(sv.mean_abs(), out / "tables" / "shap_global.csv")
    save_table(sv.to_frame(include_data=True), out / "tables" / "shap_values.csv")
    for name, fn in (("shap_summary", viz.plot_shap_summary), ("shap_bar", viz.plot_shap_bar)):
        viz.save_figure(fn(sv), out / "figures" / name, formats=fmts)
        files[name] = str(out / "figures" / f"{name}.png")
    perm = permutation_importance(model, ds.X, ds.y, n_repeats=args.repeats, random_state=args.seed)
    save_table(perm, out / "tables" / "permutation_importance.csv")
    viz.save_figure(viz.plot_importance(perm), out / "figures" / "permutation_importance", formats=fmts)
    top = sv.mean_abs()["feature"].tolist()
    pdps = [partial_dependence(model, ds.X, f, class_label=args.class_label, random_state=args.seed) for f in top[:6]]
    viz.save_figure(viz.plot_partial_dependence(pdps), out / "figures" / "partial_dependence", formats=fmts)
    for feat in top[:3]:
        second = next(f for f in top if f != feat)
        viz.save_figure(viz.plot_shap_dependence(sv, feat, color_by=second), out / "figures" / f"shap_dependence_{feat}", formats=fmts)
    for li in args.local or []:
        if not 0 <= li < len(idx):
            raise GeoExplainError(f"--local index {li} is outside the explained sample (0..{len(idx) - 1}).")
        viz.save_figure(viz.plot_waterfall(sv, li), out / "figures" / f"local_waterfall_{li}", formats=fmts)
        save_table(sv.local(li), out / "tables" / f"local_explanation_{li}.csv")
    plt.close("all")
    _print_json({"output_space": sv.output_space, "additivity": explainer.check_additivity(sv),
                 "top_features": top[:5], "out_dir": str(out)})
    return 0


def cmd_spatial_explain(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    from . import visualization as viz
    from .datasets.synthetic import FEATURE_LABELS
    from .explainers import GeoShapExplainer
    from .io.raster import load_raster_source
    from .io.tabular import load_table, save_table
    from .models import GeoExplainModel
    from .spatial import compute_spatial_shap

    model = GeoExplainModel.load(args.model)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.points:
        import pandas as pd

        df = load_table(args.points)
        missing = [f for f in model.feature_names_ if f not in df.columns]
        if missing:
            raise GeoExplainError(f"Points file lacks model features: {missing}")
        clean = df.dropna(subset=model.feature_names_).reset_index(drop=True)
        bg = clean[model.feature_names_].sample(min(100, len(clean)), random_state=args.seed)
        sv = GeoShapExplainer(model, background=bg, class_label=args.class_label, random_state=args.seed).explain(clean[model.feature_names_])
        keep = [c for c in (args.id_col, args.x, args.y) if c in clean.columns]
        table = pd.concat([clean[keep], sv.to_frame(include_data=False)], axis=1)
        save_table(table, out / "spatial_shap_points.csv")
        result = {"points_csv": str(out / "spatial_shap_points.csv"), "n": len(table), "output_space": sv.output_space}
        if args.crs and args.x in table and args.y in table:
            import geopandas as gpd

            gdf = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table[args.x], table[args.y]), crs=args.crs)
            gdf.to_file(out / "spatial_shap_points.gpkg", driver="GPKG")
            result["geopackage"] = str(out / "spatial_shap_points.gpkg")
        _print_json(result)
        return 0
    if not args.raster:
        raise GeoExplainError("Provide --raster (multi-band GeoTIFF) or --points (CSV).")
    stack = load_raster_source(args.raster, names=args.band_names or model.feature_names_)
    if args.background_data:
        bg = load_table(args.background_data)[model.feature_names_].dropna().sample(n=100, random_state=args.seed, replace=False)
    else:
        bg = stack.to_dataframe()[model.feature_names_].sample(n=min(100, int(stack.valid_mask.sum())), random_state=args.seed)
    res = compute_spatial_shap(model, stack, background=bg, class_label=args.class_label, feature_labels=FEATURE_LABELS,
                               random_state=args.seed)
    written = res.write(out / "rasters")
    save_table(res.summary_table(), out / "spatial_shap_summary.csv")
    if not args.no_maps:
        fmts = _formats(args)
        viz.save_figure(viz.plot_shap_panels(res, source=args.source), out / "maps" / "spatial_shap_panels", formats=fmts)
        viz.save_all_shap_maps(res, out / "maps", source=args.source, formats=fmts)
        viz.save_figure(viz.plot_prediction_uncertainty(res, source=args.source), out / "maps" / "prediction_uncertainty_driver", formats=fmts)
        plt.close("all")
    _print_json({"rasters": {k: str(v) for k, v in written.items()}, "additivity": res.additivity,
                 "uncertainty": res.uncertainty_method, "notes": res.notes})
    return 0


def _run_from_config(cfg) -> int:
    from .workflows import build_report, run_experiment

    index = run_experiment(cfg)
    report = build_report(cfg.out_dir)
    _print_json({"out_dir": cfg.out_dir, "report": str(report), "runtime_seconds": index["runtime_seconds"],
                 "n_tables": len(index["tables"]), "n_figures": len(index["figures"]), "notes": index["notes"]})
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .workflows import ExperimentConfig

    cfg = ExperimentConfig(data=args.data, target=args.target, features=args.features, task=args.task, x=args.x,
                           y=args.y, id_col=args.id_col if args.id_col else None, crs=args.crs, raster=args.raster,
                           models=args.models, primary_model=args.primary or args.models[0], out_dir=args.out_dir,
                           seed=args.seed, n_splits=args.n_splits, test_size=args.test_size, block_size=args.block_size,
                           buffer=args.buffer, formats=list(_formats(args)), dataset_label=args.dataset_label)
    return _run_from_config(cfg)


def cmd_run(args: argparse.Namespace) -> int:
    from .workflows import ExperimentConfig

    cfg = ExperimentConfig.from_json(args.config)
    if args.out_dir:
        cfg.out_dir = args.out_dir
    return _run_from_config(cfg)


def cmd_report(args: argparse.Namespace) -> int:
    from .workflows import build_report

    path = build_report(args.experiment_dir, args.out, title=args.title)
    print(str(path))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .models import available_algorithms
    from .utils import environment_info

    _print_json({"version": __version__, "algorithms_available": available_algorithms(), "environment": environment_info()})
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="geoxplain", description="Explainable ML for geospatial prediction.")
    parser.add_argument("--version", action="version", version=f"geoxplain {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress messages.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, func, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(func=func)
        return p

    p = add("generate-data", cmd_generate_data, "Generate the synthetic example dataset (points CSV + raster stack).")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--size", type=int, default=100, help="Grid size in cells (size x size).")
    p.add_argument("--n-samples", type=int, default=2500)
    p.add_argument("--pixel-size", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)

    p = add("train", cmd_train, "Train a model and save it (optionally reporting hold-out metrics).")
    _add_data_args(p)
    p.add_argument("--algorithm", default="random_forest", help="random_forest | xgboost | lightgbm | mlp | logistic_regression | svm")
    p.add_argument("--params", default=None, help='Estimator parameters as JSON, e.g. \'{"n_estimators": 300}\'.')
    p.add_argument("--out", default="model.joblib")
    p.add_argument("--validation", choices=["none", "random_split", "spatial_split"], default="spatial_split")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--block-size", type=float, default=None)
    p.add_argument("--buffer", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p = add("evaluate", cmd_evaluate, "Evaluate a saved model on a CSV and export metric tables (CSV + Excel).")
    _add_data_args(p, need_features=False)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", default="evaluation")

    p = add("explain", cmd_explain, "Global and local explanations (SHAP, permutation importance, PDP).")
    _add_data_args(p, need_features=False)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", default="explanations")
    p.add_argument("--max-samples", type=int, default=400)
    p.add_argument("--local", type=int, nargs="*", help="Row positions (in the explained sample) for local plots.")
    p.add_argument("--class-label", default=None)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formats", nargs="+", default=["png"])

    p = add("spatial-explain", cmd_spatial_explain, "Spatial SHAP: per-cell contributions as GeoTIFFs (or a points table).")
    p.add_argument("--model", required=True)
    p.add_argument("--raster", default=None, help="Multi-band GeoTIFF whose band descriptions are feature names.")
    p.add_argument("--band-names", nargs="+", default=None, help="Feature names in band order if not stored in the file.")
    p.add_argument("--points", default=None, help="Alternative: CSV of observations (writes a spatial SHAP table).")
    p.add_argument("--x", default="x")
    p.add_argument("--y", default="y")
    p.add_argument("--id-col", default="id")
    p.add_argument("--crs", default=None)
    p.add_argument("--background-data", default=None, help="CSV to draw the SHAP background sample from.")
    p.add_argument("--class-label", default=None)
    p.add_argument("--out-dir", default="spatial_explanation")
    p.add_argument("--no-maps", action="store_true", help="Skip PNG map figures.")
    p.add_argument("--source", default="User data", help="Source text printed on maps.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formats", nargs="+", default=["png"])

    p = add("compare", cmd_compare, "Benchmark several models with random and spatial validation, then explain the primary one.")
    _add_data_args(p)
    p.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "lightgbm", "mlp"])
    p.add_argument("--primary", default=None, help="Model to explain (default: first of --models).")
    p.add_argument("--raster", default=None, help="Optional raster stack to include spatial SHAP maps.")
    p.add_argument("--out-dir", default="comparison")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--block-size", type=float, default=None)
    p.add_argument("--buffer", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formats", nargs="+", default=["png"])
    p.add_argument("--dataset-label", default="User-supplied data")

    p = add("run", cmd_run, "Run a complete experiment from a JSON configuration and write the HTML report.")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default=None, help="Override the output directory in the config.")

    p = add("report", cmd_report, "Build an HTML report from a completed experiment directory.")
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)

    add("info", cmd_info, "Show version, available algorithms and environment.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``geoxplain`` console script; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    try:
        return int(args.func(args) or 0)
    except GeoExplainError as exc:
        print(f"geoxplain: error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"geoxplain: error: file not found: {exc.filename or exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
