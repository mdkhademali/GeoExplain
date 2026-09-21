#!/usr/bin/env python
"""Regenerate every example result and figure from scratch with one command.

    python reproduce.py            # full example (about 2 minutes on one CPU core)
    python reproduce.py --quick    # smaller grid / fewer models for a fast smoke run

Steps: (1) generate the synthetic dataset, (2) run the end-to-end experiment (validation,
model comparison, SHAP, spatial SHAP), (3) build the HTML report, (4) draw the architecture and
workflow diagrams and the logo, (5) copy the headline figures to ``figures/``, (6) refresh the
result tables embedded in the README. Random seeds, package versions and parameters are recorded in
``results/example_run/reproducibility.json`` and ``data/dataset_metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from geoxplain.datasets import write_synthetic_dataset  # noqa: E402
from geoxplain.visualization import draw_architecture, draw_logo, draw_workflow  # noqa: E402
from geoxplain.workflows import ExperimentConfig, build_report, run_experiment  # noqa: E402

#: headline figure -> source file in the experiment's figures/ directory
HEADLINE = {
    "model_comparison": "roc_pr_curves",
    "model_performance": "model_performance",
    "confusion_matrices": "confusion_matrices",
    "feature_importance": "feature_importance",
    "feature_importance_comparison": "feature_importance_comparison",
    "shap_summary": "shap_summary",
    "shap_bar": "shap_bar",
    "local_waterfall": "local_waterfall_1",
    "spatial_prediction": "spatial_prediction",
    "spatial_shap_panels": "spatial_shap_panels",
    "spatial_shap_elevation": "maps/map_elevation_shap",
    "spatial_shap_distance_to_water": "maps/map_distance_to_water_shap",
    "spatial_validation_comparison": "validation_comparison",
    "uncertainty_explanation": "spatial_uncertainty_explanation",
    "feature_vs_spatial_shap": "feature_vs_spatial_shap",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small grid and fewer models (smoke test)")
    ap.add_argument("--out", default="results/example_run", help="experiment output directory")
    ap.add_argument("--no-readme", action="store_true", help="do not refresh README result tables")
    args = ap.parse_args()

    cfg_dict = json.loads((ROOT / "examples" / "example_config.json").read_text())
    size, n_samples, seed = (40, 800, 42) if args.quick else (100, 2500, cfg_dict.get("seed", 42))
    print(f"[1/6] generating synthetic dataset (size={size}, n_samples={n_samples}, seed={seed})")
    write_synthetic_dataset(ROOT / "data", size=size, n_samples=n_samples, seed=seed)

    print("[2/6] running the end-to-end experiment")
    cfg_dict.update(data=str(ROOT / "data" / "samples.csv"), raster=str(ROOT / "data" / "stack.tif"),
                    out_dir=str(ROOT / args.out), seed=seed)
    if args.quick:
        cfg_dict["models"] = ["random_forest", "logistic_regression"]
        cfg_dict["n_splits"] = 3
        cfg_dict["shap_samples"] = 150
    cfg = ExperimentConfig(**cfg_dict)
    index = run_experiment(cfg)

    print("[3/6] building the HTML report")
    build_report(cfg.out_dir, title="GeoExplain example experiment (synthetic data)")

    print("[4/6] drawing diagrams and logo")
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    draw_architecture(fig_dir / "architecture")
    draw_workflow(fig_dir / "workflow")
    draw_logo(fig_dir / "logo", formats=("png", "svg"))

    print("[5/6] copying headline figures to figures/")
    src_dir = Path(cfg.out_dir) / "figures"
    for name, source in HEADLINE.items():
        for ext in ("png", "pdf"):
            src = src_dir / f"{source}.{ext}"
            if src.exists():
                shutil.copy2(src, fig_dir / f"{name}.{ext}")

    for leftover in src_dir.rglob("*.pdf"):  # vector copies live in figures/; keep the run directory light
        leftover.unlink()

    if not args.no_readme and not args.quick:
        print("[6/6] refreshing README result tables")
        from scripts.update_readme_results import update  # type: ignore[import-not-found]

        update(ROOT, Path(cfg.out_dir))
    else:
        print("[6/6] README refresh skipped")
    print(f"Done in {index['runtime_seconds']} s. Report: {Path(cfg.out_dir) / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
