"""Self-contained HTML report generation for a completed experiment."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.exceptions import DataValidationError

_CSS = """
body{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:#1d2733;max-width:1080px;margin:2rem auto;padding:0 1.2rem;line-height:1.5}
h1{color:#0B2545;margin-bottom:.2rem} h2{color:#0B2545;border-bottom:2px solid #1B9AAA;padding-bottom:.25rem;margin-top:2.2rem}
h3{color:#13315C} .sub{color:#5C6B7A;margin-top:0} table{border-collapse:collapse;margin:.6rem 0 1.2rem;font-size:.86rem}
th{background:#0B2545;color:#fff;text-align:left;padding:.35rem .6rem} td{border-bottom:1px solid #d9e2ea;padding:.3rem .6rem}
tr:nth-child(even) td{background:#f5f9fb} figure{margin:1rem 0} figure img{max-width:100%;border:1px solid #e1e8ee}
figcaption{font-size:.83rem;color:#5C6B7A} .warn{background:#fff8e6;border-left:4px solid #E0A800;padding:.6rem .9rem;margin:1rem 0}
.note{background:#eef8fa;border-left:4px solid #1B9AAA;padding:.6rem .9rem;margin:1rem 0} code{background:#eef2f5;padding:0 .25rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem} footer{margin-top:3rem;font-size:.8rem;color:#5C6B7A}
"""


def _img(root: Path, rel: str, caption: str) -> str:
    p = root / rel
    if not p.exists():
        return f"<p><em>Figure unavailable: {html.escape(rel)}</em></p>"
    data = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _table(root: Path, rel: str | None, digits: int = 3, max_rows: int = 60) -> str:
    if not rel or not (root / rel).exists():
        return "<p><em>Table unavailable.</em></p>"
    df = pd.read_csv(root / rel)
    return df.head(max_rows).to_html(index=False, float_format=lambda v: f"{v:.{digits}f}", border=0, na_rep="–")


def _kv(rows: list[tuple[str, Any]]) -> str:
    body = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def build_report(experiment_dir: str | Path, out_path: str | Path | None = None, title: str | None = None) -> Path:
    """Create ``report.html`` from ``<experiment_dir>/experiment.json`` and return its path.

    Images are embedded (base64) so the report is a single portable file.
    """
    root = Path(experiment_dir)
    idx_path = root / "experiment.json"
    if not idx_path.exists():
        raise DataValidationError(f"{idx_path} not found; run an experiment first (geoxplain run).")
    idx = json.loads(idx_path.read_text())
    cfg, ds, tabs, figs = idx["config"], idx["dataset"], idx["tables"], idx["figures"]
    env = idx["reproducibility"]["environment"]
    task = idx["task"]
    parts: list[str] = []
    ttl = title or "GeoExplain experiment report"
    parts.append(f"<h1>{html.escape(ttl)}</h1><p class='sub'>GeoExplain v{idx['geoxplain_version']} · generated "
                 f"{idx['created_utc']} UTC · runtime {idx['runtime_seconds']} s</p>")
    if "synthetic" in cfg.get("dataset_label", "").lower():
        parts.append("<div class='warn'><strong>Synthetic data.</strong> " + html.escape(cfg["dataset_label"]) +
                     ". All numbers below come from actually executing the workflow on artificial data; they do not "
                     "describe any real-world phenomenon.</div>")
    parts.append("<div class='note'><strong>How to read SHAP results.</strong> SHAP values quantify how a fitted model's "
                 "prediction depends on each input relative to a base value. They describe <em>model behaviour</em> "
                 "(association and prediction), not causal effects of the variables in the real world.</div>")

    parts.append("<h2>1. Dataset</h2>")
    parts.append(_kv([
        ("Source file", cfg["data"]), ("Observations", ds["n_observations"]), ("Task", task),
        ("Target", ds["target"]), ("Features", ", ".join(ds["features"])), ("CRS", ds.get("crs")),
        ("Class counts", ds.get("class_counts", "n/a")), ("Extent", ds.get("extent")),
        ("Moran's I of target", f"{ds['target_morans_i']['I']:.3f} (p = {ds['target_morans_i']['p_value']:.3f})"),
    ]))

    parts.append("<h2>2. Model configuration</h2>")
    cfgs = idx["reproducibility"]["model_configs"]
    rows = [(k, json.dumps(v["params"], default=str)) for k, v in cfgs.items()]
    parts.append(_kv(rows))
    parts.append(f"<p>Primary model for explanation: <code>{idx['primary_model']}</code>, refit on all observations. "
                 f"Random seed: <code>{cfg['seed']}</code>.</p>")

    parts.append("<h2>3. Validation strategy</h2>")
    sp = idx["reproducibility"]["splits"]
    parts.append("<p>Two strategies were compared for every model: <strong>random</strong> splitting/k-fold, and "
                 "<strong>spatial</strong> block hold-out/cross-validation in which whole spatial blocks are withheld. Nearby "
                 "observations are similar (spatial autocorrelation), so random splitting can be optimistic.</p>")
    parts.append(_kv([("Block size (map units)", f"{idx['reproducibility']['block_size']:.1f}"),
                      ("k folds", cfg["n_splits"]), ("Hold-out test fraction", cfg["test_size"]),
                      ("Buffer (map units)", cfg["buffer"]),
                      ("Spatial hold-out: train / test", f"{sp['spatial_train_after_buffer']} / {sp['spatial_test']}"),
                      ("Random hold-out: test", sp["random_test"])]))
    parts.append(_img(root, figs["cv_folds"], "Spatial block cross-validation folds (sample locations coloured by fold)."))

    parts.append("<h2>4. Metrics</h2><h3>Hold-out metrics (random vs spatial split)</h3>")
    parts.append(_table(root, tabs.get("holdout_metrics")))
    parts.append(_img(root, figs["model_performance"], "Performance of each model on the spatially held-out block."))
    parts.append("<h3>Cross-validation: random vs spatial (mean across folds)</h3>")
    parts.append(_table(root, tabs.get("validation_optimism")))
    parts.append(_img(root, figs["validation_comparison"], "Random k-fold versus spatial block cross-validation."))
    parts.append("<p>The <code>optimism</code> column is the random-CV score minus the spatial-CV score (sign-adjusted so "
                 "that positive means random CV looked better). Values are data-dependent and are not guaranteed to be positive.</p>")

    parts.append("<h2>5. Model comparison</h2>")
    if "roc_pr_curves" in figs:
        parts.append(_img(root, figs["roc_pr_curves"], "ROC and precision–recall curves on the spatial hold-out block."))
        parts.append(_img(root, figs["confusion_matrices"], "Confusion matrices on the spatial hold-out block."))
    if "feature_importance_comparison" in figs:
        parts.append(_img(root, figs["feature_importance_comparison"],
                          "Relative mean |SHAP| by model (shares; absolute units differ between models)."))

    parts.append("<h2>6. Feature importance</h2>")
    parts.append(_img(root, figs["feature_importance"], "Permutation importance on the spatial hold-out block."))
    parts.append(_table(root, tabs.get("permutation_importance")))

    parts.append(f"<h2>7. SHAP explanations ({html.escape(idx['primary_model'])})</h2>")
    parts.append(f"<p>Output space: <strong>{html.escape(idx['shap_output_space'])}</strong>. Additivity check "
                 f"(base value + Σ SHAP = model output): max abs error <code>{idx['shap_additivity']['max_abs_error']:.2e}</code>.</p>")
    parts.append("<div class='grid'>" + _img(root, figs["shap_summary"], "SHAP summary (global).") +
                 _img(root, figs["shap_bar"], "Mean |SHAP| (global).") + "</div>")
    dep = [k for k in figs if k.startswith("shap_dependence_")]
    if dep:
        parts.append("<div class='grid'>" + "".join(_img(root, figs[k], f"SHAP dependence: {k.replace('shap_dependence_', '')}") for k in dep) + "</div>")
    parts.append(_img(root, figs["partial_dependence"], "Partial dependence and ICE curves."))
    loc = [k for k in figs if k.startswith("local_")]
    if loc:
        parts.append("<h3>Local explanations</h3><div class='grid'>" + "".join(_img(root, figs[k], k.replace("_", " ")) for k in loc) + "</div>")
        parts.append(_table(root, tabs.get("local_explanations")))
    parts.append("<h3>Spatial autocorrelation of SHAP contributions (point data)</h3>")
    parts.append(_table(root, tabs.get("shap_spatial_autocorrelation")))

    parts.append("<h2>8. Spatial explanation maps</h2>")
    if "spatial_shap_panels" in figs:
        parts.append(_img(root, figs["spatial_prediction"], "Model prediction raster."))
        parts.append(_img(root, figs["spatial_shap_panels"], "Spatial SHAP: contribution of each feature per grid cell (shared colour scale)."))
        parts.append(_img(root, figs["spatial_uncertainty_explanation"], "Prediction, uncertainty and dominant driver."))
        parts.append(_img(root, figs["feature_vs_spatial_shap"], "Input feature and its spatial contribution."))
        parts.append(_table(root, tabs.get("spatial_shap_summary")))
        parts.append("<p>Raster outputs: " + ", ".join(f"<code>{html.escape(v)}</code>" for v in idx["rasters"].values()) + "</p>")
        if idx["spatial"].get("uncertainty_description"):
            parts.append(f"<p><strong>Uncertainty:</strong> {html.escape(idx['spatial']['uncertainty_description'])}</p>")
    else:
        parts.append("<p>No raster stack was supplied, so no spatial maps were produced.</p>")

    parts.append("<h2>9. Reproducibility</h2>")
    parts.append(_kv([("Python", env.get("python")), ("Platform", env.get("platform")), ("Seed", cfg["seed"]),
                      *[(k, v) for k, v in sorted(env.get("packages", {}).items())]]))
    if idx["notes"]:
        parts.append("<h3>Notes</h3><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in idx["notes"]) + "</ul>")
    parts.append("<footer>Generated by GeoExplain (GPL-3.0-or-later). Re-run with <code>geoxplain run --config &lt;config.json&gt;</code>.</footer>")
    doc = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(ttl)}</title>"
           f"<style>{_CSS}</style></head><body>{''.join(parts)}</body></html>")
    out = Path(out_path) if out_path else root / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out
