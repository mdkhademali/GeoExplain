"""Statistical figures: SHAP plots, importance, effects, model comparison, validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..explainers.effects import PartialDependenceResult
from ..explainers.shap_explainer import ShapValues
from ..utils.optional import require
from .style import GREY, QUALITATIVE, TEAL, apply_style, model_color, save_figure

_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
    "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "mae": "MAE",
    "mse": "MSE", "rmse": "RMSE", "r2": "R²", "mape_percent": "MAPE (%)",
}


def _pretty(model_key: str) -> str:
    return model_key.replace("_", " ").title().replace("Xgboost", "XGBoost").replace("Lightgbm", "LightGBM").replace(
        "Mlp", "MLP").replace("Svm", "SVM")


def _finish(fig: plt.Figure, path: str | Path | None, formats: Sequence[str]) -> plt.Figure:
    if path is not None:
        save_figure(fig, path, formats=formats, close=False)
    return fig


# --------------------------------------------------------------------------- SHAP plots
def plot_shap_summary(sv: ShapValues, path: str | Path | None = None, max_display: int = 10,
                      formats: Sequence[str] = ("png",)) -> plt.Figure:
    """SHAP beeswarm summary plot (global explanation, distribution of contributions)."""
    shap = require("shap", "SHAP plots")
    apply_style()
    plt.figure(figsize=(6.6, 0.42 * min(max_display, len(sv.feature_names)) + 1.8))
    shap.summary_plot(sv.values, features=sv.data.values, feature_names=sv.feature_names,
                      max_display=max_display, show=False, plot_size=None)
    fig = plt.gcf()
    fig.axes[0].set_xlabel(f"SHAP value (impact on model output, {sv.output_space} units)", fontsize=9)
    fig.axes[0].set_title("SHAP summary", loc="left")
    return _finish(fig, path, formats)


def plot_shap_bar(sv: ShapValues, path: str | Path | None = None, max_display: int = 15,
                  formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Bar chart of mean absolute SHAP value per feature."""
    apply_style()
    df = sv.mean_abs().head(max_display).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(df) + 1.4))
    ax.barh(df["feature"], df["mean_abs_shap"], color=TEAL)
    for y, v in enumerate(df["mean_abs_shap"]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=7, color=GREY)
    ax.set_xlabel(f"Mean |SHAP value| ({sv.output_space} units)")
    ax.set_title("Global feature importance (mean |SHAP|)", loc="left")
    ax.set_xlim(0, df["mean_abs_shap"].max() * 1.15)
    return _finish(fig, path, formats)


def plot_waterfall(sv: ShapValues, index: int, path: str | Path | None = None, max_display: int = 10,
                   formats: Sequence[str] = ("png",)) -> plt.Figure:
    """SHAP waterfall plot for observation ``index`` (local explanation)."""
    shap = require("shap", "SHAP plots")
    apply_style()
    exp = shap.Explanation(values=sv.values[index], base_values=sv.base_value, data=sv.data.values[index],
                           feature_names=sv.feature_names)
    plt.figure(figsize=(6.6, 4.2))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.axes[0].set_title(f"Local explanation, observation {index}", loc="left")
    return _finish(fig, path, formats)


def plot_force(sv: ShapValues, index: int, path: str | Path | None = None,
               formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Force-style plot for one observation (matplotlib rendering of ``shap.force_plot``)."""
    shap = require("shap", "SHAP plots")
    apply_style()
    shap.force_plot(sv.base_value, sv.values[index], sv.data.iloc[index].round(3), feature_names=sv.feature_names,
                    matplotlib=True, show=False, figsize=(10, 2.6))
    return _finish(plt.gcf(), path, formats)


def plot_shap_dependence(sv: ShapValues, feature: str, color_by: str | None = None,
                         path: str | Path | None = None, formats: Sequence[str] = ("png",)) -> plt.Figure:
    """SHAP dependence plot: feature value against its SHAP contribution (optionally coloured)."""
    apply_style()
    j = sv.feature_names.index(feature)
    x = sv.data[feature].to_numpy()
    y = sv.values[:, j]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    if color_by is not None:
        sc = ax.scatter(x, y, c=sv.data[color_by].to_numpy(), cmap="viridis", s=10, alpha=0.8, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label(color_by, fontsize=8)
    else:
        ax.scatter(x, y, s=10, alpha=0.7, color=TEAL, linewidths=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value ({sv.output_space} units)")
    ax.set_title(f"SHAP dependence: {feature}", loc="left")
    return _finish(fig, path, formats)


# --------------------------------------------------------------------------- importance / effects
def plot_importance(df: pd.DataFrame, path: str | Path | None = None, value: str = "importance_mean",
                    error: str | None = "importance_std", title: str = "Permutation importance",
                    xlabel: str | None = None, formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Horizontal bar chart from an importance table (permutation or built-in)."""
    apply_style()
    d = df.sort_values(value, ascending=True)
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(d) + 1.4))
    xerr = d[error] if error and error in d else None
    ax.barh(d["feature"], d[value], xerr=xerr, color=TEAL, ecolor=GREY, capsize=2)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel(xlabel or value)
    ax.set_title(title, loc="left")
    return _finish(fig, path, formats)


def plot_partial_dependence(results: Sequence[PartialDependenceResult], path: str | Path | None = None,
                            ncols: int = 3, formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Partial dependence curves (with thin ICE lines) for several features."""
    apply_style()
    n = len(results)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows), squeeze=False, sharey=True)
    for ax, res in zip(axes.flat, results, strict=False):
        if res.ice is not None and len(res.ice):
            ax.plot(res.grid, res.ice[:60].T, color="#BBBBBB", lw=0.4, alpha=0.6)
        ax.plot(res.grid, res.average, color=TEAL, lw=2)
        ax.set_xlabel(res.feature)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(results[0].output_label)
    fig.suptitle("Partial dependence (bold) and individual conditional expectation (grey)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path, formats)


# --------------------------------------------------------------------------- comparison
def plot_metric_comparison(table: pd.DataFrame, metrics: Sequence[str], path: str | Path | None = None,
                           errors: pd.DataFrame | None = None, title: str = "Model performance",
                           formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Grouped bar chart of metrics (rows = models); optional error bars from ``errors``."""
    apply_style()
    metrics = [m for m in metrics if m in table.columns]
    n_m = len(metrics)
    fig, axes = plt.subplots(1, n_m, figsize=(2.6 * n_m + 0.6, 3.3), squeeze=False)
    for ax, met in zip(axes.flat, metrics, strict=False):
        models = list(table.index)
        vals = table[met].to_numpy(dtype=float)
        err = errors[met].to_numpy(dtype=float) if errors is not None and met in errors else None
        ax.bar(range(len(models)), vals, yerr=err, color=[model_color(m, i) for i, m in enumerate(models)],
               capsize=2, ecolor=GREY)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([_pretty(m) for m in models], rotation=40, ha="right", fontsize=7)
        ax.set_title(_LABELS.get(met, met), fontsize=9)
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path, formats)


def plot_roc_pr_curves(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], positive_label: Any = 1,
                       path: str | Path | None = None, formats: Sequence[str] = ("png",)) -> plt.Figure:
    """ROC and precision-recall curves for several models.

    ``curves`` maps a model key to ``(y_true, positive_class_score)``.
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    prevalence = None
    for i, (name, (y_true, score)) in enumerate(curves.items()):
        yt = (np.asarray(y_true) == positive_label).astype(int)
        prevalence = yt.mean()
        fpr, tpr, _ = roc_curve(yt, score)
        prec, rec, _ = precision_recall_curve(yt, score)
        c = model_color(name, i)
        ax1.plot(fpr, tpr, color=c, lw=1.6, label=f"{_pretty(name)} (AUC {auc(fpr, tpr):.3f})")
        ax2.plot(rec, prec, color=c, lw=1.6, label=f"{_pretty(name)} (AP {auc(rec, prec):.3f})")
    ax1.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    if prevalence is not None:
        ax2.axhline(prevalence, color="#999999", lw=0.8, ls="--")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC curves")
    ax2.set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall curves")
    for ax in (ax1, ax2):
        ax.legend(loc="lower right" if ax is ax1 else "upper right", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return _finish(fig, path, formats)


def plot_confusion_matrices(matrices: Mapping[str, np.ndarray], labels: Sequence[Any],
                            path: str | Path | None = None, formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Row-normalised confusion matrices (counts annotated) for several models."""
    apply_style()
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 0.3, 3.1), squeeze=False)
    for ax, (name, cm) in zip(axes.flat, matrices.items(), strict=False):
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([str(x) for x in labels])
        ax.set_yticklabels([str(x) for x in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(_pretty(name), fontsize=9)
        for s in ax.spines.values():
            s.set_visible(True)
    fig.tight_layout()
    return _finish(fig, path, formats)


def plot_importance_comparison(importance: pd.DataFrame, path: str | Path | None = None,
                               title: str = "Relative mean |SHAP| by model",
                               formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Grouped bars comparing relative importance (index = features, columns = models)."""
    apply_style()
    feats = list(importance.index)
    models = list(importance.columns)
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, m in enumerate(models):
        ax.bar(np.arange(len(feats)) + i * width - 0.4 + width / 2, importance[m].to_numpy(), width,
               label=_pretty(m), color=model_color(m, i))
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=35, ha="right")
    ax.set_ylabel("Share of total mean |SHAP|")
    ax.set_title(title, loc="left")
    ax.legend(ncol=min(len(models), 4))
    fig.tight_layout()
    return _finish(fig, path, formats)


def plot_validation_comparison(folds: pd.DataFrame, metrics: Sequence[str], path: str | Path | None = None,
                               strategies: Sequence[str] = ("random_kfold", "spatial_block_cv"),
                               formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Random vs spatial cross-validation: per-model mean ± SD across folds.

    ``folds`` needs columns ``model``, ``strategy`` and the requested metrics.
    """
    apply_style()
    labels = {"random_kfold": "Random k-fold", "spatial_block_cv": "Spatial block CV"}
    colors = {"random_kfold": "#BBBBBB", "spatial_block_cv": TEAL}
    models = list(dict.fromkeys(folds["model"]))
    metrics = [m for m in metrics if m in folds.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.6), squeeze=False)
    width = 0.38
    for ax, met in zip(axes.flat, metrics, strict=False):
        for k, strat in enumerate(strategies):
            sub = folds[folds["strategy"] == strat].groupby("model")[met].agg(["mean", "std"]).reindex(models)
            xs = np.arange(len(models)) + (k - 0.5) * width
            ax.bar(xs, sub["mean"], width, yerr=sub["std"], capsize=2, color=colors.get(strat, QUALITATIVE[k]),
                   label=labels.get(strat, strat), ecolor=GREY)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([_pretty(m) for m in models], rotation=30, ha="right", fontsize=7)
        ax.set_title(_LABELS.get(met, met), fontsize=9)
    handles, labs = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", ncol=2, fontsize=8)
    fig.suptitle("Random vs spatial validation (mean ± SD across folds)", x=0.01, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path, formats)


def plot_fold_map(coords: np.ndarray, fold_ids: np.ndarray, path: str | Path | None = None,
                  title: str = "Spatial block cross-validation folds", formats: Sequence[str] = ("png",)) -> plt.Figure:
    """Scatter of sample locations coloured by spatial CV fold."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for f in sorted(np.unique(fold_ids)):
        m = fold_ids == f
        ax.scatter(coords[m, 0], coords[m, 1], s=6, color=QUALITATIVE[int(f) % len(QUALITATIVE)], label=f"Fold {int(f) + 1}")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, loc="left")
    ax.legend(markerscale=2, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _finish(fig, path, formats)
