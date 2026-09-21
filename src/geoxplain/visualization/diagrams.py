"""Project diagrams and logo, drawn with matplotlib (no external graph tools required)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from .style import BLUE, CYAN, GREY, LIGHT, NAVY, TEAL, apply_style, save_figure


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, fc: str = LIGHT, ec: str = NAVY,
         tc: str = NAVY, fs: float = 8.5, bold: bool = False, lw: float = 1.0) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], color: str = GREY, lw: float = 1.3,
           style: str = "-|>", rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def draw_architecture(path: str | Path, formats: Sequence[str] = ("png", "svg", "pdf")) -> list[Path]:
    """Layered architecture diagram of the ``geoxplain`` package."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.text(0.1, 6.35, "GeoExplain architecture", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(0.1, 6.08, "Package geoxplain: modular layers from data to explained, georeferenced outputs", fontsize=8.5, color=GREY)

    layers = [
        ("Interfaces", 5.05, [("Python API", ""), ("CLI\ngeoxplain ...", ""), ("Workflows\nrun_experiment", ""),
                              ("QGIS Processing\n(optional)", "")], NAVY, "white"),
        ("Explainability", 3.85, [("explainers\nSHAP · permutation\nPDP · local", ""), ("spatial\nspatial SHAP\nprediction · uncertainty", ""),
                                  ("validation\nrandom · spatial split\nblock CV · Moran's I", ""),
                                  ("metrics\nclassification\nregression", "")], TEAL, "white"),
        ("Core", 2.65, [("models\nRF · XGBoost · LightGBM\nMLP · LogReg · SVM", ""),
                        ("io\nCSV · GeoTIFF\nCRS-preserving", ""), ("datasets\nsynthetic scenes", ""),
                        ("utils\nreproducibility · errors", "")], BLUE, "white"),
    ]
    for name, y, boxes, color, _tc in layers:
        ax.add_patch(Rectangle((0.1, y - 0.1), 10.3, 1.05, fc=color, alpha=0.08, ec="none"))
        ax.text(0.2, y + 0.84, name, fontsize=8.5, fontweight="bold", color=color)
        n = len(boxes)
        bw = (10.0 - 0.15 * (n - 1)) / n
        for i, (label, _) in enumerate(boxes):
            _box(ax, 0.2 + i * (bw + 0.15), y - 0.06, bw, 0.76, label, fc="white", ec=color, tc=NAVY, fs=8)
    for y_top, y_bot in ((5.0, 4.75), (3.8, 3.55)):
        for x in (2.4, 5.2, 8.0):
            _arrow(ax, (x, y_top - 0.02), (x, y_bot), color=GREY)

    ax.add_patch(Rectangle((0.1, 0.35), 10.3, 1.75, fc=CYAN, alpha=0.18, ec="none"))
    ax.text(0.2, 1.88, "Data in / results out", fontsize=8.5, fontweight="bold", color=NAVY)
    ins = ["Point tables\nCSV: x, y,\nfeatures, target", "Raster stacks\nGeoTIFF,\nshared grid"]
    outs = ["Metric tables\nCSV · Excel", "Prediction,\nuncertainty,\nSHAP GeoTIFFs", "Figures\nPNG 300 dpi\nSVG · PDF", "HTML report\nreproducibility\nrecord"]
    for i, t in enumerate(ins):
        _box(ax, 0.25 + i * 1.75, 0.5, 1.65, 1.15, t, fc="white", ec=NAVY, fs=7.8)
    ax.text(3.85, 1.05, "→", fontsize=18, color=GREY, ha="center", va="center")
    for i, t in enumerate(outs):
        _box(ax, 4.2 + i * 1.55, 0.5, 1.45, 1.15, t, fc="white", ec=TEAL, fs=7.8)
    _arrow(ax, (5.2, 2.6), (5.2, 2.12), color=GREY)
    return save_figure(fig, path, formats=formats)


def draw_workflow(path: str | Path, formats: Sequence[str] = ("png", "svg", "pdf")) -> list[Path]:
    """Conceptual end-to-end workflow diagram."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(0.1, 4.35, "GeoExplain workflow: from data to where features matter", fontsize=12.5, fontweight="bold", color=NAVY)

    row1 = ["Dataset\n(points / rasters)", "Validate &\npreprocess", "Split\nrandom | spatial", "Train models\nRF · XGBoost · LightGBM\nMLP · LogReg", "Evaluate\n+ spatial validation"]
    row2 = ["Global SHAP\n+ permutation", "Local SHAP\n(waterfall, force)", "Spatial SHAP\nper-cell contributions", "Maps & GeoTIFFs\nprediction · uncertainty", "Export & HTML\nreport"]
    w, h, gap = 1.8, 0.95, 0.3
    for i, t in enumerate(row1):
        x = 0.15 + i * (w + gap)
        _box(ax, x, 2.75, w, h, t, fc=NAVY if i in (3,) else "white", ec=NAVY, tc="white" if i == 3 else NAVY, fs=8)
        if i < len(row1) - 1:
            _arrow(ax, (x + w, 3.22), (x + w + gap, 3.22))
    for i, t in enumerate(row2):
        x = 0.15 + i * (w + gap)
        hi = i == 2
        _box(ax, x, 1.0, w, h, t, fc=TEAL if hi else "white", ec=TEAL, tc="white" if hi else NAVY, fs=8, bold=hi)
        if i < len(row2) - 1:
            _arrow(ax, (x + w, 1.47), (x + w + gap, 1.47))
    x_last = 0.15 + 4 * (w + gap) + w / 2
    x_first = 0.15 + w / 2
    ax.plot([x_last, x_last, x_first], [2.75, 2.35, 2.35], color=GREY, lw=1.3)
    _arrow(ax, (x_first, 2.35), (x_first, 1.97))
    ax.text((x_last + x_first) / 2, 2.42, "final model used for explanation", fontsize=7.5, color=GREY, ha="center", style="italic")
    ax.text(0.15, 0.4, "Key idea: SHAP values are computed for every observation or grid cell and written back to the same\nlocation, so the explanation itself becomes a map.", fontsize=8.3, color=NAVY)
    ax.text(0.15, 0.02, "SHAP describes model behaviour (association), not causal effects.", fontsize=8, color=GREY, style="italic")
    return save_figure(fig, path, formats=formats)


def draw_logo(path: str | Path, formats: Sequence[str] = ("png", "svg"), with_text: bool = True,
              transparent: bool = False) -> list[Path]:
    """GeoExplain logo: raster grid (GIS) + attribution lens (explainability) + bars (ML)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0 if with_text else 2.6, 2.6))
    ax.set_xlim(0, 6.0 if with_text else 2.6)
    ax.set_ylim(0, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 2.3, 2.3, boxstyle="round,pad=0.0,rounding_size=0.28", fc=NAVY, ec="none"))
    rng = np.random.default_rng(3)
    n = 5
    cell = 0.34
    x0, y0 = 0.42, 0.62
    vals = np.array([[-.2, .1, .3, .5, .2], [-.1, .3, .6, .8, .4], [-.4, .1, .5, .9, .6], [-.6, -.2, .2, .4, .3], [-.8, -.5, -.1, .1, .1]])
    for i in range(n):
        for j in range(n):
            v = vals[i, j] + 0.0 * rng.random()
            col = np.array([0x1B, 0x9A, 0xAA]) / 255 if v > 0 else np.array([0x9F, 0xB4, 0xD6]) / 255
            alpha = 0.25 + 0.75 * min(abs(v), 1)
            ax.add_patch(Rectangle((x0 + j * cell, y0 + (n - 1 - i) * cell), cell * 0.92, cell * 0.92, fc=col, alpha=alpha, ec="none"))
    ax.add_patch(Circle((1.62, 1.72), 0.52, fc="none", ec="white", lw=3.2))
    ax.plot([1.99, 2.24], [1.35, 1.1], color="white", lw=4, solid_capstyle="round")
    for k, hgt in enumerate((0.22, 0.42, 0.3)):
        ax.add_patch(Rectangle((1.36 + k * 0.17, 1.5), 0.11, hgt, fc=CYAN, ec="none"))
    if with_text:
        ax.text(2.85, 1.55, "GeoExplain", fontsize=30, fontweight="bold", color=NAVY, va="center")
        ax.text(2.88, 0.85, "Explainable machine learning\nfor geospatial prediction", fontsize=11, color=TEAL, va="center", linespacing=1.3)
        ax.add_patch(Polygon([[2.88, 1.3], [5.6, 1.3], [5.6, 1.32], [2.88, 1.32]], fc=TEAL, ec="none"))
    path = Path(path)
    written: list[Path] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=300, bbox_inches="tight", transparent=transparent, facecolor="none" if transparent else "white")
        written.append(target)
    plt.close(fig)
    return written
