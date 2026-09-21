"""Shared figure style and saving helpers (clean, publication-oriented defaults)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt

#: Colour identity of the project (dark blue / white / teal accents).
NAVY = "#0B2545"
BLUE = "#13315C"
TEAL = "#1B9AAA"
CYAN = "#8DE0EA"
GREY = "#5C6B7A"
LIGHT = "#EEF4F8"

#: Colour-blind-safe qualitative palette used for model comparisons.
MODEL_COLORS = {
    "random_forest": "#1B9AAA",
    "xgboost": "#0B2545",
    "lightgbm": "#E07A5F",
    "mlp": "#81B29A",
    "logistic_regression": "#8D6B94",
    "linear_regression": "#8D6B94",
    "svm": "#C9A227",
}
QUALITATIVE = ["#1B9AAA", "#0B2545", "#E07A5F", "#81B29A", "#8D6B94", "#C9A227", "#5C6B7A"]

DIVERGING_CMAP = "RdBu_r"  # positive contributions red, negative blue, zero white
SEQUENTIAL_CMAP = "viridis"
UNCERTAINTY_CMAP = "magma"


def apply_style() -> None:
    """Apply the GeoExplain matplotlib style (idempotent)."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def model_color(name: str, index: int = 0) -> str:
    """Return a stable colour for a model key (falls back to the qualitative palette)."""
    return MODEL_COLORS.get(name, QUALITATIVE[index % len(QUALITATIVE)])


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    close: bool = True,
) -> list[Path]:
    """Save ``fig`` in each of ``formats`` next to ``path`` (extension of ``path`` is ignored).

    Returns the list of written files. PNG is written at ``dpi``; PDF/SVG are vector.
    """
    base = Path(path)
    if base.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
        written.append(target)
    if close:
        plt.close(fig)
    return written


def panel_label(ax: plt.Axes, text: str) -> None:
    """Add a bold panel label such as ``(a)`` to the upper-left corner of ``ax``."""
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


def flatten(axes: Iterable) -> list:
    """Flatten an array of axes to a list."""
    return list(getattr(axes, "flat", axes))
