"""Insert result tables computed by the example run into README.md / docs/examples.md.

Numbers are read from ``results/example_run/tables/*.csv`` (never typed by hand). Text between
``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->`` is replaced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP",
         "logistic_regression": "Logistic Regression", "svm": "SVM"}


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_block(run_dir: Path) -> str:
    idx = json.loads((run_dir / "experiment.json").read_text())
    t = run_dir / "tables"
    opt = pd.read_csv(t / "validation_optimism.csv")
    ho = pd.read_csv(t / "holdout_metrics.csv")
    ss = pd.read_csv(t / "spatial_shap_summary.csv")
    piv = {m: opt[opt.metric == m].set_index("model") for m in ("roc_auc", "pr_auc")}
    rows = []
    for model in piv["roc_auc"].index:
        rows.append({
            "Model": NAMES.get(model, model),
            "ROC-AUC random CV": float(piv["roc_auc"].loc[model, "random_kfold"]),
            "ROC-AUC spatial CV": float(piv["roc_auc"].loc[model, "spatial_block_cv"]),
            "PR-AUC random CV": float(piv["pr_auc"].loc[model, "random_kfold"]),
            "PR-AUC spatial CV": float(piv["pr_auc"].loc[model, "spatial_block_cv"]),
        })
    cv_tab = pd.DataFrame(rows)
    sp = ho[ho.strategy == "spatial_split"][["model", "accuracy", "f1", "roc_auc", "pr_auc"]].copy()
    sp["model"] = sp["model"].map(lambda m: NAMES.get(m, m))
    sp.columns = ["Model", "Accuracy", "F1", "ROC-AUC", "PR-AUC"]
    s2 = ss[["feature", "mean_abs_shap", "share_cells_dominant"]].copy()
    s2.columns = ["Feature", "Mean abs SHAP (probability units)", "Share of cells where dominant"]
    env = idx["reproducibility"]["environment"]
    ds = idx["dataset"]
    header = (f"_Computed by `python reproduce.py` on synthetic data ({ds['n_observations']} sampled points, seed "
              f"{idx['config']['seed']}, {idx['config']['n_splits']}-fold CV, Python {env['python']}, scikit-learn "
              f"{env['packages'].get('scikit-learn')}, SHAP {env['packages'].get('shap')}). Values come from actual "
              "execution; they describe artificial data and are not evidence about real-world performance._")
    return "\n\n".join([
        header,
        "**Random vs spatial cross-validation** (mean across folds):\n\n" + _md(cv_tab),
        "**Spatial hold-out block** (models trained on the remaining area):\n\n" + _md(sp),
        f"**Spatial SHAP summary** ({idx['primary_model'].replace('_', ' ')}, all valid grid cells):\n\n" + _md(s2, 4),
    ])


def update(root: Path, run_dir: Path) -> None:
    block = f"{START}\n{build_block(run_dir)}\n{END}"
    for rel in ("README.md", "docs/examples.md"):
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text()
        if START in text and END in text:
            text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _m: block, text, flags=re.S)
            path.write_text(text)
            print(f"  updated {rel}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    update(root, root / "results" / "example_run")
