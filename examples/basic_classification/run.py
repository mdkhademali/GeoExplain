"""Basic classification: train a Random Forest and evaluate it with random and spatial hold-out.

Run from the repository root:  python examples/basic_classification/run.py
Data are synthetic (see data/README.md); numbers illustrate the API, not real-world skill.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from _common import DATA, FEATURES, out_dir, require_data  # noqa: E402

from geoxplain import GeoExplainModel, load_dataset, spatial_split  # noqa: E402
from geoxplain.metrics import export_metrics, metrics_table  # noqa: E402
from geoxplain.validation import random_split  # noqa: E402

require_data()
ds = load_dataset(DATA / "samples.csv", target="target", features=FEATURES, crs="EPSG:32645")
print(ds.summary()["class_counts"], "observations:", len(ds.X))

splits = {
    "random hold-out": random_split(len(ds.X), 0.25, random_state=42, stratify=ds.y.to_numpy()),
    "spatial hold-out": spatial_split(ds.require_coords(), 0.25, random_state=42),
}
results = {}
for name, (train, test) in splits.items():
    model = GeoExplainModel("random_forest", "classification", random_state=42).fit(ds.X.iloc[train], ds.y.iloc[train])
    results[name] = model.evaluate(ds.X.iloc[test], ds.y.iloc[test])

table = metrics_table(results, index_name="validation")
pd.set_option("display.width", 160)
print(table.round(3))
print(export_metrics(table, out_dir("basic_classification"), stem="metrics"))
