"""Regression: predict a continuous synthetic susceptibility index and explain it spatially.

Run from the repository root:  python examples/regression/run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from _common import DATA, FEATURES, out_dir, require_data  # noqa: E402

from geoxplain import (  # noqa: E402
    GeoExplainModel,
    GeoShapExplainer,
    compute_spatial_shap,
    load_dataset,
    read_raster_stack,
    spatial_split,
)
from geoxplain import visualization as viz  # noqa: E402

require_data()
out = out_dir("regression")
ds = load_dataset(DATA / "samples.csv", target="susceptibility_index", features=FEATURES, crs="EPSG:32645")
train, test = spatial_split(ds.require_coords(), 0.25, random_state=1)

probe = GeoExplainModel("random_forest", "regression", random_state=1).fit(ds.X.iloc[train], ds.y.iloc[train])
print("Spatial hold-out:", {k: round(v, 3) for k, v in probe.evaluate(ds.X.iloc[test], ds.y.iloc[test]).items()})

model = GeoExplainModel("random_forest", "regression", random_state=1).fit(ds.X, ds.y)
explainer = GeoShapExplainer(model, background=ds.X.sample(100, random_state=1))
sv = explainer.explain(ds.X.sample(300, random_state=1))
print(sv.mean_abs().round(3).to_string(index=False), "\nadditivity:", explainer.check_additivity(sv))
viz.plot_shap_summary(sv, path=out / "shap_summary")

stack = read_raster_stack(DATA / "stack.tif")
res = compute_spatial_shap(model, stack, background=ds.X.sample(100, random_state=1))
print("uncertainty:", res.uncertainty_method)
res.write(out / "rasters")
viz.plot_shap_panels(res, path=out / "spatial_shap_panels")
viz.plot_prediction_uncertainty(res, path=out / "prediction_uncertainty")
print("Outputs written to", out)
