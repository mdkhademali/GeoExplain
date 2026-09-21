"""Spatial explainability: where do the features matter? (signature GeoExplain workflow)

Run from the repository root:  python examples/spatial_explainability/run.py
Writes one SHAP GeoTIFF per feature plus prediction, uncertainty and dominant-driver rasters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import rasterio  # noqa: E402
from _common import DATA, FEATURES, out_dir, require_data  # noqa: E402

from geoxplain import GeoExplainModel, compute_spatial_shap, load_dataset, read_raster_stack  # noqa: E402
from geoxplain import visualization as viz  # noqa: E402
from geoxplain.datasets import FEATURE_LABELS  # noqa: E402

require_data()
out = out_dir("spatial_explainability")
ds = load_dataset(DATA / "samples.csv", target="target", features=FEATURES, crs="EPSG:32645")
model = GeoExplainModel("random_forest", "classification", random_state=42).fit(ds.X, ds.y)

stack = read_raster_stack(DATA / "stack.tif")  # band descriptions are the feature names
result = compute_spatial_shap(model, stack, background=ds.X.sample(100, random_state=42), feature_labels=FEATURE_LABELS)
written = result.write(out / "rasters")
print(result.summary_table().round(4).to_string(index=False))
print("additivity check:", result.additivity)

# CRS / transform / size are preserved exactly:
with rasterio.open(written["NDVI_SHAP"]) as shap_raster:
    assert shap_raster.crs == stack.crs and shap_raster.transform == stack.transform
    assert (shap_raster.height, shap_raster.width) == stack.shape

viz.plot_shap_map(result, "elevation", path=out / "map_elevation_shap")
viz.plot_shap_panels(result, path=out / "spatial_shap_panels")
viz.plot_prediction_uncertainty(result, path=out / "prediction_uncertainty_driver")
print("Outputs written to", out)
