## Quick start

All snippets use the built-in **synthetic** dataset (see `data/README.md`); results illustrate the API, not real-world performance.

## 1. Generate data

```bash
geoxplain generate-data --out-dir data --size 100 --n-samples 2500 --seed 42
```

This writes `data/samples.csv` (points), `data/stack.tif` (7-band predictor stack), per-feature rasters, generating-process rasters in `data/truth/` and `data/dataset_metadata.json`.

## 2. Python API

```python
from geoxplain import (GeoExplainModel, GeoShapExplainer, compute_spatial_shap,
                       load_dataset, read_raster_stack, spatial_split)

features = ["ndvi", "ndbi", "lst", "elevation", "rainfall", "distance_to_water", "population"]
ds = load_dataset("data/samples.csv", target="target", features=features, crs="EPSG:32645")

# validate spatially: hold out whole blocks of the study area
train, test = spatial_split(ds.require_coords(), test_size=0.25, random_state=42)
probe = GeoExplainModel("random_forest", "classification").fit(ds.X.iloc[train], ds.y.iloc[train])
print(probe.evaluate(ds.X.iloc[test], ds.y.iloc[test]))

# explain the final model (refit on all data)
model = GeoExplainModel("random_forest", "classification").fit(ds.X, ds.y)
sv = GeoShapExplainer(model, background=ds.X.sample(100, random_state=0)).explain(ds.X.sample(300, random_state=0))
print(sv.mean_abs())                       # global importance table
print(sv.local(0))                         # local explanation of one observation

# WHERE do the features matter? -> georeferenced SHAP rasters
stack = read_raster_stack("data/stack.tif")
result = compute_spatial_shap(model, stack, background=ds.X.sample(100, random_state=0))
result.write("outputs/rasters")            # NDVI_SHAP.tif, ..., prediction.tif, uncertainty.tif
```

## 3. Command line

```bash
FEATS="ndvi ndbi lst elevation rainfall distance_to_water population"
geoxplain train --data data/samples.csv --target target --features $FEATS --crs EPSG:32645 \
                --algorithm random_forest --out outputs/rf.joblib
geoxplain evaluate --model outputs/rf.joblib --data data/samples.csv --target target --out-dir outputs/eval
geoxplain explain --model outputs/rf.joblib --data data/samples.csv --target target --out-dir outputs/explain --local 0 1
geoxplain spatial-explain --model outputs/rf.joblib --raster data/stack.tif --out-dir outputs/spatial
geoxplain compare --data data/samples.csv --target target --features $FEATS --crs EPSG:32645 \
                  --raster data/stack.tif --out-dir outputs/compare        # writes report.html too
```

## 4. Everything at once

```bash
python reproduce.py        # or: geoxplain run --config examples/example_config.json
```

Open `results/example_run/report.html`.
