## model_comparison

Benchmarks Random Forest, XGBoost, LightGBM and an MLP with random and spatial validation and writes an HTML report.

```bash
python examples/model_comparison/run.py
```

Outputs go to `outputs/examples/model_comparison/` (open `report.html`). Equivalent CLI:
`geoxplain compare --data data/samples.csv --target target --features ndvi ndbi lst elevation rainfall distance_to_water population --crs EPSG:32645 --out-dir outputs/compare`.
