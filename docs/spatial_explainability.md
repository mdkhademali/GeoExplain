## Spatial explainability

> Standard explainability says **which** variables matter. GeoExplain also shows **where** they matter.

## Concept

Every valid grid cell of a predictor raster stack is treated as one observation whose feature vector is read from the co-registered rasters. SHAP values are computed for every cell, and the contribution φᵢ of feature *i* is written back to the *same* cell of a new raster. A map of φᵢ therefore shows where the fitted model relies on feature *i* to raise (positive) or lower (negative) the prediction relative to the base value. The same feature value can contribute differently in different places because the model's response also depends on the other features at that location.

```python
from geoxplain import compute_spatial_shap, read_raster_stack
stack = read_raster_stack("data/stack.tif")                    # band descriptions = feature names
result = compute_spatial_shap(model, stack, background=X_background)
written = result.write("outputs/rasters")
```

Single-band files work too: `read_raster_stack({"ndvi": "ndvi.tif", ...})` or `read_raster_dir(folder, names)`.

## Outputs

| File | Content |
|---|---|
| `<Label>_SHAP.tif` | one per feature, e.g. `NDVI_SHAP.tif`, `NDBI_SHAP.tif`, `LST_SHAP.tif`, `Elevation_SHAP.tif`, `Rainfall_SHAP.tif`, `DistanceToWater_SHAP.tif`; the label comes from `feature_labels` (unknown names are used as they are) |
| `SHAP_sum.tif` | Σ of the feature contributions (prediction − base value) |
| `prediction.tif` | class probability (classification, positive/selected class) or predicted value |
| `prediction_class.tif` | predicted class index (classification) |
| `uncertainty.tif` | predictive uncertainty, when supported (see below) |
| `dominant_driver.tif` | index of the feature with the largest |SHAP| per cell (legend in the metadata JSON) |
| `spatial_shap_metadata.json` | base value, output space, model config, grid description, additivity check, notes |

All rasters share the **CRS, affine transform, width, height and nodata handling** of the input grid (float32, nodata −9999; the dominant-driver raster is int16 with nodata −1). Cells where any feature is NaN/nodata are excluded and written as nodata. Misaligned input rasters raise `SpatialAlignmentError`; GeoExplain never resamples or reprojects silently.

## Uncertainty

* Classification (any model with probabilities): normalised Shannon entropy of the predicted class probabilities (0 = certain, 1 = uniform).
* Regression with Random Forest: standard deviation of the individual tree predictions.
* Other combinations (e.g. XGBoost regression) have no built-in estimate; `uncertainty.tif` is then **not written** and a note is recorded rather than inventing a number.

These are model-based spreads, **not calibrated confidence intervals**, and they ignore uncertainty in the predictors and in spatial extrapolation.

## Maps

`viz.plot_shap_map(result, "elevation")` (one feature), `viz.plot_shap_panels(result)` (all features, one shared zero-centred colour scale), `viz.plot_dominant_driver`, `viz.plot_prediction_uncertainty`, `viz.plot_feature_and_shap`, and `viz.plot_map` for any raster. Each map has a title, colour bar, north arrow (omitted for rotated grids), scale bar, labelled coordinate axes, and a footnote with CRS and data source. SHAP maps use the diverging `RdBu_r` scale symmetric about zero (red raises the prediction, blue lowers it). Never interpolate SHAP values between points to make a map; compute them at each cell as done here.

## Cost and practical notes

* Tree models and linear models are exact and fast. MLP/SVM use the permutation explainer, which costs many model evaluations per cell; expect minutes for large grids and consider explaining a coarser raster.
* Explanations are computed in chunks (`chunk_size`) to bound memory.
* `zonal_summary(result, polygons, id_col)` aggregates contributions per polygon (polygons must be in the raster's CRS).
* Interpretation: SHAP maps describe the **model**; they are associations, not causal effects. Correlated features share credit unpredictably; check dependence plots and consider domain knowledge.
