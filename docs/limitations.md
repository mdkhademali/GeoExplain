## Limitations

**Interpretation**
* SHAP values describe the fitted **model**: association and prediction, not causal effects. Do not read a large SHAP value as "changing this variable would change the outcome".
* Values depend on the background distribution and on feature correlation. Correlated predictors (typical in remote sensing) share credit in ways that can be unstable.
* Comparisons between models use relative importance because output spaces differ (probability vs log-odds).

**Spatial explainability**
* Spatial SHAP treats each cell independently; it does not model spatial dependence between cells.
* Explanations in areas far from the training data (extrapolation) reflect model behaviour there, not evidence. No area-of-applicability mask is provided in 0.1.0.
* Uncertainty layers are model-based spreads (entropy, tree spread), not calibrated intervals, and are unavailable for some model/task combinations.
* Scale-dependent (MAUP) effects: results depend on raster resolution.

**Validation**
* Block CV reduces but does not remove optimism; block size matters (see [spatial_validation.md](spatial_validation.md)). No hyper-parameter tuning module is included.
* Single synthetic example: the reported example numbers say nothing about real datasets.

**Software**
* Version 0.1.0 is an initial research release (alpha). APIs may change.
* Memory: rasters are processed as in-memory arrays (chunked SHAP, but the full stack is loaded). Very large rasters need tiling by the user.
* MLP/SVM spatial SHAP is slow for large grids.
* Only single-CRS, north-up-or-rotated affine grids are supported; no reprojection or resampling is done for you.
* The QGIS provider is untested inside a live QGIS (see [qgis.md](qgis.md)).
* Saved model files are joblib pickles and must only be loaded from trusted sources.
* Not published on PyPI, conda-forge, the QGIS repository or Zenodo.
