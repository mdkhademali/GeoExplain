## Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0], Initial Research Release

First public release of GeoExplain. This is an **initial, alpha-stage research release**; APIs may change. It has **not** been published to PyPI, conda-forge, the QGIS plugin repository, Zenodo or any other platform.

### Added
- Unified `GeoExplainModel` for Random Forest, XGBoost, LightGBM, MLP, logistic/ridge regression and SVM (classification and regression).
- Metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, MAE, MSE, RMSE, R², MAPE) with CSV and Excel export.
- Random and spatial hold-out, spatial block k-fold with buffer, Moran's I and semivariogram diagnostics, random-vs-spatial comparison.
- SHAP explanations (tree, linear, permutation), permutation importance, partial dependence/ICE, local explanations with numerical additivity check.
- Spatial SHAP: per-feature SHAP GeoTIFFs, `prediction.tif`, `uncertainty.tif` (entropy / tree spread where supported), dominant-driver raster; CRS, transform, size and nodata preserved.
- Cartographic maps and statistical figures (PNG 300 dpi, PDF, SVG); architecture and workflow diagrams; logo.
- End-to-end experiment workflow, self-contained HTML report, reproducibility record, `reproduce.py`.
- `geoxplain` CLI (`generate-data`, `train`, `evaluate`, `explain`, `spatial-explain`, `compare`, `run`, `report`, `info`).
- Synthetic example dataset generator, examples, notebook, documentation, pytest suite, GitHub Actions (tests, lint, packaging check).
- Experimental QGIS Processing provider (structure-tested against a stubbed QGIS API only).
