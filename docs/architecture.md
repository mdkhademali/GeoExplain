## Architecture

![Architecture](../figures/architecture.png)

The package `geoxplain` (in `src/geoxplain/`) is layered; upper layers depend on lower ones only.

| Sub-package | Responsibility | Main public objects |
|---|---|---|
| `utils` | exceptions, optional-dependency handling, seeding, environment capture, JSON serialisation | `GeoExplainError`, `require`, `environment_info`, `write_json` |
| `io` | tabular datasets and raster stacks; CRS-preserving raster writing | `GeoDataset`, `load_dataset`, `RasterStack`, `read_raster_stack`, `write_raster` |
| `datasets` | reproducible **synthetic** scenes for examples and tests | `make_synthetic_geodata`, `sample_points`, `write_synthetic_dataset` |
| `models` | one interface over six algorithms, classification and regression | `GeoExplainModel`, `resolve_algorithm` |
| `metrics` | classification/regression metrics, metric tables, CSV/Excel export | `classification_metrics`, `regression_metrics`, `export_metrics` |
| `validation` | random and spatial splits, spatial block CV, autocorrelation diagnostics | `spatial_split`, `SpatialBlockKFold`, `compare_validation_strategies`, `morans_i` |
| `explainers` | SHAP, permutation importance, partial dependence | `GeoShapExplainer`, `ShapValues`, `permutation_importance`, `partial_dependence` |
| `spatial` | prediction rasters, uncertainty, **spatial SHAP** | `compute_spatial_shap`, `SpatialShapResult`, `predict_raster` |
| `visualization` | academic style, cartographic maps, statistical plots, diagrams | `plot_shap_map`, `plot_shap_panels`, `plot_map`, `draw_architecture` |
| `workflows` | end-to-end experiment and HTML report | `ExperimentConfig`, `run_experiment`, `build_report` |
| `cli` | `geoxplain` command | `main` |

## Design decisions

* **One model interface.** `GeoExplainModel` wraps a scikit-learn-compatible estimator. Scale-sensitive models (MLP, logistic regression, SVM) are wrapped in a `StandardScaler` pipeline automatically. Feature order is enforced by name at prediction time.
* **Explanations in native model space.** Tree models are explained with `TreeExplainer`: Random Forest in probability units, XGBoost/LightGBM in log-odds (classification). Logistic regression uses `LinearExplainer` (log-odds). MLP and SVM use the model-agnostic permutation explainer on probabilities. `ShapValues.output_space` always says which; comparisons between models therefore use *relative* importance shares.
* **No silent spatial changes.** Raster stacks must share CRS, transform and shape (`SpatialAlignmentError` otherwise). Outputs reuse the input grid. Cells with nodata in any feature are excluded and written back as nodata.
* **Optional dependencies fail softly.** Missing XGBoost/LightGBM/SHAP raise a descriptive `OptionalDependencyError` at use time, not at import time.
* **Reproducibility by construction.** Every workflow takes a seed, and `run_experiment` records environment, parameters and splits in `reproducibility.json`.
* **QGIS is a thin optional wrapper** in `qgis/processing/`, never imported by the core package.

## Data flow of `run_experiment`

1. load and validate the table → 2. random and spatial hold-out for every model → 3. random vs spatial k-fold CV → 4. comparison figures on the spatial hold-out → 5. permutation importance (spatial hold-out) → 6. final model refit on all observations, global/local SHAP, PDP → 7. spatial SHAP rasters and maps → 8. tables (CSV + Excel), `experiment.json`, `reproducibility.json`, HTML report.
