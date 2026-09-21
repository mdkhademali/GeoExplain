## GeoExplain QGIS Processing provider (experimental)

A lightweight QGIS Processing provider that exposes GeoExplain from the Processing Toolbox
(group **Explainable GeoAI**). The provider lives in `geoxplain_provider/` and is a thin wrapper:
all computation is done by the `geoxplain` Python package.

| Algorithm | What it does |
|---|---|
| Train Geospatial ML Model | Trains RF / XGBoost / LightGBM / MLP / LogReg / SVM from a layer's attribute table; logs a spatial hold-out estimate |
| Generate SHAP Explanation | Global SHAP, permutation importance, CSV tables and figures for point observations |
| Generate Spatial SHAP | Per-cell SHAP GeoTIFFs (e.g. `NDVI_SHAP.tif`), `prediction.tif`, `uncertainty.tif`, `dominant_driver.tif` |
| Compare Models | Random vs spatial validation for several models plus an HTML report |
| Generate Prediction Raster | Applies a trained model to a multi-band raster |

## Status and limitations

* **Not tested inside a live QGIS in the build environment.** The repository's test-suite validates the
  provider's structure statically (syntax, algorithm registry, metadata) and the underlying Python API
  end-to-end, but the QGIS runtime layer itself has not been exercised. Please report problems.
* Not published to the QGIS plugin repository.
* The core `geoxplain` package never imports QGIS; this folder is entirely optional.

## Installation (development)

1. Install GeoExplain **into the Python that QGIS uses**. On Linux/macOS this is usually the system or
   conda Python; on Windows use the *OSGeo4W Shell*:
   ```bash
   pip install geoxplain          # or: pip install -e /path/to/GeoExplain
   ```
   (`geoxplain` is not yet on PyPI; install from a clone or the release ZIP until then.)
2. Copy or symlink `geoxplain_provider/` into your QGIS plugin directory:
   * Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   * macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   * Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
3. In QGIS: *Plugins → Manage and Install Plugins → Installed → enable "GeoExplain"* (tick "Show experimental plugins" if needed).
4. Open the Processing Toolbox and look for **GeoExplain → Explainable GeoAI**.

## Notes

* Vector inputs use feature **centroids** as x/y. Use a projected CRS for meaningful spatial blocks.
* Multi-band rasters need band descriptions equal to the model's feature names, or provide the names in band order.
* Only open `.joblib` model files from sources you trust.
