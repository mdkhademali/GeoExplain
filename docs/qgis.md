## QGIS integration (optional, experimental)

GeoExplain ships a lightweight **QGIS Processing provider** in `qgis/processing/geoxplain_provider/`. It adds five algorithms to the Processing Toolbox (group *Explainable GeoAI*):

1. **Train Geospatial ML Model**
2. **Generate SHAP Explanation**
3. **Generate Spatial SHAP**
4. **Compare Models**
5. **Generate Prediction Raster**

Each is a thin wrapper around the public Python API. The core package does not import QGIS and works without it.

## Installation

See [`qgis/processing/README.md`](../qgis/processing/README.md): install `geoxplain` into the Python environment used by QGIS, copy `geoxplain_provider/` into the QGIS plugins directory, enable the plugin, then open the Processing Toolbox.

## Status

* Not published to the QGIS plugin repository.
* The test-suite exercises the provider against a **stubbed** `qgis` module (registration, parameters, and that each algorithm drives the real GeoExplain API correctly). It has **not** been run inside a live QGIS in the build environment, so treat it as experimental and please report issues.
* Vector inputs use feature centroids as coordinates; use a projected CRS for meaningful spatial blocks.

## Development tips

* Symlink the folder into your profile's `python/plugins` directory and use the *Plugin Reloader* plugin while iterating.
* Keep algorithms free of computation: add functionality to `geoxplain` first, then expose it.
