## Installation

GeoExplain needs Python 3.10 or newer. The package is **not yet published on PyPI or conda-forge**; install it from a clone or from the release ZIP.

## From a clone / ZIP

```bash
git clone https://github.com/mdkhademali/GeoExplain.git      # or unzip GeoExplain_v0.1.0.zip
cd GeoExplain
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                       # add ,notebooks for Jupyter support
geoxplain --version
```

`pip install -e .` installs the dependencies listed in `pyproject.toml`: NumPy, pandas, SciPy, scikit-learn, matplotlib, SHAP, XGBoost, LightGBM, rasterio, GeoPandas, Shapely, pyproj, joblib, openpyxl.

## Geospatial libraries

`rasterio` and `geopandas` ship binary wheels for Linux, macOS and Windows on current Python versions, so plain `pip` normally works. If you have trouble, conda is the most reliable route:

```bash
conda create -n geoxplain -c conda-forge python=3.12 rasterio geopandas shap xgboost lightgbm scikit-learn matplotlib openpyxl
conda activate geoxplain && pip install -e . --no-deps
```

## Optional dependencies

XGBoost and LightGBM are installed by default. If one is missing, GeoExplain still imports; requesting that model raises an `OptionalDependencyError` with the install command, and `geoxplain info` shows what is available.

QGIS support is a separate, optional plugin folder (see [qgis.md](qgis.md)); the core package never imports QGIS.

## Verify

```bash
pytest                       # full test-suite on small synthetic data
python reproduce.py --quick  # about half a minute; regenerates data, results and figures
```
