"""Structural and logic tests for the optional QGIS provider, using a stubbed ``qgis`` module.

A live QGIS is not available in CI, so the QGIS classes are replaced by minimal stand-ins. This
verifies registration, parameter declarations, and that each algorithm's ``processAlgorithm``
drives the real ``geoxplain`` API correctly. It does not replace testing inside QGIS.
"""

import ast
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio

PROVIDER_DIR = Path(__file__).resolve().parents[1] / "qgis" / "processing"
PKG = PROVIDER_DIR / "geoxplain_provider"


class _Meta(type):
    def __getattr__(cls, name):  # enum-like attributes (Numeric, Integer, ...)
        return name


class _Param(metaclass=_Meta):
    def __init__(self, name=None, description=None, *args, **kwargs):
        self.name, self.description, self.kwargs = name, description, kwargs


class _QgsException(Exception):
    pass


class _Algorithm:
    def __init__(self):
        self.params = {}

    def addParameter(self, p):  # noqa: N802
        self.params[p.name] = p

    # parameterAs* helpers read from the plain dict handed to processAlgorithm
    def parameterAsSource(self, p, n, c): return p[n]  # noqa: E704, N802
    def parameterAsString(self, p, n, c): return p.get(n, "")  # noqa: E704, N802
    def parameterAsFields(self, p, n, c): return p[n]  # noqa: E704, N802
    def parameterAsEnum(self, p, n, c): return p[n]  # noqa: E704, N802
    def parameterAsEnums(self, p, n, c): return p[n]  # noqa: E704, N802
    def parameterAsInt(self, p, n, c): return p[n]  # noqa: E704, N802
    def parameterAsFile(self, p, n, c): return p[n]  # noqa: E704, N802
    def parameterAsFileOutput(self, p, n, c): return p[n]  # noqa: E704, N802
    def parameterAsOutputLayer(self, p, n, c): return p[n]  # noqa: E704, N802
    def parameterAsRasterLayer(self, p, n, c): return p.get(n)  # noqa: E704, N802


class _Provider:
    def __init__(self):
        self.algorithms = []

    def addAlgorithm(self, a):  # noqa: N802
        self.algorithms.append(a)


@pytest.fixture()
def provider_pkg(monkeypatch):
    core = types.ModuleType("qgis.core")
    for n in ["QgsProcessingParameterEnum", "QgsProcessingParameterFeatureSource", "QgsProcessingParameterField",
              "QgsProcessingParameterFile", "QgsProcessingParameterFileDestination",
              "QgsProcessingParameterFolderDestination", "QgsProcessingParameterNumber",
              "QgsProcessingParameterRasterDestination", "QgsProcessingParameterRasterLayer",
              "QgsProcessingParameterString"]:
        setattr(core, n, type(n, (_Param,), {}))
    core.QgsProcessingAlgorithm = _Algorithm
    core.QgsProcessingException = _QgsException
    core.QgsProcessingProvider = _Provider
    core.QgsApplication = type("QgsApplication", (), {})
    qgis = types.ModuleType("qgis")
    qgis.core = core
    monkeypatch.setitem(sys.modules, "qgis", qgis)
    monkeypatch.setitem(sys.modules, "qgis.core", core)
    monkeypatch.syspath_prepend(str(PROVIDER_DIR))
    for m in [k for k in sys.modules if k.startswith("geoxplain_provider")]:
        monkeypatch.delitem(sys.modules, m)
    return importlib.import_module("geoxplain_provider.algorithms")


def test_all_provider_files_parse_and_metadata_is_complete():
    for f in PKG.glob("*.py"):
        ast.parse(f.read_text())
    meta = (PKG / "metadata.txt").read_text()
    for key in ["name=", "qgisMinimumVersion=", "version=", "author=", "hasProcessingProvider=yes", "experimental="]:
        assert key in meta
    assert (PKG / "icon.png").exists()


def test_provider_registers_five_uniquely_named_algorithms(provider_pkg):
    prov_mod = importlib.import_module("geoxplain_provider.provider")
    p = prov_mod.GeoExplainProvider()
    p.loadAlgorithms()
    names = [a.name() for a in p.algorithms]
    assert len(names) == 5 and len(set(names)) == 5
    assert {"train_model", "generate_shap_explanation", "generate_spatial_shap", "compare_models",
            "generate_prediction_raster"} == set(names)
    for a in p.algorithms:
        a.initAlgorithm()
        assert a.params and a.displayName() and a.shortHelpString() and a.group()
        assert type(a.createInstance()) is type(a)


class _Geom:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def isNull(self):  # noqa: N802
        return False

    def centroid(self):
        return self

    def asPoint(self):  # noqa: N802
        return types.SimpleNamespace(x=lambda: self._x, y=lambda: self._y)


class _Feat(dict):
    def __init__(self, i, row):
        super().__init__(row)
        self._i = i
        self._g = _Geom(row["x"], row["y"])

    def id(self):  # noqa: A003
        return self._i

    def geometry(self):
        return self._g


class _Source:
    def __init__(self, df): self.df = df  # noqa: E704
    def getFeatures(self): return (_Feat(i, r) for i, r in enumerate(self.df.to_dict("records")))  # noqa: E704, N802
    def featureCount(self): return len(self.df)  # noqa: E704, N802
    def sourceCrs(self): return types.SimpleNamespace(authid=lambda: "EPSG:32645")  # noqa: E704, N802


class _RasterLayer:
    def __init__(self, path): self._p = str(path)  # noqa: E704
    def source(self): return self._p  # noqa: E704


class _Fb:
    def __init__(self): self.msgs = []  # noqa: E704
    def pushInfo(self, m): self.msgs.append(m)  # noqa: E704, N802
    def setProgress(self, v): pass  # noqa: E704, N802


FEATS = ["ndvi", "ndbi", "lst", "elevation", "rainfall", "distance_to_water", "population"]


def test_train_then_spatial_shap_and_predict_via_algorithms(provider_pkg, data_dir, points, tmp_path):
    src = _Source(points[["id", "x", "y", *FEATS, "target"]].iloc[:300])
    fb = _Fb()
    model_path = str(tmp_path / "m.joblib")
    train = provider_pkg.TrainModelAlgorithm()
    out = train.processAlgorithm({"INPUT": src, "TARGET": "target", "FEATURES": FEATS, "TASK": 0, "ALGORITHM": 0,
                                  "VALIDATION": 0, "SEED": 1, "OUTPUT": model_path}, None, fb)
    assert Path(out["OUTPUT"]).exists() and any("hold-out" in m for m in fb.msgs)

    stack = data_dir / "stack.tif"
    sp = provider_pkg.SpatialShapAlgorithm().processAlgorithm(
        {"MODEL": model_path, "RASTER": _RasterLayer(stack), "BAND_NAMES": "", "OUTPUT_FOLDER": str(tmp_path / "sp")}, None, fb)
    assert (tmp_path / "sp" / "prediction.tif").exists() and "Elevation_SHAP" in sp
    with rasterio.open(stack) as a, rasterio.open(tmp_path / "sp" / "Elevation_SHAP.tif") as b:
        assert a.crs == b.crs and a.transform == b.transform and a.shape == b.shape

    pr = provider_pkg.PredictRasterAlgorithm().processAlgorithm(
        {"MODEL": model_path, "RASTER": _RasterLayer(stack), "BAND_NAMES": "", "OUTPUT": str(tmp_path / "p.tif")}, None, fb)
    with rasterio.open(pr["OUTPUT"]) as r:
        arr = r.read(1, masked=True)
        assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0 and np.ma.count(arr) > 0

    ex = provider_pkg.ExplainAlgorithm().processAlgorithm(
        {"MODEL": model_path, "INPUT": src, "TARGET": "target", "MAX_SAMPLES": 40, "OUTPUT_FOLDER": str(tmp_path / "ex")}, None, fb)
    assert (Path(ex["OUTPUT_FOLDER"]) / "shap_global.csv").exists()


def test_missing_geoxplain_gives_install_hint(provider_pkg, monkeypatch):
    helpers = importlib.import_module("geoxplain_provider.helpers")
    monkeypatch.setitem(sys.modules, "geoxplain", None)
    with pytest.raises(_QgsException) as exc:
        helpers.import_geoxplain()
    assert "pip install" in str(exc.value)
