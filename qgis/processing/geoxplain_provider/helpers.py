"""Helpers that convert QGIS objects to the plain pandas/GeoTIFF inputs GeoExplain expects."""

import pandas as pd
from qgis.core import QgsProcessingException

INSTALL_HINT = (
    "The 'geoxplain' Python package is not available in QGIS's Python environment. "
    "Install it (e.g. `pip install geoxplain` or `pip install -e /path/to/GeoExplain`) using the "
    "same interpreter that QGIS uses, then restart QGIS."
)


def import_geoxplain():
    """Import geoxplain or raise a Processing exception with installation guidance."""
    try:
        import geoxplain

        return geoxplain
    except ImportError as exc:  # pragma: no cover - depends on the user's QGIS environment
        raise QgsProcessingException(INSTALL_HINT) from exc


def layer_to_dataframe(layer, fields, feedback=None):
    """Convert a vector layer or Processing feature source (points/polygons) to a DataFrame with ``x``/``y`` (centroid) columns.

    Features with null values in ``fields`` are kept (GeoExplain's loader drops or reports them).
    """
    rows = []
    total = layer.featureCount() or 1
    for i, feat in enumerate(layer.getFeatures()):
        geom = feat.geometry()
        if geom is None or geom.isNull():
            continue
        c = geom.centroid().asPoint()
        row = {"id": feat.id(), "x": c.x(), "y": c.y()}
        for f in fields:
            value = feat[f]
            row[f] = None if value is None or str(value) == "NULL" else value
        rows.append(row)
        if feedback is not None and i % 500 == 0:
            feedback.setProgress(int(100 * i / total))
    if not rows:
        raise QgsProcessingException("The input layer has no usable features.")
    df = pd.DataFrame(rows)
    for f in fields:
        try:
            df[f] = pd.to_numeric(df[f])
        except (ValueError, TypeError):
            pass  # keep non-numeric columns (e.g. class labels) as they are
    return df
