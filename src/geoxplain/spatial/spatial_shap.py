"""Spatial SHAP: per-cell feature contributions written as georeferenced layers.

Concept
-------
Every valid raster cell is treated as one observation whose feature vector is read from
the co-registered predictor rasters. SHAP values are computed for each cell, and the
contribution ``phi_i`` of feature ``i`` is written back to the *same cell* of a new grid,
so the output rasters share CRS, transform, width, height and nodata handling with the
inputs. Maps of ``phi_i`` therefore show **where** the fitted model relies on feature
``i`` to raise (positive) or lower (negative) the prediction relative to the base value.

Interpretation caveats: SHAP values describe the model, not causal effects; they depend
on the model, the background/perturbation scheme and feature correlations.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._version import __version__
from ..explainers.shap_explainer import GeoShapExplainer, ShapValues
from ..io.raster import RasterStack, write_raster
from ..models.model import GeoExplainModel
from ..utils.exceptions import NotSupportedError, SpatialAlignmentError
from ..utils.optional import require
from ..utils.serialization import write_json
from .predict import model_stack
from .uncertainty import predictive_uncertainty


def layer_stem(name: str, labels: Mapping[str, str] | None = None) -> str:
    """Filesystem-safe stem for a feature's SHAP raster (``NDVI`` -> ``NDVI_SHAP``)."""
    label = (labels or {}).get(name, name)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)) + "_SHAP"


@dataclass
class SpatialShapResult:
    """Result of :func:`compute_spatial_shap` (all grids are ``(height, width)``)."""

    stack: RasterStack
    shap: ShapValues
    layers: dict[str, np.ndarray]
    total: np.ndarray
    dominant: np.ndarray  # int16 feature index of max |SHAP|; -1 outside valid area
    prediction: np.ndarray | None
    predicted_class: np.ndarray | None
    uncertainty: np.ndarray | None
    uncertainty_method: str | None
    uncertainty_description: str | None
    feature_labels: dict[str, str]
    model_config: dict[str, Any]
    additivity: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return list(self.shap.feature_names)

    def summary_table(self) -> pd.DataFrame:
        """Per-feature statistics of the SHAP layers over all valid cells."""
        dom_share = np.bincount(
            self.dominant[self.dominant >= 0].astype(int), minlength=len(self.feature_names)
        ) / max(1, int((self.dominant >= 0).sum()))
        rows = []
        for i, name in enumerate(self.feature_names):
            v = self.shap.values[:, i]
            rows.append(
                {
                    "feature": name,
                    "mean_shap": float(v.mean()),
                    "std_shap": float(v.std()),
                    "min_shap": float(v.min()),
                    "max_shap": float(v.max()),
                    "mean_abs_shap": float(np.abs(v).mean()),
                    "share_cells_positive": float((v > 0).mean()),
                    "share_cells_dominant": float(dom_share[i]),
                }
            )
        return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def metadata(self) -> dict[str, Any]:
        """JSON-friendly description of the outputs."""
        return {
            "geoxplain_version": __version__,
            "model": self.model_config,
            "grid": self.stack.describe(),
            "base_value": self.shap.base_value,
            "output_space": self.shap.output_space,
            "explained_class": None if self.shap.class_label is None else str(self.shap.class_label),
            "shap_method": self.shap.method,
            "feature_order": self.feature_names,
            "dominant_driver_legend": {int(i): n for i, n in enumerate(self.feature_names)},
            "uncertainty_method": self.uncertainty_method,
            "uncertainty_description": self.uncertainty_description,
            "additivity_check": self.additivity,
            "notes": self.notes,
            "interpretation": (
                "SHAP values describe how the fitted model uses each feature. They quantify "
                "association captured by the model, not causal effects."
            ),
        }

    def write(self, out_dir: str | Path, write_extras: bool = True) -> dict[str, Path]:
        """Write GeoTIFFs and a metadata JSON to ``out_dir``.

        Files: ``<Label>_SHAP.tif`` per feature, ``SHAP_sum.tif``, ``dominant_driver.tif``,
        ``prediction.tif``, ``prediction_class.tif`` (classification), ``uncertainty.tif``
        (if supported) and ``spatial_shap_metadata.json``.
        """
        out = Path(out_dir)
        tags = {
            "GEOXPLAIN_VERSION": __version__,
            "OUTPUT_SPACE": self.shap.output_space,
            "BASE_VALUE": f"{self.shap.base_value:.8g}",
            "EXPLAINED_CLASS": str(self.shap.class_label),
            "SHAP_METHOD": self.shap.method,
        }
        paths: dict[str, Path] = {}
        for name in self.feature_names:
            stem = layer_stem(name, self.feature_labels)
            paths[stem] = write_raster(
                out / f"{stem}.tif", self.layers[name], self.stack,
                descriptions=[f"SHAP contribution of {name}"], tags={**tags, "FEATURE": name},
            )
        if write_extras:
            paths["SHAP_sum"] = write_raster(
                out / "SHAP_sum.tif", self.total, self.stack,
                descriptions=["Sum of SHAP values (prediction - base value)"], tags=tags,
            )
            paths["dominant_driver"] = write_raster(
                out / "dominant_driver.tif", self.dominant, self.stack, nodata=-1, dtype="int16",
                descriptions=["Index of feature with largest |SHAP| (see metadata legend)"], tags=tags,
            )
            if self.prediction is not None:
                paths["prediction"] = write_raster(
                    out / "prediction.tif", self.prediction, self.stack,
                    descriptions=["Predicted probability of explained class / predicted value"], tags=tags,
                )
            if self.predicted_class is not None:
                paths["prediction_class"] = write_raster(
                    out / "prediction_class.tif", self.predicted_class, self.stack, nodata=-1,
                    dtype="int16", descriptions=["Predicted class index (see metadata)"], tags=tags,
                )
            if self.uncertainty is not None:
                paths["uncertainty"] = write_raster(
                    out / "uncertainty.tif", self.uncertainty, self.stack,
                    descriptions=[f"Predictive uncertainty ({self.uncertainty_method})"],
                    tags={**tags, "UNCERTAINTY_METHOD": str(self.uncertainty_method)},
                )
        paths["metadata"] = write_json(out / "spatial_shap_metadata.json", self.metadata())
        return paths


def compute_spatial_shap(
    model: GeoExplainModel,
    stack: RasterStack,
    background: pd.DataFrame | None = None,
    class_label: Any = None,
    method: str = "auto",
    chunk_size: int = 5000,
    include_prediction: bool = True,
    include_uncertainty: bool = True,
    uncertainty_method: str = "auto",
    feature_labels: Mapping[str, str] | None = None,
    explainer: GeoShapExplainer | None = None,
    additivity_sample: int = 2000,
    random_state: int = 42,
    **explainer_kwargs: Any,
) -> SpatialShapResult:
    """Compute SHAP values for every valid cell of ``stack`` and map them back to the grid.

    Parameters
    ----------
    model:
        Fitted model; its ``feature_names_`` must all be bands of ``stack``.
    stack:
        Co-registered predictor rasters (never resampled). Cells with NaN in any *model*
        band are treated as nodata and excluded.
    background:
        Reference feature table (training data) for linear / model-agnostic explainers.
    class_label:
        Class whose SHAP values are mapped (classification).
    include_prediction, include_uncertainty:
        Also compute a prediction grid and (if supported) an uncertainty grid.
    additivity_sample:
        Number of random cells used to verify ``base + sum(SHAP) == model output``.

    Notes
    -----
    Computation cost grows with the number of cells; tree models are fastest. Every valid
    cell is explained - GeoExplain does not silently subsample or resample the grid.
    """
    ms = model_stack(model, stack)
    if not ms.is_north_up:
        warnings.warn(
            "The raster transform is rotated or not north-up. Outputs keep the same transform; "
            "only map plotting assumes north-up grids.",
            stacklevel=2,
        )
    table = ms.to_dataframe()
    if table.empty:
        raise SpatialAlignmentError("No valid cells: every cell is nodata in at least one band.")
    feats = table[model.feature_names_]
    ex = explainer or GeoShapExplainer(
        model, background=background, method=method, class_label=class_label,
        random_state=random_state, **explainer_kwargs,
    )
    sv = ex.explain(feats, chunk_size=chunk_size)
    sv.extra["rows"], sv.extra["cols"] = table["row"].to_numpy(), table["col"].to_numpy()

    layers = {name: ms.scatter(sv.values[:, i]) for i, name in enumerate(sv.feature_names)}
    total = ms.scatter(sv.values.sum(axis=1))
    dominant = ms.scatter(np.argmax(np.abs(sv.values), axis=1), fill=-1, dtype=np.int16)

    # additivity verification on a random sample of cells
    n_check = min(additivity_sample, len(feats))
    idx = np.random.default_rng(random_state).choice(len(feats), n_check, replace=False)
    additivity = ex.check_additivity(sv.subset(idx))
    notes: list[str] = []
    if not additivity["ok"]:
        notes.append("Additivity check exceeded tolerance; inspect SHAP settings.")

    prediction = predicted_class = None
    if include_prediction:
        if model.task == "classification":
            proba = model.predict_proba(feats)
            prediction = ms.scatter(proba[:, ex.class_idx])
            predicted_class = ms.scatter(np.argmax(proba, axis=1), fill=-1, dtype=np.int16)
        else:
            prediction = ms.scatter(model.predict(feats))

    uncertainty = u_method = u_desc = None
    if include_uncertainty:
        try:
            u = predictive_uncertainty(model, feats, uncertainty_method, class_label)
            uncertainty, u_method, u_desc = ms.scatter(u.values), u.method, u.description
        except NotSupportedError as exc:
            notes.append(f"Uncertainty unavailable: {exc}")

    return SpatialShapResult(
        stack=ms, shap=sv, layers=layers, total=total, dominant=dominant, prediction=prediction,
        predicted_class=predicted_class, uncertainty=uncertainty, uncertainty_method=u_method,
        uncertainty_description=u_desc, feature_labels=dict(feature_labels or {}),
        model_config=model.get_config(), additivity=additivity, notes=notes,
    )


def zonal_summary(result: SpatialShapResult, zones: Any, id_col: str) -> pd.DataFrame:
    """Mean SHAP value of each feature inside polygon zones.

    Parameters
    ----------
    result:
        Output of :func:`compute_spatial_shap`.
    zones:
        ``geopandas.GeoDataFrame`` of polygons **in the same CRS as the rasters** (no
        implicit reprojection is done).
    id_col:
        Column identifying each zone.

    Returns a DataFrame with one row per zone: ``n_cells`` plus ``mean_shap__<feature>``
    and ``mean_abs_shap__<feature>`` columns.
    """
    features = require("rasterio.features", "zonal statistics")
    stack = result.stack
    zones_crs = getattr(zones, "crs", None)
    if stack.crs is not None and zones_crs is not None and zones_crs != stack.crs:
        raise SpatialAlignmentError(
            f"Zone CRS ({zones_crs}) differs from raster CRS ({stack.crs}); reproject the zones first."
        )
    zone_ids = list(zones[id_col])
    rows = []
    for zid, geom in zip(zone_ids, zones.geometry, strict=False):
        mask = features.rasterize(
            [(geom, 1)], out_shape=stack.shape, transform=stack.transform, fill=0, dtype="uint8"
        ).astype(bool)
        row: dict[str, Any] = {id_col: zid}
        valid = mask & stack.valid_mask
        row["n_cells"] = int(valid.sum())
        for name in result.feature_names:
            layer = result.layers[name][valid]
            row[f"mean_shap__{name}"] = float(layer.mean()) if layer.size else float("nan")
            row[f"mean_abs_shap__{name}"] = float(np.abs(layer).mean()) if layer.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
