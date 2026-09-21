"""Spatial explainability: prediction, uncertainty and spatial SHAP rasters."""

from .predict import RasterPrediction, model_stack, predict_raster
from .spatial_shap import SpatialShapResult, compute_spatial_shap, layer_stem, zonal_summary
from .uncertainty import UncertaintyResult, predictive_uncertainty

__all__ = [
    "RasterPrediction",
    "SpatialShapResult",
    "UncertaintyResult",
    "compute_spatial_shap",
    "layer_stem",
    "model_stack",
    "predict_raster",
    "predictive_uncertainty",
    "zonal_summary",
]
