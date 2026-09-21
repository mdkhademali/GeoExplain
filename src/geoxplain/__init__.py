"""GeoExplain: explainable machine learning for geospatial prediction.

Traditional explainability tells you *which* variables matter. GeoExplain additionally shows
*where* they matter, by computing SHAP contributions per observation / grid cell and writing
them back to georeferenced rasters.

The most commonly used entry points are re-exported here; see the sub-packages for the rest.
"""

from ._version import __version__
from .datasets import FEATURES, make_synthetic_geodata, sample_points, write_synthetic_dataset
from .explainers import GeoShapExplainer, ShapValues, partial_dependence, permutation_importance
from .io import GeoDataset, RasterStack, load_dataset, read_raster_stack, write_raster
from .models import GeoExplainModel
from .spatial import SpatialShapResult, compute_spatial_shap, predict_raster
from .validation import SpatialBlockKFold, compare_validation_strategies, spatial_split
from .workflows import ExperimentConfig, build_report, run_experiment

__all__ = [
    "FEATURES",
    "ExperimentConfig",
    "GeoDataset",
    "GeoExplainModel",
    "GeoShapExplainer",
    "RasterStack",
    "ShapValues",
    "SpatialBlockKFold",
    "SpatialShapResult",
    "__version__",
    "build_report",
    "compare_validation_strategies",
    "compute_spatial_shap",
    "load_dataset",
    "make_synthetic_geodata",
    "partial_dependence",
    "permutation_importance",
    "predict_raster",
    "read_raster_stack",
    "run_experiment",
    "sample_points",
    "spatial_split",
    "write_raster",
    "write_synthetic_dataset",
]
