"""Processing provider listing the GeoExplain algorithms."""

from qgis.core import QgsProcessingProvider

from .algorithms import (
    CompareModelsAlgorithm,
    ExplainAlgorithm,
    PredictRasterAlgorithm,
    SpatialShapAlgorithm,
    TrainModelAlgorithm,
)


class GeoExplainProvider(QgsProcessingProvider):
    """Provider ``geoxplain`` (group: Explainable GeoAI)."""

    def loadAlgorithms(self):  # noqa: N802
        for algorithm in (TrainModelAlgorithm, ExplainAlgorithm, SpatialShapAlgorithm, CompareModelsAlgorithm,
                          PredictRasterAlgorithm):
            self.addAlgorithm(algorithm())

    def id(self):  # noqa: A003
        return "geoxplain"

    def name(self):
        return "GeoExplain"

    def longName(self):  # noqa: N802
        return "GeoExplain: explainable ML for geospatial prediction"
