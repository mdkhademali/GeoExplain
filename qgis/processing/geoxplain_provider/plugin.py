"""Plugin class registering the GeoExplain Processing provider."""

from qgis.core import QgsApplication

from .provider import GeoExplainProvider


class GeoExplainPlugin:
    """Registers / unregisters :class:`GeoExplainProvider` with the Processing framework."""

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):  # noqa: N802
        self.provider = GeoExplainProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):  # noqa: N802
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
