"""QGIS plugin entry point for the GeoExplain Processing provider."""


def classFactory(iface):  # noqa: N802 - name required by QGIS
    """Load the plugin (called by QGIS)."""
    from .plugin import GeoExplainPlugin

    return GeoExplainPlugin(iface)
