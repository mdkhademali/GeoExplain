"""Helpers for optional dependencies with graceful, actionable error messages."""

from __future__ import annotations

import importlib
from types import ModuleType

from .exceptions import OptionalDependencyError

# Maps import name -> installation hint.
_INSTALL_HINTS = {
    "xgboost": "pip install xgboost   (or: pip install 'geoxplain[boost]')",
    "lightgbm": "pip install lightgbm   (or: pip install 'geoxplain[boost]')",
    "shap": "pip install shap",
    "rasterio": "pip install rasterio",
    "geopandas": "pip install geopandas",
    "openpyxl": "pip install openpyxl",
    "qgis": "QGIS must be installed separately; run inside the QGIS Python environment",
}


def is_available(module_name: str) -> bool:
    """Return ``True`` if ``module_name`` can be imported."""
    try:
        importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - any import-time failure means "unavailable"
        return False
    return True


def require(module_name: str, purpose: str = "") -> ModuleType:
    """Import and return ``module_name`` or raise :class:`OptionalDependencyError`.

    Parameters
    ----------
    module_name:
        Importable module name, e.g. ``"xgboost"``.
    purpose:
        Short description of what the module is needed for; included in the message.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        hint = _INSTALL_HINTS.get(module_name, f"pip install {module_name}")
        why = f" for {purpose}" if purpose else ""
        raise OptionalDependencyError(
            f"The optional dependency '{module_name}' is required{why} but could not be "
            f"imported ({exc.__class__.__name__}: {exc}). Install it with: {hint}"
        ) from exc
