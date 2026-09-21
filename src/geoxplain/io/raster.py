"""Raster stacks: reading aligned rasters, converting to tables and writing GeoTIFFs.

GeoExplain never resamples or reprojects silently. All rasters in a stack must share the
same CRS, affine transform and shape; otherwise :class:`SpatialAlignmentError` is raised.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.exceptions import DataValidationError, SpatialAlignmentError
from ..utils.optional import require

DEFAULT_NODATA = -9999.0


@dataclass
class RasterStack:
    """A set of co-registered single-band rasters held in memory.

    Attributes
    ----------
    data:
        Array of shape ``(bands, height, width)``; nodata cells are ``NaN``.
    names:
        One name per band.
    crs:
        ``rasterio.crs.CRS`` (or ``None`` when the source had no CRS).
    transform:
        ``affine.Affine`` transform of the grid.
    nodata:
        Nodata value used when *writing* outputs derived from this stack.
    """

    data: np.ndarray
    names: list[str]
    crs: Any
    transform: Any
    nodata: float = DEFAULT_NODATA
    sources: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise DataValidationError("RasterStack.data must have shape (bands, height, width)")
        if self.data.shape[0] != len(self.names):
            raise DataValidationError("Number of names must equal number of bands")
        if len(set(self.names)) != len(self.names):
            raise DataValidationError("Band names must be unique")

    # ------------------------------------------------------------------ geometry
    @property
    def count(self) -> int:
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        return int(self.data.shape[2])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` of the grid."""
        t = self.transform
        # Affine is applied explicitly (a*x + b*y + c, d*x + e*y + f) so this works with all affine versions.
        def apply(col: float, row: float) -> tuple[float, float]:
            return t.a * col + t.b * row + t.c, t.d * col + t.e * row + t.f

        left, top = apply(0, 0)
        right, bottom = apply(self.width, self.height)
        return (min(left, right), min(bottom, top), max(left, right), max(bottom, top))

    @property
    def is_north_up(self) -> bool:
        """``True`` if the transform has no rotation/shear and a negative y pixel size."""
        t = self.transform
        return bool(np.isclose(t.b, 0) and np.isclose(t.d, 0) and t.e < 0 and t.a > 0)

    @property
    def crs_string(self) -> str | None:
        if self.crs is None:
            return None
        try:
            epsg = self.crs.to_epsg()
            return f"EPSG:{epsg}" if epsg else self.crs.to_string()
        except Exception:  # noqa: BLE001
            return str(self.crs)

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean ``(height, width)`` mask of cells finite in *all* bands."""
        return np.all(np.isfinite(self.data), axis=0)

    def pixel_centers(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map-coordinates of pixel centres for integer ``rows`` and ``cols``."""
        t = self.transform
        r = np.asarray(rows, dtype=float) + 0.5
        c = np.asarray(cols, dtype=float) + 0.5
        return t.a * c + t.b * r + t.c, t.d * c + t.e * r + t.f

    # ---------------------------------------------------------------- conversion
    def to_dataframe(self, dropna: bool = True) -> pd.DataFrame:
        """Flatten the stack to a table (``row``, ``col``, ``x``, ``y`` + one column per band).

        Rows are in C order (row-major). With ``dropna=True`` only cells valid in all
        bands are returned, and :meth:`scatter` maps per-row values back onto the grid.
        """
        mask = self.valid_mask if dropna else np.ones(self.shape, dtype=bool)
        rows, cols = np.nonzero(mask)
        xs, ys = self.pixel_centers(rows, cols)
        frame = pd.DataFrame({"row": rows, "col": cols, "x": xs, "y": ys})
        for i, name in enumerate(self.names):
            frame[name] = self.data[i][rows, cols].astype(float)
        return frame

    def scatter(self, values: np.ndarray, fill: float = np.nan, dtype: Any = np.float32) -> np.ndarray:
        """Place one value per *valid* cell (C order) back on a ``(height, width)`` grid."""
        mask = self.valid_mask
        values = np.asarray(values)
        n_valid = int(mask.sum())
        if values.shape[0] != n_valid:
            raise DataValidationError(
                f"Expected {n_valid} values (one per valid cell), got {values.shape[0]}."
            )
        out = np.full(self.shape, fill, dtype=dtype)
        out[mask] = values
        return out

    def subset_bands(self, names: Sequence[str]) -> "RasterStack":
        """Return a stack with the requested bands in the requested order."""
        missing = [n for n in names if n not in self.names]
        if missing:
            raise DataValidationError(f"Bands not in stack: {missing}. Available: {self.names}")
        idx = [self.names.index(n) for n in names]
        return RasterStack(
            self.data[idx], list(names), self.crs, self.transform, self.nodata,
            {n: self.sources.get(n, "") for n in names},
        )

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        crs: Any,
        transform: Any,
        nodata: float = DEFAULT_NODATA,
    ) -> "RasterStack":
        """Build a stack from a ``{name: 2-D array}`` mapping (NaN = nodata)."""
        names = list(arrays)
        if not names:
            raise DataValidationError("At least one array is required.")
        shapes = {np.asarray(a).shape for a in arrays.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise SpatialAlignmentError(f"All arrays must be 2-D with equal shape; got {shapes}")
        data = np.stack([np.asarray(arrays[n], dtype=np.float32) for n in names])
        return cls(data, names, _coerce_crs(crs), transform, nodata)

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description of the grid."""
        left, bottom, right, top = self.bounds
        return {
            "bands": self.names,
            "height": self.height,
            "width": self.width,
            "crs": self.crs_string,
            "transform": list(self.transform)[:6],
            "bounds": [left, bottom, right, top],
            "pixel_size": [abs(self.transform.a), abs(self.transform.e)],
            "n_valid_cells": int(self.valid_mask.sum()),
            "nodata_output": self.nodata,
        }


def _coerce_crs(crs: Any) -> Any:
    """Return a ``rasterio.crs.CRS`` from a string / EPSG int / CRS / ``None``."""
    if crs is None:
        return None
    rio_crs = require("rasterio.crs", "handling CRS")
    if isinstance(crs, rio_crs.CRS):
        return crs
    if isinstance(crs, int):
        return rio_crs.CRS.from_epsg(crs)
    return rio_crs.CRS.from_string(str(crs))


def _read_band(path: str | Path, band: int = 1) -> tuple[np.ndarray, Any, Any, float | None]:
    rasterio = require("rasterio", "reading rasters")
    with rasterio.open(path) as src:
        masked = src.read(band, masked=True)
        arr = np.ma.filled(masked.astype("float32"), np.nan)
        return arr, src.crs, src.transform, src.nodata


def read_raster_stack(
    sources: Mapping[str, str | Path] | Sequence[str | Path] | str | Path,
    names: Sequence[str] | None = None,
    nodata: float | None = None,
    atol: float = 1e-6,
) -> RasterStack:
    """Read co-registered rasters into a :class:`RasterStack`.

    Parameters
    ----------
    sources:
        * a mapping ``{name: path}`` of single-band rasters,
        * a sequence of paths (names default to the file stems), or
        * one multi-band path (names from ``names`` or band descriptions).
    names:
        Optional band names (sequence inputs and multi-band files).
    nodata:
        Nodata value used for *outputs*. Defaults to the first raster's nodata value if
        set, otherwise ``-9999``.
    atol:
        Absolute tolerance when comparing affine transforms.

    Raises
    ------
    SpatialAlignmentError
        If CRS, transform or shape differ between rasters. No resampling is attempted.
    """
    rasterio = require("rasterio", "reading rasters")

    # --- multi-band single file ------------------------------------------------
    if isinstance(sources, (str, Path)):
        path = Path(sources)
        with rasterio.open(path) as src:
            count = src.count
            arr = np.ma.filled(src.read(masked=True).astype("float32"), np.nan)
            band_names = list(names) if names else [
                (d or f"band_{i + 1}") for i, d in enumerate(src.descriptions)
            ]
            if len(band_names) != count:
                raise DataValidationError(
                    f"{count} bands found but {len(band_names)} names supplied/derived."
                )
            out_nodata = nodata if nodata is not None else (
                src.nodata if src.nodata is not None and np.isfinite(src.nodata) else DEFAULT_NODATA
            )
            return RasterStack(arr, band_names, src.crs, src.transform, float(out_nodata),
                               {n: str(path) for n in band_names})

    # --- several single-band files ---------------------------------------------
    if isinstance(sources, Mapping):
        items = [(str(k), Path(v)) for k, v in sources.items()]
    else:
        paths = [Path(p) for p in sources]
        stems = list(names) if names else [p.stem for p in paths]
        if len(stems) != len(paths):
            raise DataValidationError("names and sources must have the same length")
        items = list(zip(stems, paths, strict=False))
    if not items:
        raise DataValidationError("No raster sources supplied.")

    arrays, ref_crs, ref_transform, ref_nodata = [], None, None, None
    for i, (name, path) in enumerate(items):
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")
        arr, crs, transform, nd = _read_band(path)
        if i == 0:
            ref_crs, ref_transform, ref_nodata, ref_shape = crs, transform, nd, arr.shape
        else:
            if arr.shape != ref_shape:
                raise SpatialAlignmentError(
                    f"Raster '{name}' has shape {arr.shape}, expected {ref_shape} "
                    "(rasters are never resampled implicitly)."
                )
            if (crs is None) != (ref_crs is None) or (crs is not None and crs != ref_crs):
                raise SpatialAlignmentError(
                    f"Raster '{name}' has CRS {crs}, expected {ref_crs}."
                )
            if not np.allclose(list(transform)[:6], list(ref_transform)[:6], atol=atol):
                raise SpatialAlignmentError(
                    f"Raster '{name}' has a different affine transform than '{items[0][0]}'."
                )
        arrays.append(arr)

    out_nodata = nodata if nodata is not None else (
        ref_nodata if ref_nodata is not None and np.isfinite(ref_nodata) else DEFAULT_NODATA
    )
    return RasterStack(
        np.stack(arrays), [n for n, _ in items], ref_crs, ref_transform, float(out_nodata),
        {n: str(p) for n, p in items},
    )


def read_raster_dir(directory: str | Path, names: Sequence[str], pattern: str = "{name}.tif") -> RasterStack:
    """Read ``directory/pattern.format(name=...)`` for every name in ``names``."""
    directory = Path(directory)
    mapping = {}
    for name in names:
        candidate = directory / pattern.format(name=name)
        if not candidate.exists():
            alt = [p for p in directory.glob("*.tif") if p.stem.lower() == name.lower()]
            if not alt:
                raise FileNotFoundError(f"No raster for feature '{name}' in {directory}")
            candidate = alt[0]
        mapping[name] = candidate
    return read_raster_stack(mapping)


def load_raster_source(source: str | Path, names: Sequence[str] | None = None) -> RasterStack:
    """Read a multi-band GeoTIFF or a directory holding one ``<feature>.tif`` per feature."""
    path = Path(source)
    if path.is_dir():
        if not names:
            raise DataValidationError("Feature names are required to read a directory of single-band rasters.")
        return read_raster_dir(path, names)
    return read_raster_stack(path, names=names)


def write_raster(
    path: str | Path,
    array: np.ndarray,
    reference: RasterStack,
    nodata: float | None = None,
    dtype: str = "float32",
    descriptions: Sequence[str] | None = None,
    tags: Mapping[str, str] | None = None,
    compress: str = "deflate",
) -> Path:
    """Write a GeoTIFF that shares CRS, transform, width and height with ``reference``.

    ``NaN`` cells in floating-point arrays are written as ``nodata``. Integer outputs
    should already contain ``nodata`` in invalid cells.
    """
    rasterio = require("rasterio", "writing rasters")
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1:] != reference.shape:
        raise SpatialAlignmentError(
            f"Array shape {arr.shape} does not match the reference grid {reference.shape}."
        )
    nd = float(reference.nodata if nodata is None else nodata)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isfinite(arr), arr, nd)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": reference.height,
        "width": reference.width,
        "count": int(arr.shape[0]),
        "dtype": dtype,
        "crs": reference.crs,
        "transform": reference.transform,
        "nodata": nd,
        "compress": compress,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, str(desc))
        if tags:
            dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
    return path


def raster_metadata(path: str | Path) -> dict[str, Any]:
    """Return CRS, transform, size, nodata and dtype of a raster file."""
    rasterio = require("rasterio", "reading rasters")
    with rasterio.open(path) as src:
        return {
            "crs": None if src.crs is None else src.crs.to_string(),
            "transform": list(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
            "bounds": list(src.bounds),
        }
