"""Reproducible *synthetic* geospatial data for examples, tests and documentation.

Everything produced here is artificial. The rasters are smooth random fields, and the
targets are drawn from an explicitly documented generating process. Results computed on
these data illustrate how GeoExplain behaves; they are **not** real-world observations
and must not be interpreted as findings about any actual place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

from ..io.raster import RasterStack, write_raster
from ..utils.serialization import write_json

FEATURES = ["ndvi", "ndbi", "lst", "elevation", "rainfall", "distance_to_water", "population"]

FEATURE_INFO = {
    "ndvi": "Normalised difference vegetation index (unitless, synthetic)",
    "ndbi": "Normalised difference built-up index (unitless, synthetic)",
    "lst": "Land surface temperature (degrees Celsius, synthetic)",
    "elevation": "Elevation above an arbitrary datum (metres, synthetic)",
    "rainfall": "Annual rainfall (millimetres, synthetic)",
    "distance_to_water": "Distance to nearest synthetic water cell (metres)",
    "population": "Population density (persons per km2, synthetic)",
}

FEATURE_LABELS = {
    "ndvi": "NDVI",
    "ndbi": "NDBI",
    "lst": "LST",
    "elevation": "Elevation",
    "rainfall": "Rainfall",
    "distance_to_water": "DistanceToWater",
    "population": "Population",
}

# Coefficients of the *data-generating* logit (documented for transparency).
GENERATING_COEFFICIENTS = {
    "intercept": "solved numerically so that mean(p) equals target_prevalence",
    "elevation (standardised)": -1.3,
    "proximity_to_water = exp(-distance/300 m) (centred)": 2.5,
    "rainfall (standardised)": 0.8,
    "ndvi (standardised)": -0.6,
    "log1p(population) (standardised)": 0.3,
    "rainfall x (-elevation) interaction (standardised)": 0.3,
    "ndbi (standardised)": 0.0,
    "lst (standardised)": 0.0,
    "unobserved smooth spatial field (not available to models)": 1.4,
}


@dataclass
class SyntheticGeoData:
    """Container for the synthetic raster world."""

    rasters: dict[str, np.ndarray]
    probability: np.ndarray
    target: np.ndarray
    susceptibility_index: np.ndarray
    landcover: np.ndarray
    water_mask: np.ndarray
    crs: str
    transform: Any
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def stack(self) -> RasterStack:
        """Feature rasters as a :class:`RasterStack` (band order = :data:`FEATURES`)."""
        return RasterStack.from_arrays({k: self.rasters[k] for k in FEATURES}, self.crs, self.transform)


def _standardise(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-12)


def _field(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Standardised Gaussian random field obtained by smoothing white noise."""
    return _standardise(gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect"))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic_geodata(
    size: int = 100,
    seed: int = 42,
    pixel_size: float = 30.0,
    origin: tuple[float, float] = (500000.0, 2600000.0),
    crs: str = "EPSG:32645",
    target_prevalence: float = 0.30,
) -> SyntheticGeoData:
    """Generate a synthetic multi-band raster world with binary, continuous and class targets.

    Parameters
    ----------
    size:
        Number of rows and columns.
    seed:
        Seed of :func:`numpy.random.default_rng`; the output is fully determined by it.
    pixel_size:
        Pixel size in CRS units (metres for the default UTM CRS).
    origin:
        ``(west, north)`` coordinates of the upper-left corner. The default is arbitrary.
    crs:
        CRS assigned to the rasters (projected CRS recommended).
    target_prevalence:
        Mean of the true event probability (used to solve the logit intercept).
    """
    from affine import Affine

    if size < 16:
        raise ValueError("size must be at least 16")
    rng = np.random.default_rng(seed)
    shape = (size, size)
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    xx, yy = cols / (size - 1), rows / (size - 1)

    # --- terrain and water ---------------------------------------------------
    z_elev = _standardise(0.9 * _field(rng, shape, size * 0.15) + 0.8 * (xx - 0.5) * 2.0)
    river_row = size * (0.45 + 0.08 * np.sin(2 * np.pi * 1.3 * xx)) + 2.0 * _field(rng, shape, size * 0.08)
    river = np.abs(rows - river_row) < 1.6
    dist_river_px = distance_transform_edt(~river)
    elevation = 12.0 + 6.0 * z_elev - 3.0 * np.exp(-dist_river_px / 6.0)
    pond = elevation < np.percentile(elevation, 2.0)
    water = river | pond
    elevation = np.clip(elevation, 0.5, None)
    distance_to_water = distance_transform_edt(~water) * pixel_size

    # --- climate and urban form ---------------------------------------------
    rainfall = 1900.0 + 220.0 * _standardise(0.9 * _field(rng, shape, size * 0.2) + 0.3 * (1 - xx) * 2.0)
    cx, cy = rng.uniform(0.25, 0.75, size=2)
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.16**2)))
    urban_raw = 0.5 * _sigmoid(1.8 * _field(rng, shape, size * 0.10)) + 0.6 * blob
    urban = (urban_raw - urban_raw.min()) / (urban_raw.max() - urban_raw.min())

    ndvi = 0.72 - 0.42 * urban + 0.06 * _field(rng, shape, size * 0.04)
    ndbi = -0.22 + 0.50 * urban + 0.05 * _field(rng, shape, size * 0.04)
    lst = 27.0 + 4.5 * urban + 3.0 * (0.6 - ndvi) + 0.7 * _field(rng, shape, size * 0.08)
    ndvi = np.where(water, -0.2 + 0.03 * rng.standard_normal(shape), ndvi)
    ndbi = np.where(water, -0.4 + 0.02 * rng.standard_normal(shape), ndbi)
    lst = np.where(water, lst - 3.5, lst)
    population = 150.0 * np.exp(3.0 * urban + 0.4 * _field(rng, shape, size * 0.05))
    population = np.where(water, 0.0, population)
    ndvi, ndbi = np.clip(ndvi, -0.3, 0.95), np.clip(ndbi, -0.6, 0.6)

    # --- event probability (documented generating process) -------------------
    prox = np.exp(-distance_to_water / 300.0)
    z_rain, z_elev2 = _standardise(rainfall), _standardise(elevation)
    latent = _field(rng, shape, size * 0.10)
    core = (
        -1.3 * z_elev2
        + 2.5 * (prox - prox.mean())
        + 0.8 * z_rain
        - 0.6 * _standardise(ndvi)
        + 0.3 * _standardise(np.log1p(population))
        + 0.3 * z_rain * (-z_elev2)
        + 1.4 * latent
    )
    lo, hi = -15.0, 15.0
    for _ in range(60):  # bisection for the intercept
        mid = 0.5 * (lo + hi)
        if _sigmoid(core + mid).mean() < target_prevalence:
            lo = mid
        else:
            hi = mid
    intercept = 0.5 * (lo + hi)
    probability = _sigmoid(core + intercept)
    target = (rng.uniform(size=shape) < probability).astype(np.int8)
    susceptibility = np.clip(100.0 * probability + 3.0 * rng.standard_normal(shape), 0.0, 100.0)

    # --- categorical land cover derived from features with 8% label noise -----
    lc = np.full(shape, "cropland", dtype=object)
    lc[(ndvi > 0.55) & (urban < 0.35)] = "vegetation"
    lc[urban > 0.55] = "built_up"
    lc[water] = "water"
    classes = np.array(["water", "built_up", "vegetation", "cropland"], dtype=object)
    flip = rng.uniform(size=shape) < 0.08
    lc[flip] = classes[rng.integers(0, len(classes), size=int(flip.sum()))]

    transform = Affine(pixel_size, 0.0, origin[0], 0.0, -pixel_size, origin[1])
    rasters = {
        "ndvi": ndvi, "ndbi": ndbi, "lst": lst, "elevation": elevation,
        "rainfall": rainfall, "distance_to_water": distance_to_water, "population": population,
    }
    rasters = {k: v.astype(np.float32) for k, v in rasters.items()}
    params = {
        "size": size, "seed": seed, "pixel_size": pixel_size, "origin": list(origin),
        "crs": crs, "target_prevalence": target_prevalence,
        "realised_prevalence": float(target.mean()),
        "generator": "numpy.random.default_rng + scipy.ndimage.gaussian_filter",
    }
    return SyntheticGeoData(
        rasters=rasters, probability=probability.astype(np.float32), target=target,
        susceptibility_index=susceptibility.astype(np.float32), landcover=lc,
        water_mask=water, crs=crs, transform=transform, seed=seed, params=params,
    )


def sample_points(data: SyntheticGeoData, n_samples: int = 2500, seed: int | None = None) -> pd.DataFrame:
    """Draw ``n_samples`` distinct cells uniformly at random and return a training table.

    Columns: ``id, x, y, ndvi, ndbi, lst, elevation, rainfall, distance_to_water,
    population, target`` plus ``susceptibility_index`` (continuous, for regression demos)
    and ``landcover`` (string classes, for multi-class demos).
    """
    seed = data.seed if seed is None else seed
    rng = np.random.default_rng(seed + 1)
    height, width = data.target.shape
    n_cells = height * width
    if n_samples > n_cells:
        raise ValueError(f"n_samples={n_samples} exceeds the number of cells ({n_cells})")
    flat = rng.choice(n_cells, size=n_samples, replace=False)
    r, c = np.divmod(flat, width)
    t = data.transform
    frame = pd.DataFrame(
        {
            "id": np.arange(n_samples),
            "x": t.a * (c + 0.5) + t.c,
            "y": t.e * (r + 0.5) + t.f,
        }
    )
    for name in FEATURES:
        frame[name] = data.rasters[name][r, c].astype(float)
    frame["target"] = data.target[r, c].astype(int)
    frame["susceptibility_index"] = data.susceptibility_index[r, c].astype(float)
    frame["landcover"] = data.landcover[r, c]
    return frame


def write_synthetic_dataset(
    out_dir: str | Path,
    size: int = 100,
    n_samples: int = 2500,
    seed: int = 42,
    pixel_size: float = 30.0,
) -> dict[str, Any]:
    """Generate and write the example dataset.

    Output layout::

        out_dir/
          samples.csv               training table (points sampled from the rasters)
          stack.tif                 multi-band predictor stack (band descriptions = feature names)
          rasters/<feature>.tif     one GeoTIFF per predictor
          truth/*.tif               generating probability and targets (for inspection only)
          dataset_metadata.json     synthetic-data disclaimer, parameters, coefficients

    Returns a dictionary with the paths that were written.
    """
    out_dir = Path(out_dir)
    data = make_synthetic_geodata(size=size, seed=seed, pixel_size=pixel_size)
    samples = sample_points(data, n_samples=n_samples, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.csv"
    samples.to_csv(samples_path, index=False)

    stack = data.stack
    raster_paths = {}
    for name in FEATURES:
        raster_paths[name] = write_raster(
            out_dir / "rasters" / f"{name}.tif", data.rasters[name], stack,
            descriptions=[name], tags={"SYNTHETIC": "true", "DESCRIPTION": FEATURE_INFO[name]},
        )
    truth = {
        "true_probability": data.probability,
        "target": data.target.astype(np.float32),
        "susceptibility_index": data.susceptibility_index,
    }
    truth_paths = {
        k: write_raster(out_dir / "truth" / f"{k}.tif", v, stack, tags={"SYNTHETIC": "true"})
        for k, v in truth.items()
    }
    meta = {
        "SYNTHETIC_DATA_NOTICE": (
            "All data in this directory are artificial and were generated by GeoExplain for "
            "demonstration and testing. They do not describe any real place or observation."
        ),
        "parameters": data.params,
        "crs": data.crs,
        "columns": {
            "id": "observation id", "x": "easting (CRS units)", "y": "northing (CRS units)",
            **FEATURE_INFO,
            "target": "binary event indicator drawn as Bernoulli(p_true) (classification demo)",
            "susceptibility_index": "continuous 0-100 index = 100 * p_true + noise (regression demo)",
            "landcover": "string class derived from features + 8% label noise (multi-class demo)",
        },
        "generating_coefficients": GENERATING_COEFFICIENTS,
        "n_samples": n_samples,
    }
    stack_path = write_raster(out_dir / "stack.tif", stack.data, stack, descriptions=list(stack.names),
                              tags={"SYNTHETIC": "true"})
    meta_path = write_json(out_dir / "dataset_metadata.json", meta)
    return {
        "samples": samples_path,
        "rasters": raster_paths,
        "stack": stack_path,
        "truth": truth_paths,
        "metadata": meta_path,
        "data": data,
    }
