"""Spatial autocorrelation diagnostics: empirical semivariogram and Moran's I."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from ..utils.exceptions import DataValidationError


def empirical_semivariogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_bins: int = 15,
    max_dist: float | None = None,
    max_points: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Classical (Matheron) empirical semivariogram.

    ``gamma(h) = 1 / (2 N(h)) * sum (z_i - z_j)^2`` over pairs whose separation falls in
    each distance bin. Points are sub-sampled to ``max_points`` for tractability.

    Returns a DataFrame with ``distance`` (bin centre), ``semivariance`` and ``n_pairs``.
    """
    xy = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    if len(xy) != len(z):
        raise DataValidationError("coords and values must have the same length")
    if len(xy) > max_points:
        idx = np.random.default_rng(random_state).choice(len(xy), max_points, replace=False)
        xy, z = xy[idx], z[idx]
    dist = pdist(xy)
    diff2 = pdist(z[:, None], metric="sqeuclidean")
    max_dist = max_dist or float(np.percentile(dist, 50))
    edges = np.linspace(0, max_dist, n_bins + 1)
    which = np.digitize(dist, edges) - 1
    rows = []
    for b in range(n_bins):
        sel = which == b
        if sel.sum() == 0:
            continue
        rows.append(
            {
                "distance": 0.5 * (edges[b] + edges[b + 1]),
                "semivariance": 0.5 * float(diff2[sel].mean()),
                "n_pairs": int(sel.sum()),
            }
        )
    return pd.DataFrame(rows)


def estimate_range(semivariogram: pd.DataFrame, fraction: float = 0.95) -> float:
    """Heuristic effective range: first bin reaching ``fraction`` of the sill.

    The sill is approximated by the mean semivariance of the last three bins. Returns
    ``NaN`` if the semivariogram never reaches that level (no clear range within the
    distance window). This is a rough guide for choosing a block size, not a fitted model.
    """
    sv = semivariogram["semivariance"].to_numpy()
    if len(sv) < 4:
        return float("nan")
    sill = float(sv[-3:].mean())
    reached = np.flatnonzero(sv >= fraction * sill)
    return float(semivariogram["distance"].iloc[reached[0]]) if len(reached) else float("nan")


def morans_i(
    values: np.ndarray, coords: np.ndarray, k: int = 8, permutations: int = 199, random_state: int = 42
) -> dict[str, float]:
    """Moran's I with row-standardised k-nearest-neighbour weights.

    Returns ``I``, its expectation under no autocorrelation ``-1/(n-1)`` and a two-sided
    pseudo p-value from ``permutations`` random reshufflings of ``values``.
    Positive ``I`` indicates that similar values cluster in space.
    """
    z = np.asarray(values, dtype=float)
    xy = np.asarray(coords, dtype=float)
    n = len(z)
    if n < k + 2:
        raise DataValidationError("Too few observations for the requested number of neighbours.")
    z = z - z.mean()
    denom = float((z**2).sum())
    if denom == 0:
        return {"I": float("nan"), "expected": -1 / (n - 1), "p_value": float("nan"), "k": k, "n": n}
    _, nbr = cKDTree(xy).query(xy, k=k + 1)
    nbr = nbr[:, 1:]

    def _stat(v: np.ndarray) -> float:
        return float((v * v[nbr].mean(axis=1)).sum() / (v**2).sum())

    observed = _stat(z)
    rng = np.random.default_rng(random_state)
    perm = np.array([_stat(rng.permutation(z)) for _ in range(permutations)])
    expected = -1.0 / (n - 1)
    extreme = np.sum(np.abs(perm - expected) >= abs(observed - expected))
    return {
        "I": observed,
        "expected": expected,
        "p_value": float((1 + extreme) / (permutations + 1)),
        "k": k,
        "n": n,
    }
