"""Random and spatial data splitting.

Why spatial splitting matters
-----------------------------
Geospatial samples are spatially autocorrelated: nearby observations resemble each other
(Tobler's first law). A random train/test split places test points *next to* training
points, so a flexible model can succeed by interpolating local neighbours instead of
learning relationships that transfer to new areas. Spatial block splitting keeps whole
spatial blocks (optionally separated by a buffer) out of training and therefore gives a
more honest estimate of performance in unsampled locations.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split

from ..utils.exceptions import DataValidationError


def _as_coords(coords: object) -> np.ndarray:
    arr = np.asarray(getattr(coords, "to_numpy", lambda: coords)(), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise DataValidationError("coords must have shape (n, 2)")
    if not np.isfinite(arr).all():
        raise DataValidationError("coords contain NaN/inf")
    return arr


def default_block_size(coords: object, n_splits: int = 5) -> float:
    """Heuristic block edge length giving roughly ``4 * n_splits`` blocks over the extent."""
    xy = _as_coords(coords)
    extent = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    if extent <= 0:
        raise DataValidationError("All coordinates are identical; cannot build spatial blocks.")
    per_side = int(np.ceil(np.sqrt(4 * n_splits)))
    return extent / per_side


def make_spatial_blocks(coords: object, block_size: float) -> np.ndarray:
    """Assign each point to a square block of edge ``block_size`` (CRS units).

    Returns integer block ids ``0..B-1`` (dense labels). Use a projected CRS so that the
    block size has a consistent meaning in metres.
    """
    xy = _as_coords(coords)
    if block_size <= 0:
        raise DataValidationError("block_size must be positive")
    ix = np.floor((xy[:, 0] - xy[:, 0].min()) / block_size).astype(int)
    iy = np.floor((xy[:, 1] - xy[:, 1].min()) / block_size).astype(int)
    _, labels = np.unique(np.stack([ix, iy], axis=1), axis=0, return_inverse=True)
    return labels.reshape(-1)


def _apply_buffer(train_idx: np.ndarray, test_idx: np.ndarray, xy: np.ndarray, buffer: float) -> np.ndarray:
    """Drop training points closer than ``buffer`` to any test point."""
    if buffer <= 0 or len(test_idx) == 0:
        return train_idx
    dist, _ = cKDTree(xy[test_idx]).query(xy[train_idx])
    return train_idx[dist > buffer]


def random_split(
    n: int, test_size: float = 0.25, random_state: int = 42, stratify: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Random hold-out split returning ``(train_idx, test_idx)`` integer positions."""
    idx = np.arange(n)
    try:
        train, test = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError:  # e.g. a class with a single member
        train, test = train_test_split(idx, test_size=test_size, random_state=random_state)
    return np.sort(train), np.sort(test)


def spatial_split(
    coords: object,
    test_size: float = 0.25,
    block_size: float | None = None,
    random_state: int = 42,
    buffer: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial hold-out: entire blocks are assigned to the test set.

    Blocks are drawn at random until at least ``test_size`` of the observations are in
    the test set. Optionally, training points within ``buffer`` (CRS units) of any test
    point are removed to reduce leakage across block edges.
    """
    if not 0 < test_size < 1:
        raise DataValidationError("test_size must be in (0, 1)")
    xy = _as_coords(coords)
    block_size = block_size or default_block_size(xy)
    blocks = make_spatial_blocks(xy, block_size)
    n_blocks = blocks.max() + 1
    if n_blocks < 2:
        raise DataValidationError("Fewer than two spatial blocks; reduce block_size.")
    rng = np.random.default_rng(random_state)
    counts = np.bincount(blocks)
    target = test_size * len(xy)
    chosen, total = [], 0
    for b in rng.permutation(n_blocks):
        if total >= target:
            break
        if len(chosen) == n_blocks - 1:
            break
        chosen.append(b)
        total += counts[b]
    is_test = np.isin(blocks, chosen)
    test_idx = np.flatnonzero(is_test)
    train_idx = _apply_buffer(np.flatnonzero(~is_test), test_idx, xy, buffer)
    return train_idx, test_idx


class SpatialBlockKFold:
    """K-fold cross-validation over spatial blocks (scikit-learn style splitter).

    Parameters
    ----------
    n_splits:
        Number of folds (>= 2).
    block_size:
        Block edge length in CRS units. ``None`` uses :func:`default_block_size`.
    shuffle, random_state:
        Blocks are shuffled before being balanced across folds.
    buffer:
        Optional exclusion distance around each test fold (CRS units).

    Examples
    --------
    >>> cv = SpatialBlockKFold(n_splits=5, block_size=600.0)   # doctest: +SKIP
    >>> for train, test in cv.split(X, y, coords=coords):       # doctest: +SKIP
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size: float | None = None,
        shuffle: bool = True,
        random_state: int | None = 42,
        buffer: float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise DataValidationError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.block_size = block_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.buffer = buffer

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D401 - sklearn API
        """Return the number of folds."""
        return self.n_splits

    def fold_assignments(self, coords: object) -> np.ndarray:
        """Return the fold id (``0..n_splits-1``) of every observation."""
        xy = _as_coords(coords)
        size = self.block_size or default_block_size(xy, self.n_splits)
        blocks = make_spatial_blocks(xy, size)
        n_blocks = int(blocks.max()) + 1
        if n_blocks < self.n_splits:
            raise DataValidationError(
                f"Only {n_blocks} spatial blocks for {self.n_splits} folds; reduce block_size."
            )
        counts = np.bincount(blocks)
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(n_blocks) if self.shuffle else np.arange(n_blocks)
        order = order[np.argsort(-counts[order], kind="stable")]  # big blocks first, greedy balance
        fold_of_block = np.empty(n_blocks, dtype=int)
        load = np.zeros(self.n_splits)
        for b in order:
            f = int(np.argmin(load))
            fold_of_block[b] = f
            load[f] += counts[b]
        return fold_of_block[blocks]

    def split(self, X=None, y=None, groups=None, coords: object | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs. ``coords`` is required."""
        if coords is None:
            raise DataValidationError("SpatialBlockKFold.split requires coords=(n, 2) array or DataFrame.")
        xy = _as_coords(coords)
        folds = self.fold_assignments(xy)
        for f in range(self.n_splits):
            test_idx = np.flatnonzero(folds == f)
            train_idx = _apply_buffer(np.flatnonzero(folds != f), test_idx, xy, self.buffer)
            if len(train_idx) == 0 or len(test_idx) == 0:
                warnings.warn(f"Fold {f} is empty after buffering; skipping.", stacklevel=2)
                continue
            yield train_idx, test_idx
