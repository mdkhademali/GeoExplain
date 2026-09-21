"""Tabular data loading, validation and export.

The central object is :class:`GeoDataset`: a feature table ``X``, a target ``y`` and
(optional) projected/geographic coordinates that are needed for spatial validation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.exceptions import DataValidationError
from ..utils.optional import require

TASKS = ("classification", "regression")


@dataclass
class GeoDataset:
    """A validated geospatial modelling table.

    Attributes
    ----------
    X:
        Feature matrix (all numeric, finite).
    y:
        Target values aligned with ``X``.
    coords:
        DataFrame with columns ``x`` and ``y`` (in the dataset CRS) or ``None``.
    ids:
        Identifier of each observation (defaults to the row number).
    target_name:
        Name of the target column.
    task:
        ``"classification"`` or ``"regression"``.
    crs:
        CRS of ``coords`` as an authority string (e.g. ``"EPSG:32645"``) or ``None``.
    name:
        Human-readable dataset name.
    notes:
        Free-form bookkeeping (e.g. number of dropped rows, source path).
    """

    X: pd.DataFrame
    y: pd.Series
    coords: pd.DataFrame | None
    ids: pd.Series
    target_name: str
    task: str
    crs: str | None = None
    name: str = "dataset"
    notes: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def feature_names(self) -> list[str]:
        """Names of the predictor columns."""
        return list(self.X.columns)

    def subset(self, index: Sequence[int] | np.ndarray) -> "GeoDataset":
        """Return a new dataset with the rows at integer positions ``index``."""
        idx = np.asarray(index, dtype=int)
        return GeoDataset(
            X=self.X.iloc[idx].reset_index(drop=True),
            y=self.y.iloc[idx].reset_index(drop=True),
            coords=None if self.coords is None else self.coords.iloc[idx].reset_index(drop=True),
            ids=self.ids.iloc[idx].reset_index(drop=True),
            target_name=self.target_name,
            task=self.task,
            crs=self.crs,
            name=self.name,
            notes=dict(self.notes),
        )

    def require_coords(self) -> np.ndarray:
        """Return coordinates as an ``(n, 2)`` array or raise if unavailable."""
        if self.coords is None:
            raise DataValidationError(
                "This operation needs coordinates. Provide x/y columns when creating the "
                "dataset (e.g. x='x', y='y')."
            )
        return self.coords[["x", "y"]].to_numpy(dtype=float)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly dataset description."""
        info: dict[str, Any] = {
            "name": self.name,
            "n_observations": int(len(self)),
            "n_features": int(self.X.shape[1]),
            "features": self.feature_names,
            "target": self.target_name,
            "task": self.task,
            "crs": self.crs,
            "has_coordinates": self.coords is not None,
            "feature_statistics": {
                col: {
                    "min": float(self.X[col].min()),
                    "mean": float(self.X[col].mean()),
                    "max": float(self.X[col].max()),
                    "std": float(self.X[col].std(ddof=0)),
                }
                for col in self.X.columns
            },
        }
        if self.task == "classification":
            counts = self.y.value_counts().sort_index()
            info["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        else:
            info["target_statistics"] = {
                "min": float(self.y.min()),
                "mean": float(self.y.mean()),
                "max": float(self.y.max()),
                "std": float(self.y.std(ddof=0)),
            }
        if self.coords is not None:
            info["extent"] = {
                "xmin": float(self.coords["x"].min()),
                "xmax": float(self.coords["x"].max()),
                "ymin": float(self.coords["y"].min()),
                "ymax": float(self.coords["y"].max()),
            }
        info.update({k: v for k, v in self.notes.items() if k not in info})
        return info


def infer_task(y: pd.Series, max_integer_classes: int = 20) -> str:
    """Guess ``"classification"`` or ``"regression"`` from a target column.

    Rules: non-numeric / boolean / categorical -> classification; integer-valued with at
    most ``max_integer_classes`` distinct values -> classification; otherwise regression.
    Pass ``task`` explicitly when this heuristic is not appropriate.
    """
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    values = y.dropna()
    is_integer_valued = bool(np.all(np.isclose(values, np.round(values))))
    if is_integer_valued and values.nunique() <= max_integer_classes:
        return "classification"
    return "regression"


def load_table(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Load CSV, Excel, Parquet or a vector file (GPKG/SHP/GeoJSON) into a DataFrame.

    For vector point layers, ``x`` and ``y`` columns are added from the geometry (unless
    they already exist), the geometry column is dropped and the layer CRS is stored in
    ``df.attrs['crs']``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, **read_kwargs)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        require("openpyxl", "reading Excel files")
        return pd.read_excel(path, **read_kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **read_kwargs)
    if suffix in {".gpkg", ".shp", ".geojson", ".json"}:
        gpd = require("geopandas", "reading vector data")
        gdf = gpd.read_file(path, **read_kwargs)
        df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
        if not (gdf.geom_type == "Point").all():
            raise DataValidationError("Only point geometries are supported for tabular loading.")
        if "x" not in df.columns:
            df["x"] = gdf.geometry.x.to_numpy()
        if "y" not in df.columns:
            df["y"] = gdf.geometry.y.to_numpy()
        df.attrs["crs"] = None if gdf.crs is None else gdf.crs.to_string()
        return df
    raise DataValidationError(f"Unsupported file type '{suffix}' for {path}")


def dataframe_to_dataset(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str] | None = None,
    x: str | None = "x",
    y: str | None = "y",
    id_col: str | None = "id",
    task: str | None = None,
    crs: str | None = None,
    missing: str = "drop",
    name: str = "dataset",
) -> GeoDataset:
    """Validate a DataFrame and convert it to a :class:`GeoDataset`.

    Parameters
    ----------
    df:
        Input table.
    target:
        Name of the target column.
    features:
        Predictor columns. If ``None``, all numeric columns except the target, id and
        coordinate columns are used.
    x, y:
        Coordinate column names. Pass ``None`` for both to build a dataset without
        coordinates (spatial validation will then be unavailable).
    id_col:
        Identifier column; if missing, the row number is used.
    task:
        ``"classification"``, ``"regression"`` or ``None`` (inferred, see :func:`infer_task`).
    crs:
        CRS string; defaults to ``df.attrs['crs']`` when present.
    missing:
        ``"drop"`` (drop incomplete rows), ``"median"`` (impute *features* with the median,
        still dropping rows with a missing target/coordinate) or ``"error"``.
    name:
        Dataset name.

    Raises
    ------
    DataValidationError
        On missing columns, non-numeric features, infinite values, unknown task, etc.
    """
    if missing not in {"drop", "median", "error"}:
        raise DataValidationError("missing must be one of 'drop', 'median', 'error'")
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")
    if df.empty:
        raise DataValidationError("Input table is empty.")
    if target not in df.columns:
        raise DataValidationError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    if task is not None and task not in TASKS:
        raise DataValidationError(f"task must be one of {TASKS}, got {task!r}")

    use_coords = x is not None and y is not None
    if (x is None) != (y is None):
        raise DataValidationError("Provide both x and y coordinate columns, or neither.")
    if use_coords:
        for col in (x, y):
            if col not in df.columns:
                raise DataValidationError(f"Coordinate column '{col}' not found.")

    excluded = {target}
    if use_coords:
        excluded |= {x, y}
    if id_col is not None:
        excluded.add(id_col)

    if features is None:
        features = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
        if not features:
            raise DataValidationError("No numeric feature columns could be determined.")
    features = list(features)
    if len(set(features)) != len(features):
        raise DataValidationError("Duplicate feature names supplied.")
    absent = [c for c in features if c not in df.columns]
    if absent:
        raise DataValidationError(f"Feature columns not found: {absent}")
    overlap = [c for c in features if c == target]
    if overlap:
        raise DataValidationError("The target cannot also be a feature (data leakage).")
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(
            f"Non-numeric feature columns: {non_numeric}. Encode categorical variables "
            "(e.g. one-hot) before creating a GeoDataset."
        )

    work = df.copy()
    work[features] = work[features].astype(float).replace([np.inf, -np.inf], np.nan)
    if use_coords:
        work[[x, y]] = work[[x, y]].astype(float).replace([np.inf, -np.inf], np.nan)

    critical = [target] + ([x, y] if use_coords else [])
    n_before = len(work)
    if missing == "error":
        bad = work[features + critical].isna().any(axis=1)
        if bad.any():
            raise DataValidationError(f"{int(bad.sum())} rows contain missing/infinite values.")
        n_imputed = 0
    elif missing == "drop":
        work = work.dropna(subset=features + critical)
        n_imputed = 0
    else:  # median
        work = work.dropna(subset=critical)
        n_imputed = int(work[features].isna().sum().sum())
        work[features] = work[features].fillna(work[features].median())
    n_dropped = n_before - len(work)
    if len(work) < 10:
        raise DataValidationError(f"Only {len(work)} valid rows remain; at least 10 are required.")
    work = work.reset_index(drop=True)

    if task is None:
        task = infer_task(work[target])
    if task == "classification":
        if work[target].nunique() < 2:
            raise DataValidationError("Classification target needs at least two classes.")
    else:
        if not pd.api.types.is_numeric_dtype(work[target]):
            raise DataValidationError("Regression target must be numeric.")

    if id_col is not None and id_col in work.columns:
        ids = work[id_col].reset_index(drop=True)
    else:
        ids = pd.Series(np.arange(len(work)), name="id")

    notes: dict[str, Any] = {"rows_input": int(n_before), "rows_dropped": int(n_dropped)}
    if n_imputed:
        notes["values_imputed_median"] = n_imputed
    constant = [c for c in features if work[c].nunique() <= 1]
    if constant:
        notes["constant_features"] = constant

    return GeoDataset(
        X=work[features].astype(float),
        y=work[target].reset_index(drop=True),
        coords=work[[x, y]].rename(columns={x: "x", y: "y"}).astype(float) if use_coords else None,
        ids=ids,
        target_name=target,
        task=task,
        crs=crs or df.attrs.get("crs"),
        name=name,
        notes=notes,
    )


def load_dataset(
    path: str | Path,
    target: str,
    features: Sequence[str] | None = None,
    x: str | None = "x",
    y: str | None = "y",
    id_col: str | None = "id",
    task: str | None = None,
    crs: str | None = None,
    missing: str = "drop",
) -> GeoDataset:
    """Convenience wrapper: :func:`load_table` followed by :func:`dataframe_to_dataset`."""
    df = load_table(path)
    ds = dataframe_to_dataset(
        df, target, features, x, y, id_col, task, crs, missing, name=Path(path).stem
    )
    ds.notes["source"] = str(path)
    return ds


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a DataFrame as CSV or Excel depending on the file suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        require("openpyxl", "writing Excel files")
        df.to_excel(path, index=index)
    else:
        df.to_csv(path, index=index)
    return path


def save_tables_excel(tables: Mapping[str, pd.DataFrame], path: str | Path, index: bool = True) -> Path:
    """Write several DataFrames as sheets of one Excel workbook (sheet names <= 31 chars)."""
    require("openpyxl", "writing Excel files")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, sheet_name=str(sheet)[:31], index=index)
    return path


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Return numeric column names excluding those listed in ``exclude``."""
    skip = set(exclude)
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
