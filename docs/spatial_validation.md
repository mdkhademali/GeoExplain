## Spatial validation

## Why it matters

Geographic data are usually **spatially autocorrelated** (Tobler's first law: near things are more similar than distant things). With ordinary random train/test splits, a test point often has a near-identical neighbour in the training set, so the model can score well by memorising local patterns. That measures interpolation between neighbours, not the ability to predict in **new areas**, which is what most mapping applications need. Random validation is therefore frequently optimistic, sometimes strongly so. Spatial validation withholds contiguous blocks of the study area so that test locations are far from training locations.

The size of the effect depends on the data, the model and the block size; it is not guaranteed, and this is why GeoExplain measures it instead of assuming it.

## Tools

| Function | Purpose |
|---|---|
| `random_split(n, test_size, random_state, stratify)` | ordinary (optionally stratified) hold-out |
| `spatial_split(coords, test_size, block_size, random_state, buffer)` | whole spatial blocks go to the test set; optional `buffer` removes training points within that distance of any test point |
| `SpatialBlockKFold(n_splits, block_size, shuffle, random_state, buffer)` | scikit-learn-style block cross-validation; `.fold_assignments(coords)` gives fold ids |
| `compare_validation_strategies(model, dataset, ...)` | runs random k-fold and spatial block CV, returning per-fold metrics, a summary and `.optimism()` |
| `default_block_size`, `make_spatial_blocks` | grid blocks over the extent |
| `morans_i(values, coords)` | Moran's I with a permutation p-value (kNN weights) |
| `empirical_semivariogram`, `estimate_range` | diagnose the distance over which values are correlated |

```python
from geoxplain import compare_validation_strategies
res = compare_validation_strategies(model, ds, n_splits=5)
print(res.summary)      # mean/std per strategy and metric
print(res.optimism())   # random minus spatial, per metric
```

## Choosing the block size

Blocks should be at least about as large as the spatial range of autocorrelation. Estimate it with the semivariogram (`estimate_range`) or Moran's I of the target/residuals, and pass `block_size=` (map units). The default splits the extent into a coarse grid sized for the requested number of folds. A `buffer` adds a further safety margin at the cost of training data.

## Caveats

* Block CV is still an approximation to the real prediction situation; if you will predict far outside the sampled area, also consider environmental (feature-space) dissimilarity.
* Small blocks or few blocks give noisy fold scores (see the fold standard deviations).
* A spatial fold can contain a single class; ROC/PR-AUC are then `NaN` for that fold.
* Hyper-parameter tuning must also use spatial folds, or the optimism returns.
