## Reproducibility

One command regenerates all example data, results and figures:

```bash
python reproduce.py            # about 2 minutes on one CPU core
python reproduce.py --quick    # smoke run (small grid, two models)
```

It (1) generates the synthetic dataset, (2) runs the experiment defined in `examples/example_config.json`, (3) builds the HTML report, (4) draws the diagrams and logo, (5) copies the headline figures to `figures/`, and (6) refreshes the result tables in `README.md` and `docs/examples.md` from the actual CSV outputs.

## What is recorded

| Item | Where |
|---|---|
| Random seed, all model parameters, split sizes, block size, full configuration | `results/example_run/reproducibility.json` |
| Python, OS/platform, versions of NumPy, pandas, SciPy, scikit-learn, matplotlib, SHAP, XGBoost, LightGBM, rasterio, GeoPandas, Shapely, pyproj, joblib, openpyxl | same file, `environment` |
| Dataset generation parameters, generating coefficients, synthetic-data disclaimer | `data/dataset_metadata.json` |
| Index of every table, figure and raster | `results/example_run/experiment.json` |

Every random component takes an explicit seed (splits, models, SHAP background sampling, permutation importance, dataset generation), and the test-suite checks that two runs with the same seed give identical metric tables.

## Caveats

* Bit-for-bit equality across machines is not guaranteed: BLAS/thread scheduling, library versions and CPU architectures can change floating-point results slightly. Tables in the README were produced with the versions listed above them.
* Multi-threaded models (XGBoost, LightGBM, Random Forest with `n_jobs`) are deterministic for a fixed seed in our tests, but this is library behaviour, not a GeoExplain guarantee.
* Regenerating overwrites `data/` and `results/example_run/`.

## Using your own data

Copy `examples/example_config.json`, change `data`, `raster`, `features`, `target`, `crs`, `out_dir`, and run `geoxplain run --config my_config.json`. Archive the resulting `reproducibility.json` with your results.
