## Scientific methodology

## 1. SHAP: Shapley-value feature attribution

Let *f* be a fitted model, *x* an observation with *M* features, and *v(S)* the expected model output when only the features in a subset *S* are "known" (the others are integrated out over a background distribution). The Shapley value of feature *i* is

φᵢ(x) = Σ_{S ⊆ F∖{i}} [ |S|! (M − |S| − 1)! / M! ] · ( v(S ∪ {i}) − v(S) ),

the average marginal contribution of *i* over all orders in which features could be revealed. SHAP values satisfy **local accuracy** (additivity): f(x) = φ₀ + Σᵢ φᵢ(x), where φ₀ = E[f] over the background, together with consistency and missingness. GeoExplain verifies additivity numerically for every explanation.

Implementation notes:
* Tree models: exact TreeSHAP (`tree_path_dependent` conditional expectations by default).
* Linear models: exact LinearSHAP with an independent background.
* MLP/SVM: model-agnostic permutation SHAP (exact additivity, estimated values).
* Output space: probability (Random Forest, MLP, SVM), log-odds (XGBoost, LightGBM, logistic regression), or target units (regression). Always recorded in `ShapValues.output_space`.

## 2. Spatial SHAP

For a raster with grid cells *c = (row, col)* and co-registered predictor bands, the feature vector at cell *c* is x_c. GeoExplain computes φᵢ(x_c) for every valid cell and stores it at cell *c* of an output raster with the identical grid definition. Thus

prediction(c) = φ₀ + Σᵢ φᵢ(c)   (in the explainer's output space),

and each map φᵢ(·) shows where the model uses feature *i* to raise or lower the prediction. The base value φ₀ is constant across the map. For point data the same values are attached to observations (`ShapValues.to_frame`, `spatial-explain --points`). Because SHAP is computed independently per cell, spatial patterns arise from (a) the spatial pattern of the inputs and (b) how the model combines them; the method makes no spatial-process assumption and never interpolates.

Summaries: mean φᵢ, mean |φᵢ|, share of cells with φᵢ > 0, and share of cells where feature *i* has the largest |φ| ("dominant driver"). The Moran's I of φᵢ (point data) quantifies how spatially clustered a feature's contribution is.

## 3. Spatial dependence and validation

Spatial autocorrelation means values at nearby locations are correlated. In random *k*-fold or hold-out validation, test observations have close training neighbours, so predictive skill partly reflects local memorisation and the estimate can be optimistic for prediction in new locations. Spatial block validation assigns whole blocks (optionally with a buffer) to test folds, so test points are separated from training points. GeoExplain reports both strategies side by side and quantifies the difference (`optimism`), as measured, never assumed. Moran's I (kNN weights, permutation test) and the empirical semivariogram help select block sizes.

## 4. Association, prediction, contribution and causation

| Concept | Meaning here |
|---|---|
| **Association** | statistical dependence between inputs and outcome in the training data |
| **Prediction** | the model's output for given inputs; assessed by validation |
| **Feature contribution** | how much of one prediction, relative to the base value, is attributed to a feature by SHAP (a property of the *model*) |
| **Causal effect** | what would happen to the outcome if a variable were intervened on; **not** identified by SHAP or by predictive validation |

A high SHAP value for "distance to water" says the *model* uses that variable strongly, not that moving a river would change flooding. Causal claims need a causal design (experiment, quasi-experiment, or explicit causal assumptions and identification strategy). GeoExplain never presents SHAP as evidence of causality.

## 5. Uncertainty

Entropy of predicted probabilities (classification) and the standard deviation across Random Forest trees (regression) measure model disagreement/ambiguity. They are not calibrated intervals and do not capture data error or extrapolation.

## 6. Synthetic example data

The example scenes are generated from smooth random fields and a documented generating process (`data/dataset_metadata.json`, `geoxplain.datasets.GENERATING_COEFFICIENTS`) with known dependence on elevation, proximity to water, rainfall, etc., and an unobserved spatially structured term. Because the truth is known, one can check that explanations behave sensibly. The data are **artificial** and say nothing about real environments.

## References

* Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
* Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*, 56–67.
* Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games II*.
* Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*, 913–929.
* Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications, 11*, 4540.
* Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. *Economic Geography, 46*, 234–240.
* Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika, 37*, 17–23.
