## Explainability

## Global explanations

| Tool | Function | Output |
|---|---|---|
| SHAP summary (beeswarm) and bar | `viz.plot_shap_summary`, `viz.plot_shap_bar` | distribution / mean of |SHAP| per feature |
| Permutation importance | `permutation_importance(model, X, y)` | drop in ROC-AUC (classification) or R² (regression) when a feature is shuffled; any metric key via `scoring=` |
| Built-in importance | `builtin_importance(model)` | impurity importance (trees) or |coefficient| (linear); not available for MLP/SVM |

## Local explanations

`ShapValues.local(i)` returns each feature's value and SHAP contribution for observation `i`, sorted by magnitude. `viz.plot_waterfall(sv, i)` and `viz.plot_force(sv, i)` draw the waterfall and force-style plots.

## Feature effects

`partial_dependence(model, X, feature)` returns the average prediction and ICE curves over a grid between the 5th and 95th percentiles; `viz.plot_partial_dependence` draws them. `viz.plot_shap_dependence(sv, feature, color_by=other)` plots feature value against SHAP value.

## The additivity check

For every explanation, `base_value + Σ SHAP = model output` (in the explainer's output space). `GeoShapExplainer.check_additivity(sv)` verifies this numerically, and both `run_experiment` and spatial SHAP record the result. It guards against mismatched data or a wrong output space.

## What SHAP is (and is not)

SHAP assigns each feature a Shapley value: its average marginal contribution to the model's output over all orders in which features could be added, relative to a background distribution. The values are additive and depend on **the model, the background data and feature correlations**. They describe how the *model* behaves, i.e. association and prediction; they do **not** establish causal effects. See [methodology.md](methodology.md).
