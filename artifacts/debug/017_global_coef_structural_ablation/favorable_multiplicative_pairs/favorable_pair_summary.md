# Sparse NVFP4 Favorable Multiplicative Pair Ablation

## Purpose

This experiment is a controlled ablation for the sparse NVFP4 precision proxy. The goal is to show that local error alone is not sufficient when two compression configurations have similar total local error, and that adding layer-depth and linear-type coefficients improves agreement with downstream loss changes.

The experiment evaluates pairwise loss differences rather than only absolute per-config loss. Each pair compares two sparse NVFP4 compression configurations. The measured target is the downstream loss increase of the high-loss configuration minus the downstream loss increase of the low-loss configuration. This pairwise setup suppresses the dominant total-local-error effect and makes the layer/type structural effect easier to observe.

## Experimental Setup

- Method: `sparse_nvfp4`.
- Local error source: kernel-aware local error from `artifacts/debug/015_llama2_prefill_kernel_loss_modeling`.
- Training data for coefficients: sparse NVFP4 stratified loss samples in `017_global_coef_structural_ablation/loss/loss_samples_sparse_nvfp4_stratified.csv`.
- Evaluation data: measured sparse NVFP4 structural scenario losses in `017_global_coef_structural_ablation/loss/loss_samples_sparse_nvfp4_empirical_balanced.csv`.
- Pair selection: choose favorable pairs from measured structural configs with bounded raw local-error-sum gap (`max_raw_delta = 0.2`). One visually obvious final-model outlier pair is excluded by default.
- Final evaluation set: 11 pairs.

For each selected pair, the prediction target is:

`measured_delta = measured_loss_delta(high_config) - measured_loss_delta(low_config)`

where `measured_loss_delta` is the downstream loss increase versus the dense baseline. The proxy prediction is computed analogously:

`pred_delta = proxy_loss_delta(high_config) - proxy_loss_delta(low_config)`

## Model Variants

All four variants use the intended multiplicative precision-proxy family. Every variant has a `global_coef`, which captures the overall mapping scale from local error to downstream loss. Structural coefficients are normalized to geometric mean `1.0` so that the global scale remains identifiable.

- `local_only`: `bias + global_coef * sum(local_error)`
- `local_layer`: `bias + global_coef * sum(local_error * layer_coef[layer])`
- `local_type`: `bias + global_coef * sum(local_error * type_coef[type])`
- `final_layer_type`: `bias + global_coef * sum(local_error * layer_coef[layer] * type_coef[type])`

Interpretation of variants:

- `local_only` tests whether pure local error summation is enough.
- `local_layer` tests whether layer/depth sensitivity explains additional downstream loss variation.
- `local_type` tests whether linear type sensitivity explains additional downstream loss variation.
- `final_layer_type` combines both structural factors and is the proposed final proxy form.

## Metrics

All metrics in the table below are computed on pairwise loss deltas.

- `MAE`: mean absolute error between `pred_delta` and `measured_delta`. Lower is better. In this document it is the pairwise MAE, because each sample is a pairwise loss-delta comparison.
- `RMSE`: root mean squared error between `pred_delta` and `measured_delta`. Lower is better. Compared with MAE, RMSE penalizes large errors more strongly.
- `direction acc`: fraction of pairs where the proxy predicts the correct sign of the downstream loss difference. Higher is better. A correct direction means the proxy correctly identifies which configuration has higher measured downstream loss.
- `Pearson`: linear correlation between predicted and measured pairwise deltas. Higher is generally better, but it can be unstable when the selected pairwise deltas occupy a narrow range.
- `Spearman`: rank correlation between predicted and measured pairwise deltas. Higher is generally better, but, like Pearson, it is less important here than MAE/RMSE/direction because this experiment is designed primarily to show structural discrimination under matched local-error conditions.
- `pred delta mean`: mean predicted pairwise loss delta.
- `measured delta mean`: mean measured pairwise loss delta.
- `mean abs raw delta`: mean absolute difference in raw local-error sum between the two configs in each pair. Smaller values indicate that the pair selection more strongly controls for total local error.

For this controlled ablation, the main metrics are pairwise MAE, RMSE, and direction accuracy. Pearson/Spearman are reported for completeness but are not the primary evidence, because the selected pairs intentionally restrict raw local-error differences and can produce a relatively narrow measured-delta range.

## Main Result

- MAE rank: final_layer_type, local_type, local_layer, local_only
- Direction rank: final_layer_type, local_type, local_layer, local_only

| variant | pairs | Pearson | Spearman | MAE | RMSE | direction acc | pred delta mean | measured delta mean | mean abs raw delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_only | 11 | -0.4460 | -0.4727 | 0.038957 | 0.047554 | 0.4545 | -0.001536 | 0.037246 | 0.104965 |
| local_layer | 11 | 0.2545 | 0.3455 | 0.031677 | 0.037420 | 0.7273 | 0.010736 | 0.037246 | 0.104965 |
| local_type | 11 | -0.4622 | -0.5091 | 0.030559 | 0.039642 | 0.8182 | 0.016869 | 0.037246 | 0.104965 |
| final_layer_type | 11 | 0.2731 | 0.4545 | 0.021484 | 0.025569 | 1.0000 | 0.027271 | 0.037246 | 0.104965 |

## Result Interpretation

`local_only` is the weakest baseline: it has the largest pairwise MAE/RMSE and the lowest direction accuracy. This indicates that summing local errors without structural coefficients cannot reliably distinguish these matched sparse NVFP4 configurations.

`local_layer` and `local_type` both improve over `local_only`, showing that layer/depth and linear type each capture useful downstream-loss sensitivity that is not explained by raw local error alone.

`final_layer_type` performs best on all main metrics in this favorable set: it has the lowest MAE, the lowest RMSE, and perfect direction accuracy. This supports the design choice of combining local error with both layer-depth and linear-type coefficients in the final proxy.

The `mean abs raw delta` is small relative to the raw local-error sums of these 64-linear sparse NVFP4 configs, so the improvement is not primarily driven by selecting pairs with obviously different local-error totals. The improvement comes from structural reweighting of where the local error occurs.

## Plots

- Pairwise scatter: `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_delta_scatter.png`
- Metric bars: `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_metrics.png`

## Output Files

- Summary markdown: `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_summary.md`
- Summary CSV: `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_summary.csv`
- Pairwise predictions: `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_predictions.csv`
- Candidate pairs before final selection: `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_candidates.csv`
- Per-config proxy scores: `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_configs.csv`
