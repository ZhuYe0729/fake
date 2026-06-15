# Precision Proxy Ablation Plan

## Summary
- Add an offline ablation experiment under `artifacts/debug/016_llama2_sparse_bf16_precision_proxy`.
- Compare proxy variants for `sparse_bf16`, `dense_nvfp4`, and `sparse_nvfp4` using existing sampled loss rows.
- No new GPU loss evaluation is needed.
- Use the same deterministic train/holdout split as current fitting.

## Key Changes
- Add `scripts/fit_proxy_ablation.py`.
- Fit these variants per method:
  - `local_only`: `bias + global_coef * sum(local_error)`
  - `local_layer`: `bias + sum(local_error * layer_coef[layer])`
  - `local_type`: `bias + sum(local_error * type_coef[type])`
  - `final_layer_type`: `bias + sum(local_error * layer_coef[layer] * type_coef[type])`
- Use sparse BF16 local errors from the existing sparse BF16 source and kernel-aware NVFP4 local errors from `015`.
- Report dense NVFP4 nonlinear calibration separately as `dense_calibrated_reference`.

## Outputs
- Save outputs under `artifacts/debug/016_llama2_sparse_bf16_precision_proxy/ablation/`:
  - `proxy_ablation_metrics.csv`
  - `proxy_ablation_predictions.csv`
  - `proxy_ablation_coefficients.json`
  - `proxy_ablation_summary.md`
  - `proxy_ablation_holdout_spearman.png`
  - `proxy_ablation_holdout_rmse.png`

## Test Plan
- Syntax check the new script.
- Run the ablation script on existing 120-row loss CSVs.
- Verify each method has four structural variants and 120 predictions per variant.
- Verify `holdout` uses the same 36-policy split.

## Assumptions
- The main final variant is the core `layer * type` multiplicative model.
- Dense NVFP4 nonlinear calibration is only a reference row, not part of the four structural ablations.
