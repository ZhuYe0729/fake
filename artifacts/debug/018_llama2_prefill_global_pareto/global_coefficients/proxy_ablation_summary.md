# Precision Proxy Ablation Summary

## Holdout Metrics

| method | variant | rows | Pearson | Spearman | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| dense_nvfp4 | local_only | 24 | 0.8296 | 0.8061 | 0.035786 | 0.040576 |
| dense_nvfp4 | local_layer | 24 | 0.8234 | 0.8017 | 0.036368 | 0.041322 |
| dense_nvfp4 | local_type | 24 | 0.8325 | 0.8183 | 0.035895 | 0.040905 |
| dense_nvfp4 | final_layer_type | 24 | 0.8287 | 0.7930 | 0.036401 | 0.041492 |
| dense_nvfp4 | dense_calibrated_reference | 24 | 0.8542 | 0.8765 | 0.007535 | 0.009639 |
| sparse_bf16 | local_only | 24 | 0.9426 | 0.9017 | 0.042451 | 0.050313 |
| sparse_bf16 | local_layer | 24 | 0.9485 | 0.9165 | 0.046530 | 0.054684 |
| sparse_bf16 | local_type | 24 | 0.9379 | 0.9191 | 0.043038 | 0.051673 |
| sparse_bf16 | final_layer_type | 24 | 0.9425 | 0.9200 | 0.047178 | 0.056050 |
| sparse_nvfp4 | local_only | 24 | 0.9318 | 0.9278 | 0.057226 | 0.073479 |
| sparse_nvfp4 | local_layer | 24 | 0.9343 | 0.9400 | 0.061483 | 0.075697 |
| sparse_nvfp4 | local_type | 24 | 0.9391 | 0.9278 | 0.054233 | 0.070726 |
| sparse_nvfp4 | final_layer_type | 24 | 0.9437 | 0.9574 | 0.056766 | 0.071804 |

## Plots

- `artifacts/debug/018_llama2_prefill_global_pareto/global_coefficients/proxy_ablation_holdout_spearman.png`
- `artifacts/debug/018_llama2_prefill_global_pareto/global_coefficients/proxy_ablation_holdout_rmse.png`
- `artifacts/debug/018_llama2_prefill_global_pareto/global_coefficients/proxy_ablation_holdout_predictions_sparse_bf16.png`
- `artifacts/debug/018_llama2_prefill_global_pareto/global_coefficients/proxy_ablation_holdout_predictions_dense_nvfp4.png`
- `artifacts/debug/018_llama2_prefill_global_pareto/global_coefficients/proxy_ablation_holdout_predictions_sparse_nvfp4.png`
