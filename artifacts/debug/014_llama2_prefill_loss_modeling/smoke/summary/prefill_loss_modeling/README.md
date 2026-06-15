# Llama2 Prefill-Only Loss Modeling Summary

This report summarizes local linear errors and WikiText-2 mean CE loss deltas for prefill-only precision modeling.

## Best Local Error Proxies

| metric | Pearson | Spearman | n |
|---|---:|---:|---:|
| weight_mse | nan | nan | 0 |
| weight_rel_mse | nan | nan | 0 |
| weight_rmse_over_rms | nan | nan | 0 |
| weight_max_abs_error | nan | nan | 0 |
| output_mse | nan | nan | 0 |
| output_rel_mse | nan | nan | 0 |

## Highest Loss-Delta Layers

| method | layer | mean loss delta |
|---|---:|---:|
| dense_nvfp4 | 0 | -0.000100 |

## Highest Loss-Delta Linear Types

| method | type | mean loss delta |
|---|---|---:|
| dense_nvfp4 | q_proj | -0.000786 |

## Outputs

- `local_error_loss_correlations.csv`
- `layer_loss_summary.csv`
- `linear_type_loss_summary.csv`
- `layer_depth_loss_delta.png`
- `linear_type_loss_delta.png`
- `best_local_proxy_vs_loss_delta.png`
- `weight_rel_mse_vs_output_rel_mse.png`
