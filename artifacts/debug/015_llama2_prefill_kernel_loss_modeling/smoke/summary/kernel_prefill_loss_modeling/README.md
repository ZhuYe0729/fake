# Llama2 Kernel-Aware Prefill Loss Modeling Summary

This report uses real runtime NVFP4 kernel forward paths. `dense_nvfp4` and `sparse_nvfp4` include runtime activation quantization.

## Best Kernel Local Error Proxies

| metric | Pearson | Spearman | n |
|---|---:|---:|---:|
| weight_mse | -1.0000 | -1.0000 | 2 |
| weight_rel_mse | -1.0000 | -1.0000 | 2 |
| weight_rmse_over_rms | -1.0000 | -1.0000 | 2 |
| weight_max_abs_error | -1.0000 | -1.0000 | 2 |
| output_mse | -1.0000 | -1.0000 | 2 |
| output_rel_mse | -1.0000 | -1.0000 | 2 |

## Highest Loss-Delta Layers

| method | layer | mean loss delta |
|---|---:|---:|
| dense_nvfp4 | 0 | -0.002463 |
| sparse_nvfp4 | 0 | -0.004957 |

## Highest Loss-Delta Linear Types

| method | type | mean loss delta |
|---|---|---:|
| dense_nvfp4 | q_proj | -0.002463 |
| sparse_nvfp4 | q_proj | -0.004957 |

## Outputs

- `kernel_local_error_loss_correlations.csv`
- `layer_loss_summary.csv`
- `linear_type_loss_summary.csv`
- `layer_depth_loss_delta.png`
- `linear_type_loss_delta.png`
- `best_kernel_proxy_vs_loss_delta.png`
- `weight_rel_mse_vs_kernel_output_rel_mse.png`
