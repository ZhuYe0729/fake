# Llama2 Kernel-Aware Prefill Loss Modeling Summary

This report uses real runtime NVFP4 kernel forward paths. `dense_nvfp4` and `sparse_nvfp4` include runtime activation quantization.

## Best Kernel Local Error Proxies

| metric | Pearson | Spearman | n |
|---|---:|---:|---:|
| output_rel_mse | 0.3146 | 0.5807 | 448 |
| output_rmse_over_rms | 0.3117 | 0.5807 | 448 |
| weight_max_abs_error | 0.1797 | 0.4513 | 448 |
| weight_mse | 0.2521 | 0.4442 | 448 |
| weight_rel_mse | 0.3293 | 0.4429 | 448 |
| weight_rmse_over_rms | 0.3410 | 0.4429 | 448 |

## Highest Loss-Delta Layers

| method | layer | mean loss delta |
|---|---:|---:|
| sparse_nvfp4 | 30 | 0.050552 |
| sparse_nvfp4 | 31 | 0.050269 |
| sparse_nvfp4 | 29 | 0.038292 |
| sparse_nvfp4 | 2 | 0.031766 |
| sparse_nvfp4 | 1 | 0.030253 |
| dense_nvfp4 | 1 | 0.023288 |
| sparse_nvfp4 | 25 | 0.019786 |
| sparse_nvfp4 | 17 | 0.019766 |

## Highest Loss-Delta Linear Types

| method | type | mean loss delta |
|---|---|---:|
| sparse_nvfp4 | down_proj | 0.213208 |
| sparse_nvfp4 | up_proj | 0.158648 |
| sparse_nvfp4 | gate_proj | 0.141538 |
| sparse_nvfp4 | v_proj | 0.080044 |
| sparse_nvfp4 | o_proj | 0.040050 |
| dense_nvfp4 | down_proj | 0.037456 |
| sparse_nvfp4 | q_proj | 0.028318 |
| sparse_nvfp4 | k_proj | 0.020061 |

## Outputs

- `kernel_local_error_loss_correlations.csv`
- `layer_loss_summary.csv`
- `linear_type_loss_summary.csv`
- `layer_depth_loss_delta.png`
- `linear_type_loss_delta.png`
- `best_kernel_proxy_vs_loss_delta.png`
- `weight_rel_mse_vs_kernel_output_rel_mse.png`
