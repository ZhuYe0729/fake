# Llama2 Prefill-Only Loss Modeling Summary

This report summarizes local linear errors and WikiText-2 mean CE loss deltas for prefill-only precision modeling.

## Validity Note

This summary is **not valid for final prefill NVFP4 quality modeling**.
The underlying 014 scripts evaluate prepared dense replacement weights through
PyTorch `F.linear`; they do not run the real `dense_nvfp4` / `sparse_nvfp4`
CUTLASS kernels and therefore omit runtime activation quantization. Treat these
numbers only as offline weight-error diagnostics.

## Best Local Error Proxies

| metric | Pearson | Spearman | n |
|---|---:|---:|---:|
| output_rel_mse | 0.3703 | 0.5715 | 896 |
| output_rmse_over_rms | 0.3603 | 0.5715 | 896 |
| weight_rel_mse | 0.3425 | 0.4692 | 896 |
| weight_rmse_over_rms | 0.3381 | 0.4692 | 896 |
| weight_mse | 0.2618 | 0.4519 | 896 |
| output_mse | 0.2024 | 0.4242 | 896 |

## Highest Loss-Delta Layers

| method | layer | mean loss delta |
|---|---:|---:|
| sparse_nvfp4 | 30 | 0.042433 |
| sparse_nvfp4 | 31 | 0.034548 |
| sparse_nvfp4 | 1 | 0.030564 |
| sparse_nvfp4 | 29 | 0.030462 |
| sparse_bf16 | 30 | 0.026695 |
| dense_nvfp4 | 1 | 0.022374 |
| marlin_nvfp4 | 1 | 0.022374 |
| sparse_nvfp4 | 2 | 0.019546 |

## Highest Loss-Delta Linear Types

| method | type | mean loss delta |
|---|---|---:|
| sparse_nvfp4 | down_proj | 0.190230 |
| sparse_nvfp4 | up_proj | 0.146602 |
| sparse_nvfp4 | gate_proj | 0.119231 |
| sparse_nvfp4 | v_proj | 0.070731 |
| sparse_bf16 | down_proj | 0.063874 |
| sparse_bf16 | up_proj | 0.063017 |
| sparse_bf16 | v_proj | 0.058727 |
| sparse_bf16 | gate_proj | 0.043626 |

## Outputs

- `local_error_loss_correlations.csv`
- `layer_loss_summary.csv`
- `linear_type_loss_summary.csv`
- `layer_depth_loss_delta.png`
- `linear_type_loss_delta.png`
- `best_local_proxy_vs_loss_delta.png`
- `weight_rel_mse_vs_output_rel_mse.png`
