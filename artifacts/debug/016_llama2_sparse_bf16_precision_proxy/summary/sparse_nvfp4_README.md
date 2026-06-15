# sparse_nvfp4 Kernel Precision Proxy Summary

Formula: `bias + sum(kernel_local_error * layer_coef[layer] * type_coef[linear_type])`
Local error metric: `output_rel_mse`

## Metrics

| split | rows | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| train | 84 | 0.9828 | 0.9815 | 0.071906 | 0.084557 |
| holdout | 36 | 0.9801 | 0.9778 | 0.076802 | 0.089538 |
| all | 120 | 0.9818 | 0.9835 | 0.073375 | 0.086081 |

## Main Plot

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/plots/holdout_sparse_nvfp4_proxy_vs_loss_delta.png`
