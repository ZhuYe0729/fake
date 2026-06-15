# dense_nvfp4 Kernel Precision Proxy Summary

Formula: `bias + sum(kernel_local_error * layer_coef[layer] * type_coef[linear_type])`
Local error metric: `output_rel_mse`

Dense calibration: `calibrated_pred = c0 + c1 * base_pred + c2 * base_pred^2 + c3 * log1p(selected_modules)`

## Metrics

| split | rows | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| train | 84 | 0.9739 | 0.9595 | 0.004405 | 0.007410 |
| holdout | 36 | 0.9680 | 0.9361 | 0.004917 | 0.007741 |
| all | 120 | 0.9722 | 0.9596 | 0.004559 | 0.007511 |

## Main Plot

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/plots/holdout_dense_nvfp4_proxy_vs_loss_delta.png`
