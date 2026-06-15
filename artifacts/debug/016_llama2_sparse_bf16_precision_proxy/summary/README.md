# Sparse BF16 Precision Proxy Summary

Formula: `bias + sum(local_error * layer_coef[layer] * type_coef[linear_type])`
Local error metric: `output_rel_mse`

## Metrics

| split | rows | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| train | 84 | 0.9846 | 0.9820 | 0.026221 | 0.030616 |
| holdout | 36 | 0.9870 | 0.9822 | 0.021816 | 0.025729 |
| all | 120 | 0.9851 | 0.9833 | 0.024900 | 0.029236 |

## Main Plot

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/plots/holdout_proxy_vs_loss_delta.png`
