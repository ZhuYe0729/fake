# Precision Proxy Ablation Summary

## Holdout Metrics

| method | variant | rows | Pearson | Spearman | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| dense_nvfp4 | local_only | 36 | 0.9685 | 0.9317 | 0.030699 | 0.036109 |
| dense_nvfp4 | local_layer | 36 | 0.9688 | 0.9335 | 0.030714 | 0.036086 |
| dense_nvfp4 | local_type | 36 | 0.9687 | 0.9325 | 0.029649 | 0.034950 |
| dense_nvfp4 | final_layer_type | 36 | 0.9689 | 0.9363 | 0.029674 | 0.034916 |
| dense_nvfp4 | dense_calibrated_reference | 36 | 0.9680 | 0.9361 | 0.004917 | 0.007741 |
| sparse_bf16 | local_only | 36 | 0.9866 | 0.9825 | 0.022496 | 0.026563 |
| sparse_bf16 | local_layer | 36 | 0.9865 | 0.9804 | 0.023090 | 0.027075 |
| sparse_bf16 | local_type | 36 | 0.9870 | 0.9807 | 0.022221 | 0.026154 |
| sparse_bf16 | final_layer_type | 36 | 0.9870 | 0.9822 | 0.021816 | 0.025729 |
| sparse_nvfp4 | local_only | 36 | 0.9801 | 0.9784 | 0.077615 | 0.089607 |
| sparse_nvfp4 | local_layer | 36 | 0.9797 | 0.9789 | 0.077808 | 0.090347 |
| sparse_nvfp4 | local_type | 36 | 0.9801 | 0.9791 | 0.077454 | 0.089439 |
| sparse_nvfp4 | final_layer_type | 36 | 0.9801 | 0.9778 | 0.076802 | 0.089538 |

## Plots

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/ablation/proxy_ablation_holdout_spearman.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/ablation/proxy_ablation_holdout_rmse.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/ablation/proxy_ablation_holdout_predictions_sparse_bf16.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/ablation/proxy_ablation_holdout_predictions_dense_nvfp4.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/ablation/proxy_ablation_holdout_predictions_sparse_nvfp4.png`
