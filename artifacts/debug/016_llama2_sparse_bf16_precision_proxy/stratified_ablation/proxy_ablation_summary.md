# Precision Proxy Ablation Summary

## Holdout Metrics

| method | variant | rows | Pearson | Spearman | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| dense_nvfp4 | local_only | 24 | 0.8296 | 0.8061 | 0.035786 | 0.040576 |
| dense_nvfp4 | local_layer | 24 | 0.8234 | 0.8017 | 0.036368 | 0.041322 |
| dense_nvfp4 | local_type | 24 | 0.8325 | 0.8183 | 0.035895 | 0.040905 |
| dense_nvfp4 | final_layer_type | 24 | 0.8287 | 0.7930 | 0.036401 | 0.041492 |
| dense_nvfp4 | dense_calibrated_reference | 24 | 0.8542 | 0.8765 | 0.007535 | 0.009639 |
| sparse_bf16 | local_only | 24 | 0.9426 | 0.9017 | 0.039234 | 0.046643 |
| sparse_bf16 | local_layer | 24 | 0.9485 | 0.9157 | 0.042650 | 0.050189 |
| sparse_bf16 | local_type | 24 | 0.9402 | 0.9243 | 0.040022 | 0.048106 |
| sparse_bf16 | final_layer_type | 24 | 0.9448 | 0.9270 | 0.043086 | 0.051208 |
| sparse_nvfp4 | local_only | 24 | 0.9318 | 0.9278 | 0.051177 | 0.067106 |
| sparse_nvfp4 | local_layer | 24 | 0.9331 | 0.9357 | 0.053208 | 0.068415 |
| sparse_nvfp4 | local_type | 24 | 0.9388 | 0.9461 | 0.049625 | 0.064875 |
| sparse_nvfp4 | final_layer_type | 24 | 0.9449 | 0.9591 | 0.049960 | 0.064005 |

## Plots

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/stratified_ablation/proxy_ablation_holdout_spearman.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/stratified_ablation/proxy_ablation_holdout_rmse.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/stratified_ablation/proxy_ablation_holdout_predictions_sparse_bf16.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/stratified_ablation/proxy_ablation_holdout_predictions_dense_nvfp4.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/stratified_ablation/proxy_ablation_holdout_predictions_sparse_nvfp4.png`
