# Precision Proxy Ablation Summary

## Holdout Metrics

| method | variant | rows | Pearson | Spearman | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| sparse_nvfp4 | local_only | 24 | 0.9318 | 0.9278 | 0.051177 | 0.067106 |
| sparse_nvfp4 | local_layer | 24 | 0.9327 | 0.9409 | 0.054265 | 0.068979 |
| sparse_nvfp4 | local_type | 24 | 0.9424 | 0.9487 | 0.047822 | 0.063349 |
| sparse_nvfp4 | final_layer_type | 24 | 0.9458 | 0.9643 | 0.050428 | 0.063783 |

## Plots

- `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/stratified_global_ablation/proxy_ablation_holdout_spearman.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/stratified_global_ablation/proxy_ablation_holdout_rmse.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/stratified_global_ablation/proxy_ablation_holdout_predictions_sparse_bf16.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/stratified_global_ablation/proxy_ablation_holdout_predictions_dense_nvfp4.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/017_global_coef_structural_ablation/stratified_global_ablation/proxy_ablation_holdout_predictions_sparse_nvfp4.png`
