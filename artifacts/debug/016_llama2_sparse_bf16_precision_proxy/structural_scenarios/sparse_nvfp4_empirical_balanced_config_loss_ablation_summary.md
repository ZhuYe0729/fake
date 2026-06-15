# Sparse NVFP4 Balanced Config-Level Loss Prediction Ablation

Each ablation variant is fitted on the stratified sparse NVFP4 samples and evaluated on the balanced structural scenario configs.

| variant | configs | Pearson | Spearman | MAE | RMSE | pred delta mean | measured delta mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_only | 24 | 0.0331 | 0.1252 | 0.048739 | 0.053834 | 0.234392 | 0.188080 |
| local_depth | 24 | 0.5534 | 0.5522 | 0.048262 | 0.057711 | 0.234575 | 0.188080 |
| local_type | 24 | 0.5976 | 0.6896 | 0.041896 | 0.049542 | 0.227970 | 0.188080 |
| final_depth_type | 24 | 0.5713 | 0.5513 | 0.049704 | 0.064567 | 0.230327 | 0.188080 |

## Plots

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_config_loss_ablation_scatter.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_config_loss_ablation_metrics.png`
