# Sparse NVFP4 Empirical Scenario Ablation

Evaluation set: raw-local-matched sparse NVFP4 empirical structural pairs.

| variant | pairs | Pearson | Spearman | MAE | RMSE | direction acc | pred delta mean | measured delta mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local_only | 12 | -0.0824 | 0.1259 | 0.032965 | 0.042064 | 0.4167 | -0.000005 | 0.030578 |
| local_depth | 12 | 0.1197 | -0.0629 | 0.040960 | 0.044670 | 0.9167 | 0.064834 | 0.030578 |
| local_type | 12 | 0.1907 | 0.0769 | 0.036157 | 0.039692 | 0.9167 | 0.058174 | 0.030578 |
| final_depth_type | 12 | -0.0363 | -0.1119 | 0.069338 | 0.075114 | 0.9167 | 0.099916 | 0.030578 |

## Plots

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_ablation_loss_delta.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_ablation_metrics.png`
