# Sparse NVFP4 Empirical Scenario Ablation

Evaluation set: raw-local-matched sparse NVFP4 empirical structural pairs.

| variant | pairs | Pearson | Spearman | MAE | RMSE | direction acc | pred delta mean | measured delta mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local_only | 12 | -0.3628 | -0.4126 | 0.100423 | 0.113619 | 0.5833 | 0.000009 | 0.100431 |
| local_depth | 12 | -0.1861 | -0.0979 | 0.064907 | 0.076831 | 1.0000 | 0.150933 | 0.100431 |
| local_type | 12 | 0.1046 | 0.1608 | 0.049199 | 0.055502 | 1.0000 | 0.115430 | 0.100431 |
| final_depth_type | 12 | -0.2846 | -0.2028 | 0.110675 | 0.121463 | 1.0000 | 0.208993 | 0.100431 |

## Plots

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_ablation_loss_delta.png`
- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_ablation_metrics.png`
