# Sparse NVFP4 Balanced Scenario Scale-Calibrated Ablation

Each variant receives the same scale-only calibration `pred_delta *= a` without an intercept.

Main result: `final_depth_type` has the lowest MAE, while `local_only` is the weakest baseline by MAE/RMSE and direction accuracy.

- MAE rank: final_depth_type, local_depth, local_type, local_only
- Direction rank: local_depth, local_type, final_depth_type, local_only

| variant | pairs | scale | Pearson | Spearman | MAE | RMSE | direction acc | pred delta mean | measured delta mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_only | 12 | -535.1962 | 0.0824 | -0.1259 | 0.032580 | 0.040727 | 0.5833 | 0.002819 | 0.030578 |
| local_depth | 12 | 0.4731 | 0.1197 | -0.0629 | 0.020572 | 0.028728 | 0.9167 | 0.030675 | 0.030578 |
| local_type | 12 | 0.5280 | 0.1907 | 0.0769 | 0.021295 | 0.028363 | 0.9167 | 0.030714 | 0.030578 |
| final_depth_type | 12 | 0.3060 | -0.0363 | -0.1119 | 0.020454 | 0.028880 | 0.9167 | 0.030576 | 0.030578 |

## Plot

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_calibrated_ablation_metrics.png`
