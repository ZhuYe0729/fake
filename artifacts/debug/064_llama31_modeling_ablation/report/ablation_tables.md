# Compact ablation tables

## Latency prediction

| Variant | Samples | MAPE | p90 APE | Spearman |
|---|---:|---:|---:|---:|
| Direct shape fit | 4657 | 14.80% | 37.47% | 0.957 |
| + Calibrated Roofline prior | 4657 | 10.78% | 27.92% | 0.962 |
| + Shape residual correction | 4657 | 8.01% | 22.03% | 0.972 |

| Variant | MAE (ms) | MAPE | Spearman |
|---|---:|---:|---:|
| Raw E2E aggregation | 499.16 | 18.17% | 0.300 |
| + Monotonic E2E calibration | 224.24 | 8.30% | 0.000 |

## Quality prediction

| Variant | MAE | RMSE | Spearman |
|---|---:|---:|---:|
| Local sum | 0.0571 | 0.0711 | 0.920 |
| + Structure-aware | 0.0468 | 0.0568 | 0.944 |
| + Phase-aware | 0.0413 | 0.0529 | 0.963 |

Kernel rows report MAPE on unseen (N,K) benchmark groups; policy rows report MAPE on five held-out heterogeneous policies; quality rows report ΔNLL MAE on 18 held-out policies.
