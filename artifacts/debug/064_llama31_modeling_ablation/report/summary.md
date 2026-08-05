# Llama3.1 modeling ablation

The experiment uses only frozen RTX 5090 measurements from 057: 54/18 quality policies and 7/5 speed policies.

## Latency model

### Strict unseen-(N,K) kernel generalization

| Variant | Samples | MAPE | p90 APE | Spearman |
|---|---:|---:|---:|---:|
| Direct shape fit | 4657 | 14.80% | 37.47% | 0.957 |
| + Calibrated Roofline prior | 4657 | 10.78% | 27.92% | 0.962 |
| + Shape residual correction | 4657 | 8.01% | 22.03% | 0.972 |

All three variants use the same shape features, ridge strength, local-neighbor mixture, seeds, and 25% held-out (N,K) groups. The direct-fit baseline predicts log latency; the other variants add the method-aware Roofline prior and its residual correction.

### Independent Llama3 module profiles

| Variant | Samples | MAE (ms) | MAPE | Spearman |
|---|---:|---:|---:|---:|
| Roofline | 15 | 1.0887 | 19.99% | 0.850 |
| + Shape correction | 15 | 0.6449 | 8.29% | 0.989 |

Shape correction reduces local MAPE by 58.5% relative to the method-aware Roofline baseline.

### Held-out heterogeneous policies

| Variant | MAE (ms) | MAPE | Spearman |
|---|---:|---:|---:|
| Raw E2E aggregation | 499.16 | 18.17% | 0.300 |
| + Monotonic E2E calibration | 224.24 | 8.30% | 0.000 |

With the refined kernel predictor fixed, the PAVA monotonic E2E calibration reduces policy-level MAPE by 54.3%, from 18.17% to 8.30%.

## Quality model

| Variant | MAE | RMSE | Spearman |
|---|---:|---:|---:|
| Local sum | 0.0571 | 0.0711 | 0.920 |
| + Structure-aware | 0.0468 | 0.0568 | 0.944 |
| + Phase-aware | 0.0413 | 0.0529 | 0.963 |

The full quality model reduces held-out MAE by 27.8% relative to direct local-error summation.

## Interpretation

The rows are fixed before inspecting holdout labels. Non-monotonic intermediate results, if any, are retained rather than hidden or retuned on the holdout set.
