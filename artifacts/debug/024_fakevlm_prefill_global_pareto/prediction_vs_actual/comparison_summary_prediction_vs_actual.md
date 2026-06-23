# FakeVLM Prediction vs Actual Summary

- Quality policies: 40
- Single-linear comparisons: 60
- E2E policy comparisons: 40
- Predicted quality is NLL-based; no downstream accuracy prediction is inferred from NLL cost.

## Quality

| Comparison | MAE | RMSE | MAPE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Absolute NLL | 0.16711326 | 0.28210469 | 0.01246213 | -0.887949 | -0.911312 |
| Clipped NLL delta | 0.02966142 | 0.05114703 | 0.99502336 | -0.241318 | -0.729159 |
| Raw NLL delta | 0.16711326 | 0.28210469 | 1.13933866 | -0.887949 | -0.911312 |

Actual raw NLL deltas: 30 negative, 0 zero, 10 positive.
WARNING: predicted and actual NLL are negatively correlated on selected policies; the fitted NLL cost does not generalize as a calibrated selected-policy loss predictor.

## Single Linear

| Source kind | Rows | MAE ms | RMSE ms | MAPE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| all | 60 | 0.117092 | 0.178254 | 0.115054 | 0.995434 | 0.982495 |
| measured_lookup | 20 | 0.085992 | 0.120863 | 0.177924 | 0.992504 | 0.923308 |
| model_prediction | 40 | 0.132643 | 0.200893 | 0.083619 | 0.995235 | 0.994184 |

## End To End

| Batch | MAE ms | RMSE ms | MAPE | Pearson | Spearman |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.566165 | 6.467040 | 0.058136 | 0.994292 | 1.000000 |
| 2 | 4.057332 | 4.219662 | 0.025930 | 0.999836 | 1.000000 |
| 4 | 2.629646 | 2.807029 | 0.009083 | 0.999342 | 1.000000 |
| 8 | 16.104574 | 18.024092 | 0.026886 | 0.998942 | 1.000000 |
| 16 | 36.692094 | 43.886364 | 0.030228 | 0.999113 | 1.000000 |
