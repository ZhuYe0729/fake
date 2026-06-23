# FakeVLM Prediction vs Actual Summary

- Quality policies: 8
- Single-linear comparisons: 12
- E2E policy comparisons: 8
- Predicted quality is NLL-based; no downstream accuracy prediction is inferred from NLL cost.

## Quality

| Comparison | MAE | RMSE | MAPE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Absolute NLL | 0.07562477 | 0.16457191 | 0.06651910 | 0.997829 | 0.928571 |
| Clipped NLL delta | 0.07562477 | 0.16457191 | 2.81572617 | 0.997829 | 0.928571 |
| Raw NLL delta | 0.07562477 | 0.16457191 | 2.81572617 | 0.997829 | 0.928571 |

Actual raw NLL deltas: 0 negative, 0 zero, 8 positive.

## Single Linear

| Source kind | Rows | MAE ms | RMSE ms | MAPE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| all | 12 | 0.219205 | 0.310190 | 0.054075 | 0.990932 | 0.993007 |
| measured_lookup | 4 | 0.048691 | 0.055403 | 0.027740 | 0.998455 | 1.000000 |
| model_prediction | 8 | 0.304463 | 0.377879 | 0.067242 | 0.982804 | 1.000000 |

## End To End

| Batch | MAE ms | RMSE ms | MAPE | Pearson | Spearman |
|---:|---:|---:|---:|---:|---:|
| 16 | 47.481845 | 50.658997 | 0.041704 | 0.999689 | 1.000000 |
