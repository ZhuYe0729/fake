# Llama2 canonical sparse quality recalibration

| split | MAE | RMSE | Spearman |
|---|---:|---:|---:|
| train | 0.087599 | 0.108660 | 0.8277 |
| holdout | 0.089391 | 0.104758 | 0.7523 |

## Uniform controls

| policy | measured ΔNLL | predicted ΔNLL | residual |
|---|---:|---:|---:|
| p00 | 0.000000 | -0.086675 | 0.086675 |
| p01 | 0.042103 | 0.012799 | 0.029303 |
| p02 | 0.345707 | 0.644147 | -0.298440 |
| p03 | 1.017080 | 1.152322 | -0.135242 |
| p04 | 0.025887 | 0.015421 | 0.010466 |
