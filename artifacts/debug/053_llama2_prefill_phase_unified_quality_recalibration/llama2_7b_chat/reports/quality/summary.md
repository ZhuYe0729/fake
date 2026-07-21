# Llama2 phase-unified quality recalibration

The proxy formula and 54/18 split are unchanged from 046; compressed labels are rebuilt with phase runtime.

| split | 046 MAE | 053 MAE | 046 RMSE | 053 RMSE | 046 Spearman | 053 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.107590 | 0.242819 | 0.267225 | 0.335293 | 0.9097 | 0.9115 |
| holdout | 0.121419 | 0.425113 | 0.149305 | 0.545739 | 0.8204 | 0.6945 |

## Labels changed by the corrected pipeline

| policy | legacy ΔNLL | phase-unified ΔNLL | difference |
|---|---:|---:|---:|
| p01 | 0.053822 | 0.042103 | -0.011719 |
| p02 | 0.345707 | 1.801113 | +1.455406 |
| p03 | 1.147809 | 6.120044 | +4.972235 |
| p04 | 0.037336 | 0.025887 | -0.011449 |

## Uniform controls

| policy | measured ΔNLL | predicted ΔNLL | residual |
|---|---:|---:|---:|
| p00 | 0.000000 | -0.198625 | 0.198625 |
| p01 | 0.042103 | -0.019943 | 0.062045 |
| p02 | 1.801113 | 1.406927 | 0.394185 |
| p03 | 6.120044 | 5.089247 | 1.030797 |
| p04 | 0.025887 | -0.000545 | 0.026431 |
