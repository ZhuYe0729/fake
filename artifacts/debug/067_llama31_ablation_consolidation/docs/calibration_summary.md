# Quality calibration-size sensitivity

The fixed 18-policy holdout is never used for fitting or subset construction. The original 54 training policies and 18 calibration-only extensions form one exchangeable 72-policy candidate pool. All subsets contain the five dense/uniform anchors, and ten fixed random nested orders provide the median and IQR. The exact frozen 54-policy main model is reported separately and is not used to shape the sensitivity curve.

| Calibration policies | Random median MAE | Random IQR | Random median Spearman | GPU-min |
|---:|---:|---:|---:|---:|
| 5 | 0.0614 | [0.0614, 0.0614] | 0.924 | 3.3 |
| 8 | 0.0585 | [0.0562, 0.0596] | 0.940 | 6.4 |
| 12 | 0.0532 | [0.0514, 0.0542] | 0.951 | 10.3 |
| 18 | 0.0473 | [0.0463, 0.0529] | 0.959 | 16.4 |
| 27 | 0.0478 | [0.0443, 0.0500] | 0.959 | 25.8 |
| 36 | 0.0460 | [0.0417, 0.0487] | 0.964 | 34.8 |
| 45 | 0.0403 | [0.0390, 0.0413] | 0.960 | 43.9 |
| 54 | 0.0408 | [0.0375, 0.0424] | 0.959 | 53.0 |
| 60 | 0.0392 | [0.0358, 0.0430] | 0.959 | 59.2 |
| 64 | 0.0394 | [0.0373, 0.0423] | 0.959 | 63.5 |
| 66 | 0.0387 | [0.0351, 0.0422] | 0.959 | 65.3 |
| 72 | 0.0340 | [0.0340, 0.0340] | 0.959 | 71.4 |

The preregistered 5%-from-full rule identifies **72 policies** as the observed elbow. The random-subset median holdout MAE improves by 44.7% from 5 to 72 policies.

The frozen main 54-policy configuration has holdout MAE 0.0413, costs 48.6 recorded GPU-minutes, and reproduces 064 within the fixed numerical tolerance. The random-pool median at 54 policies is 0.0408; it must not be confused with the particular frozen main subset.

Using the full 72-policy pool costs 71.4 GPU-minutes: 46.7% more labeling cost than the frozen main subset and 17.6% lower MAE than that particular subset.

The curve is reported as observed; no subset, holdout, or elbow threshold was retuned after inspecting the result.
The paper-facing figure caps the displayed calibration budget at 64 policies. The complete 72-policy audit curve is retained as `calibration_size_sensitivity_full.png` and the table above reports every measured point.
