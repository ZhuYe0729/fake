# Stratified Residual Proxy Summary

Residual metrics subtract a train-fitted baseline using only `selected_modules` and `raw_error_sum`.

| method | variant | rows | baseline RMSE | proxy RMSE | residual Pearson | residual Spearman | residual MAE | residual RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense_nvfp4 | dense_calibrated_reference | 24 | 0.009873 | 0.009639 | 0.2257 | 0.2357 | 0.007535 | 0.009639 |
| dense_nvfp4 | final_layer_type | 24 | 0.009873 | 0.041492 | 0.0492 | -0.1383 | 0.036401 | 0.041492 |
| dense_nvfp4 | local_layer | 24 | 0.009873 | 0.041322 | 0.0405 | -0.1417 | 0.036368 | 0.041322 |
| dense_nvfp4 | local_only | 24 | 0.009873 | 0.040576 | 0.0549 | -0.1461 | 0.035786 | 0.040576 |
| dense_nvfp4 | local_type | 24 | 0.009873 | 0.040905 | 0.0586 | -0.1296 | 0.035895 | 0.040905 |
| sparse_bf16 | final_layer_type | 24 | 0.015006 | 0.051208 | -0.3414 | -0.3583 | 0.043086 | 0.051208 |
| sparse_bf16 | local_layer | 24 | 0.015006 | 0.050189 | -0.3113 | -0.3539 | 0.042650 | 0.050189 |
| sparse_bf16 | local_only | 24 | 0.015006 | 0.046643 | -0.3571 | -0.4035 | 0.039234 | 0.046643 |
| sparse_bf16 | local_type | 24 | 0.015006 | 0.048106 | -0.3726 | -0.4087 | 0.040022 | 0.048106 |
| sparse_nvfp4 | final_layer_type | 24 | 0.064134 | 0.064005 | 0.3039 | 0.1348 | 0.049960 | 0.064005 |
| sparse_nvfp4 | local_layer | 24 | 0.064134 | 0.068415 | 0.0902 | -0.1035 | 0.053208 | 0.068415 |
| sparse_nvfp4 | local_only | 24 | 0.064134 | 0.067106 | 0.0307 | -0.2809 | 0.051177 | 0.067106 |
| sparse_nvfp4 | local_type | 24 | 0.064134 | 0.064875 | 0.2023 | 0.0296 | 0.049625 | 0.064875 |
