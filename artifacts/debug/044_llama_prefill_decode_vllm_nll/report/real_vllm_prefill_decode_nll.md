# Real vLLM teacher-forced prefill-decoding NLL

All rows use 2048 prefill tokens + 80 teacher-forced decode tokens on 32 WikiText blocks (2560 scored tokens).
This report is isolated under debug 044 and does not replace historical proxy-NLL artifacts.

Formal-point aliases: point_000 is policy-identical to dense BF16 for both models; Llama2 point_011 is policy-identical to its max-speed result.

| model | family | policy | avg NLL | perplexity | phase trace |
|---|---|---|---:|---:|---|
| llama2 | ours-formal-pareto | point_001_runtime | 2.007059 | 7.4414 | yes |
| llama2 | ours-formal-pareto | point_002_runtime | 2.008092 | 7.4491 | yes |
| llama2 | ours-formal-pareto | point_003_runtime | 2.006944 | 7.4405 | yes |
| llama2 | ours-formal-pareto | point_004_runtime | 2.005334 | 7.4286 | yes |
| llama2 | ours-formal-pareto | point_005_runtime | 2.006654 | 7.4384 | yes |
| llama2 | ours-formal-pareto | point_006_runtime | 2.010407 | 7.4664 | yes |
| llama2 | ours-formal-pareto | point_007_runtime | 2.015685 | 7.5059 | yes |
| llama2 | ours-formal-pareto | point_008_runtime | 2.011935 | 7.4778 | yes |
| llama2 | ours-formal-pareto | point_009_runtime | 2.020842 | 7.5447 | yes |
| llama2 | ours-formal-pareto | point_010_runtime | 2.028548 | 7.6030 | yes |
| llama2 | ours-max-speed | max_speed | 2.028795 | 7.6049 | yes |
| llama2 | uniform | dense_bf16 | 2.005380 | 7.4289 | — |
| llama2 | uniform | dense_nvfp4 | 2.073926 | 7.9560 | — |
| llama2 | uniform | marlin_nvfp4 | 2.063444 | 7.8730 | — |
| llama2 | uniform | sparse_bf16 | 2.401331 | 11.0379 | — |
| llama2 | uniform | sparse_nvfp4 | 3.257294 | 25.9792 | — |
| llama31 | ours-formal-pareto | point_001_runtime | 1.911290 | 6.7618 | yes |
| llama31 | ours-formal-pareto | point_002 | 1.912755 | 6.7717 | yes |
| llama31 | ours-formal-pareto | point_003_runtime | 1.912856 | 6.7724 | yes |
| llama31 | ours-formal-pareto | point_004 | 1.913116 | 6.7742 | yes |
| llama31 | ours-formal-pareto | point_005_runtime | 1.924174 | 6.8495 | yes |
| llama31 | ours-formal-pareto | point_006_retry | 1.927893 | 6.8750 | yes |
| llama31 | ours-formal-pareto | point_007_runtime | 1.933669 | 6.9148 | yes |
| llama31 | ours-formal-pareto | point_008 | 1.950129 | 7.0296 | yes |
| llama31 | ours-formal-pareto | point_009_runtime | 1.951015 | 7.0358 | yes |
| llama31 | ours-max-speed | max_speed | 1.951015 | 7.0358 | yes |
| llama31 | uniform | dense_bf16 | 1.907542 | 6.7365 | — |
| llama31 | uniform | dense_nvfp4 | 1.997947 | 7.3739 | — |
| llama31 | uniform | marlin_nvfp4 | 1.954747 | 7.0621 | — |
| llama31 | uniform | sparse_bf16 | 2.547946 | 12.7808 | — |
| llama31 | uniform | sparse_nvfp4 | 3.463238 | 31.9202 | — |
