# Debug 045: corrected runtime-quality consolidation

Speed is copied from prior measured vLLM speed closure. Prefill-only quality is from debug 042/043 real vLLM lm-eval; prefill-decoding NLL is from debug 044 real vLLM teacher-forced decoding.
No file under `artifacts/exports/` is modified. `not re-evaluated` rows are historical intermediate speed-only points and are excluded from the corrected NLL plots.

## Generated artifacts

- `prefill_only_corrected_runtime_quality.csv` — all measured prefill-only speed rows with five corrected task metrics.
- `prefill_decode_corrected_runtime_nll.csv` — all existing prefill-decode speed rows with corrected NLL where applicable.
- `prefill_decode_downstream_tasks.csv` — the same prefill-decode rows with all three historical real-vLLM generation-task metric pairs.
- `pareto/` — one speed-quality plot per prefill-only or prefill-decoding metric, with uniform, ours-formal and intermediate styles.

## Prefill-only: measured speed and corrected runtime quality

| model | policy | family | speedup | WikiText PPL ↓ | WinoGrande (%) | ARC-Easy (%) | ARC-Challenge (%) | MMLU (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama2 | dense_bf16 | uniform | 1.000 | 12.228 | 68.27 | 68.39 | 43.34 | 46.57 |
| llama2 | ours_point_004 | ours | 1.016 | 12.334 | 68.67 | 68.22 | 43.77 | 46.32 |
| llama2 | marlin_nvfp4 | uniform | 1.043 | 13.005 | 67.80 | 67.59 | 42.75 | 45.44 |
| llama2 | ours_point_006 | ours | 1.074 | 12.660 | 67.64 | 68.01 | 42.92 | 44.96 |
| llama2 | ours_point_008 | ours | 1.213 | 12.764 | 67.64 | 67.80 | 42.24 | 44.52 |
| llama2 | ours_point_009 | ours | 1.363 | 12.869 | 67.96 | 66.79 | 42.49 | 44.45 |
| llama2 | sparse_bf16 | uniform | 1.633 | 19.869 | 63.85 | 58.21 | 35.07 | 28.24 |
| llama2 | ours_point_011 | ours | 1.635 | 23.295 | 65.11 | 66.20 | 41.98 | 42.66 |
| llama2 | ours_point_012 | ours | 1.726 | 29.662 | 63.69 | 64.27 | 39.42 | 39.48 |
| llama2 | ours_point_013 | ours | 1.811 | 59.490 | 63.38 | 61.53 | 37.46 | 36.70 |
| llama2 | dense_nvfp4 | uniform | 1.867 | 13.214 | 68.35 | 67.21 | 43.00 | 44.37 |
| llama2 | ours_point_015 | ours | 1.953 | 2334.138 | 53.20 | 40.99 | 28.07 | 23.71 |
| llama2 | ours_point_016 | ours | 2.043 | 5923.550 | 53.67 | 37.71 | 25.34 | 22.99 |
| llama2 | sparse_nvfp4 | uniform | 2.074 | 60.347 | 54.06 | 34.30 | 23.29 | 22.88 |
| llama31 | dense_bf16 | uniform | 1.000 | 9.425 | 73.32 | 79.63 | 55.97 | 68.34 |
| llama31 | dense_nvfp4 | uniform | 1.728 | 10.612 | 71.19 | 76.98 | 51.79 | 63.72 |
| llama31 | sparse_bf16 | uniform | 1.550 | 21.362 | 65.90 | 62.88 | 38.40 | 39.72 |
| llama31 | sparse_nvfp4 | uniform | 1.919 | 82.927 | 53.28 | 38.85 | 23.89 | 23.11 |
| llama31 | marlin_nvfp4 | uniform | 1.027 | 10.125 | 73.88 | 79.46 | 54.01 | 65.26 |
| llama31 | ours_point_3 | ours | 1.224 | 9.660 | 74.51 | 79.38 | 55.03 | 67.85 |
| llama31 | ours_point_5 | ours | 1.437 | 9.857 | 73.48 | 78.07 | 53.16 | 66.97 |
| llama31 | ours_point_6 | ours | 1.560 | 9.999 | 71.67 | 76.35 | 52.90 | 66.35 |
| llama31 | ours_point_8 | ours | 1.788 | 180.067 | 56.35 | 60.77 | 40.87 | 47.84 |
| llama31 | ours_point_9 | ours | 1.856 | 291.135 | 52.09 | 55.43 | 35.84 | 40.09 |
| llama31 | ours_point_11 | ours | 2.035 | 423.449 | 49.72 | 44.70 | 28.41 | 30.44 |
| llama31 | ours_point_13 | ours | 2.168 | 1262.346 | 53.12 | 40.36 | 25.85 | 24.77 |

## Prefill-decoding: measured speed and corrected real-vLLM NLL

| model | policy | family | speedup | avg NLL ↓ | PPL ↓ | status |
|---|---|---|---:|---:|---:|---|
| llama2 | dense_bf16 | uniform | 1.000 | 2.005380 | 7.4289 | complete |
| llama2 | dense_nvfp4 | uniform | 1.188 | 2.073926 | 7.9560 | complete |
| llama2 | marlin_nvfp4 | uniform | 1.387 | 2.063444 | 7.8730 | complete |
| llama2 | sparse_bf16 | uniform | 1.442 | 2.401331 | 11.0379 | complete |
| llama2 | sparse_nvfp4 | uniform | 1.185 | 3.257294 | 25.9792 | complete |
| llama2 | point_000 | ours | 1.000 | 2.005380 | 7.4289 | complete |
| llama2 | point_001 | ours | 1.012 | 2.007059 | 7.4414 | complete |
| llama2 | point_002 | ours | 1.010 | 2.008092 | 7.4491 | complete |
| llama2 | point_003 | ours | 1.032 | 2.006944 | 7.4405 | complete |
| llama2 | point_004 | ours | 1.017 | 2.005334 | 7.4286 | complete |
| llama2 | point_005 | ours | 1.058 | 2.006654 | 7.4384 | complete |
| llama2 | point_006 | ours | 1.061 | 2.010407 | 7.4664 | complete |
| llama2 | point_007 | ours | 1.165 | 2.015685 | 7.5059 | complete |
| llama2 | point_008 | ours | 1.179 | 2.011935 | 7.4778 | complete |
| llama2 | point_009 | ours | 1.071 | 2.020842 | 7.5447 | complete |
| llama2 | point_010 | ours | 1.356 | 2.028548 | 7.6030 | complete |
| llama2 | point_011 | ours | 1.714 | 2.028795 | 7.6049 | complete |
| llama2 | i34 | ours-intermediate | 1.451 | — | — | not re-evaluated |
| llama2 | i36 | ours-intermediate | 1.627 | — | — | not re-evaluated |
| llama2 | i37 | ours-intermediate | 1.659 | — | — | not re-evaluated |
| llama2 | i38 | ours-intermediate | 1.680 | — | — | not re-evaluated |
| llama31 | dense_bf16 | uniform | 1.000 | 1.907542 | 6.7365 | complete |
| llama31 | dense_nvfp4 | uniform | 1.058 | 1.997947 | 7.3739 | complete |
| llama31 | marlin_nvfp4 | uniform | 1.158 | 1.954747 | 7.0621 | complete |
| llama31 | sparse_bf16 | uniform | 1.173 | 2.547946 | 12.7808 | complete |
| llama31 | sparse_nvfp4 | uniform | 0.791 | 3.463238 | 31.9202 | complete |
| llama31 | point_000 | ours | 1.000 | 1.907542 | 6.7365 | complete |
| llama31 | point_002 | ours | 1.096 | 1.912755 | 6.7717 | complete |
| llama31 | point_004 | ours | 1.267 | 1.913116 | 6.7742 | complete |
| llama31 | point_006 | ours | 1.484 | 1.927893 | 6.8750 | complete |
| llama31 | point_008 | ours | 1.585 | 1.950129 | 7.0296 | complete |
| llama31 | point_009_max_speed | ours | 1.692 | 1.951015 | 7.0358 | complete |

## Prefill-decoding: existing measured downstream generation tasks

CNN/DM and DialogSum use ROUGE-L / BERTScore; IWSLT uses ROUGE-L / BLEU. These generation-task values are inherited unchanged from the prior measured vLLM task runs; they are not proxy-quality values.

| model | policy | speedup | CNN R-L | CNN BERTScore | DSum R-L | DSum BERTScore | IWSLT R-L | IWSLT BLEU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama2 | dense_bf16 | 1.000 | 23.671 | 87.185 | 21.688 | 87.176 | 46.701 | 19.296 |
| llama2 | dense_nvfp4 | 1.188 | 24.273 | 87.204 | 20.592 | 86.691 | 44.969 | 16.835 |
| llama2 | marlin_nvfp4 | 1.387 | 24.579 | 87.251 | 21.368 | 87.053 | 46.772 | 18.200 |
| llama2 | sparse_bf16 | 1.442 | 15.350 | 82.131 | 13.539 | 75.392 | 15.798 | 3.900 |
| llama2 | sparse_nvfp4 | 1.185 | 2.047 | 9.626 | 9.388 | 61.906 | 1.133 | 0.238 |
| llama2 | point_000 | 1.000 | 23.610 | 87.185 | 21.598 | 87.159 | 46.909 | 19.412 |
| llama2 | point_001 | 1.012 | 23.771 | 87.205 | 21.515 | 87.147 | 47.014 | 19.437 |
| llama2 | point_002 | 1.010 | 23.623 | 87.177 | 21.760 | 87.174 | 47.087 | 19.565 |
| llama2 | point_003 | 1.032 | 23.776 | 87.197 | 21.385 | 87.146 | 47.773 | 20.238 |
| llama2 | point_004 | 1.017 | 23.881 | 87.217 | 21.525 | 87.202 | 47.242 | 19.609 |
| llama2 | point_005 | 1.058 | 23.820 | 87.190 | 21.713 | 87.195 | 46.493 | 19.126 |
| llama2 | point_006 | 1.061 | 23.834 | 87.206 | 21.412 | 87.143 | 44.323 | 18.013 |
| llama2 | point_007 | 1.165 | 23.648 | 87.167 | 21.863 | 87.198 | 46.198 | 19.256 |
| llama2 | point_008 | 1.179 | 23.819 | 87.187 | 21.643 | 87.103 | 45.432 | 18.920 |
| llama2 | point_009 | 1.071 | 23.770 | 87.201 | 21.419 | 87.154 | 44.423 | 18.419 |
| llama2 | point_010 | 1.356 | 23.424 | 87.054 | 21.300 | 87.133 | 43.473 | 16.816 |
| llama2 | point_011 | 1.714 | 23.544 | 87.081 | 21.581 | 87.153 | 45.309 | 18.301 |
| llama2 | i34 | 1.451 | 23.752 | 87.088 | 21.703 | 87.171 | 45.410 | 18.304 |
| llama2 | i36 | 1.627 | 23.833 | 87.103 | 21.625 | 87.153 | 46.374 | 18.949 |
| llama2 | i37 | 1.659 | 23.843 | 87.151 | 21.685 | 87.172 | 46.025 | 18.639 |
| llama2 | i38 | 1.680 | 23.754 | 87.117 | 21.633 | 87.140 | 46.298 | 19.029 |
| llama31 | dense_bf16 | 1.000 | 19.487 | 83.600 | 13.784 | 78.187 | 28.146 | 10.680 |
| llama31 | dense_nvfp4 | 1.058 | 16.345 | 84.988 | 8.203 | 81.634 | 28.408 | 10.312 |
| llama31 | marlin_nvfp4 | 1.158 | 16.108 | 85.061 | 8.805 | 81.439 | 28.805 | 10.403 |
| llama31 | sparse_bf16 | 1.173 | 12.834 | 81.278 | 4.049 | 79.128 | 14.322 | 3.276 |
| llama31 | sparse_nvfp4 | 0.791 | 0.804 | 76.729 | 0.126 | 78.438 | 0.110 | 0.014 |
| llama31 | point_000 | 1.000 | 19.487 | 83.600 | 13.784 | 78.187 | 28.146 | 10.680 |
| llama31 | point_002 | 1.096 | 20.274 | 83.245 | 13.473 | 78.400 | 28.098 | 10.586 |
| llama31 | point_004 | 1.267 | 18.747 | 83.837 | 13.266 | 78.585 | 27.893 | 10.654 |
| llama31 | point_006 | 1.484 | 16.675 | 84.399 | 11.122 | 80.727 | 28.405 | 10.582 |
| llama31 | point_008 | 1.585 | 16.274 | 84.040 | 8.957 | 81.619 | 28.343 | 10.546 |
| llama31 | point_009_max_speed | 1.692 | 16.840 | 84.047 | 9.085 | 81.463 | 28.846 | 10.570 |
