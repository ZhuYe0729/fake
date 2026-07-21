# Llama2-7B-Chat measured result table

Every row contains a measured end-to-end speed and corresponding measured task score. All measured ours points are retained, including each max-speed endpoint; `recommended` is a paper-candidate suggestion rather than a filter.

| scenario | family | policy | recommended use | E2E ms | speedup | ARC norm. | CNN R-L | CNN BERTScore | DSum R-L | DSum BERTScore | IWSLT R-L | IWSLT BLEU | ΔNLL | task status | speed source |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| prefill-only (B=8, S=2048) | uniform | dense_bf16 | baseline | 1079.54 | 1.000 | 43.345 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_004 |  | 1062.20 | 1.016 | 43.259 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | uniform | marlin_nvfp4 |  | 1034.99 | 1.043 | 42.833 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_006 |  | 1005.59 | 1.074 | 43.174 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_008 | recommended: high-quality | 890.02 | 1.213 | 43.345 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_009 |  | 791.93 | 1.363 | 43.089 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | uniform | sparse_bf16 |  | 660.95 | 1.633 | 35.324 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_011 |  | 660.34 | 1.635 | 44.027 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_012 | recommended: primary balanced | 625.45 | 1.726 | 44.198 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_013 | recommended: dense-NVFP4 cover | 596.00 | 1.811 | 43.430 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | uniform | dense_nvfp4 |  | 578.16 | 1.867 | 42.833 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_015 |  | 552.74 | 1.953 | 32.338 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_016 | recommended: max-speed endpoint | 528.44 | 2.043 | 28.072 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-only (B=8, S=2048) | uniform | sparse_nvfp4 |  | 520.63 | 2.074 | 23.976 | — | — | — | — | — | — | — | evaluated on ARC-Challenge (1172) | measured 5-repeat closure |
| prefill-decode (B=16, S=2048, O=80) | uniform | dense_bf16 | baseline | 5150.03 | 1.000 | — | 23.671 | 87.185 | 21.688 | 87.176 | 46.701 | 19.296 | — | evaluated on all three tasks | measured task-run speed (E2E derived) |
| prefill-decode (B=16, S=2048, O=80) | uniform | dense_nvfp4 |  | 4336.26 | 1.188 | — | 24.273 | 87.204 | 20.592 | 86.691 | 44.969 | 16.835 | — | evaluated on all three tasks | measured task-run speed (E2E derived) |
| prefill-decode (B=16, S=2048, O=80) | uniform | marlin_nvfp4 |  | 3713.63 | 1.387 | — | 24.579 | 87.251 | 21.368 | 87.053 | 46.772 | 18.200 | — | evaluated on all three tasks | measured task-run speed (E2E derived) |
| prefill-decode (B=16, S=2048, O=80) | uniform | sparse_bf16 |  | 3571.87 | 1.442 | — | 15.350 | 82.131 | 13.539 | 75.392 | 15.798 | 3.900 | — | evaluated on all three tasks | measured task-run speed (E2E derived) |
| prefill-decode (B=16, S=2048, O=80) | uniform | sparse_nvfp4 |  | 4345.41 | 1.185 | — | 2.047 | 9.626 | 9.388 | 61.906 | 1.133 | 0.238 | — | evaluated on all three tasks | measured task-run speed (E2E derived) |
| prefill-decode (B=16, S=2048, O=80) | ours | point_000 | identity / dense reference | 5150.03 | 1.000 | — | 23.610 | 87.185 | 21.598 | 87.159 | 46.909 | 19.412 | 0.000 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_001 |  | 5087.92 | 1.012 | — | 23.771 | 87.205 | 21.515 | 87.147 | 47.014 | 19.437 | 0.129 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_002 |  | 5098.02 | 1.010 | — | 23.623 | 87.177 | 21.760 | 87.174 | 47.087 | 19.565 | 0.073 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_003 | recommended: high-quality | 4989.93 | 1.032 | — | 23.776 | 87.197 | 21.385 | 87.146 | 47.773 | 20.238 | 0.067 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_004 |  | 5064.93 | 1.017 | — | 23.881 | 87.217 | 21.525 | 87.202 | 47.242 | 19.609 | 0.067 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_005 |  | 4868.26 | 1.058 | — | 23.820 | 87.190 | 21.713 | 87.195 | 46.493 | 19.126 | 0.184 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_006 |  | 4853.20 | 1.061 | — | 23.834 | 87.206 | 21.412 | 87.143 | 44.323 | 18.013 | 0.381 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_007 | recommended: quality/throughput | 4419.41 | 1.165 | — | 23.648 | 87.167 | 21.863 | 87.198 | 46.198 | 19.256 | 0.639 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_008 |  | 4368.53 | 1.179 | — | 23.819 | 87.187 | 21.643 | 87.103 | 45.432 | 18.920 | 0.715 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_009 |  | 4806.92 | 1.071 | — | 23.770 | 87.201 | 21.419 | 87.154 | 44.423 | 18.419 | 1.092 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_010 |  | 3798.32 | 1.356 | — | 23.424 | 87.054 | 21.300 | 87.133 | 43.473 | 16.816 | 1.819 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_011 | recommended: max-speed endpoint | 3004.98 | 1.714 | — | 23.544 | 87.081 | 21.581 | 87.153 | 45.309 | 18.301 | 2.115 | evaluated on all three tasks | measured formal closure |
| prefill-decode (B=16, S=2048, O=80) | ours-intermediate | i34 |  | 3549.06 | 1.451 | — | 23.752 | 87.088 | 21.703 | 87.171 | 45.410 | 18.304 | 2.018 | evaluated on all three tasks | screened-stall measurement* |
| prefill-decode (B=16, S=2048, O=80) | ours-intermediate | i36 |  | 3165.89 | 1.627 | — | 23.833 | 87.103 | 21.625 | 87.153 | 46.374 | 18.949 | 2.050 | evaluated on all three tasks | screened-stall measurement* |
| prefill-decode (B=16, S=2048, O=80) | ours-intermediate | i37 |  | 3103.38 | 1.659 | — | 23.843 | 87.151 | 21.685 | 87.172 | 46.025 | 18.639 | 2.084 | evaluated on all three tasks | screened-stall measurement* |
| prefill-decode (B=16, S=2048, O=80) | ours-intermediate | i38 | recommended: fast task-validated | 3065.00 | 1.680 | — | 23.754 | 87.117 | 21.633 | 87.140 | 46.298 | 19.029 | 2.109 | evaluated on all three tasks | screened-stall measurement* |

## Notes

- Prefill-only evaluates ARC-Challenge normalized accuracy on 1172 examples.
- Prefill-decode retains both measured metrics per dataset: ROUGE-L/BERTScore for CNN/DM and DialogSum, ROUGE-L/SacreBLEU for IWSLT.
- The screened intermediate rows were rescored from their original generation JSONL to complete all six task metrics; their speed remains the original stall-screened measurement.
- `point_011` and `ours_point_016` are the tested max-speed endpoints for their respective scenarios.
- `screened-stall measurement*` points are included for coverage but have stall-screened timing samples; do not use them for fine-grained timing claims against formal-closure points.
- Suggested candidates: prefill-only `ours_point_008` (high quality), `ours_point_012` (balanced), `ours_point_013` (dense-NVFP4 coverage), `ours_point_016` (endpoint); prefill-decode `point_003` (high quality), `point_007` (quality/throughput), `i38` (fast), `point_011` (endpoint).
