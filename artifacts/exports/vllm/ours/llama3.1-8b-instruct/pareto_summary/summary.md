# Llama-3.1-8B-Instruct measured result table

All listed rows have a measured speed. All six prefill-decode closure points have measured WikiText NLL; five selected points have full downstream task scores. `recommended` labels are suggestions, not filtering.

| scenario | family | policy | recommended use | E2E ms | speedup | ARC norm. | CNN R-L | CNN BERTScore | DSum R-L | DSum BERTScore | IWSLT R-L | IWSLT BLEU | ΔNLL | task status | speed source |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| prefill-only (B=8, S=2048) | uniform | dense_bf16 | baseline | 1187.12 | 1.000 | 55.375 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | uniform | dense_nvfp4 |  | 687.15 | 1.728 | 54.096 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | uniform | sparse_bf16 |  | 766.07 | 1.550 | 38.737 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | uniform | sparse_nvfp4 |  | 618.46 | 1.919 | 26.792 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | uniform | marlin_nvfp4 |  | 1156.28 | 1.027 | 54.096 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_3 |  | 969.65 | 1.224 | 54.778 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_5 |  | 826.11 | 1.437 | 54.096 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_6 | recommended: near-lossless | 760.76 | 1.560 | 54.863 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_8 | recommended: dense-NVFP4-cover | 663.98 | 1.788 | 52.645 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_9 |  | 639.62 | 1.856 | 49.403 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_11 | optional: high-speed trade-off | 583.41 | 2.035 | 44.539 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-only (B=8, S=2048) | ours | ours_point_13 |  | 547.46 | 2.168 | 33.106 | — | — | — | — | — | — | — | evaluated | fresh 5-repeat closure |
| prefill-decode (B=16, S=2048, O=80) | uniform | dense_bf16 | baseline | 3432.16 | 1.000 | — | 19.487 | 83.600 | 13.784 | 78.187 | 28.146 | 10.680 | 0.000 | evaluated | fresh continuous closure |
| prefill-decode (B=16, S=2048, O=80) | uniform | dense_nvfp4 |  | 3242.68 | 1.058 | — | 16.345 | 84.988 | 8.203 | 81.634 | 28.408 | 10.312 | 2.882 | evaluated | fresh continuous closure |
| prefill-decode (B=16, S=2048, O=80) | uniform | marlin_nvfp4 |  | 3134.70 | 1.158 | — | 16.108 | 85.061 | 8.805 | 81.439 | 28.805 | 10.403 | 2.882 | evaluated | frozen legacy runner* |
| prefill-decode (B=16, S=2048, O=80) | uniform | sparse_bf16 |  | 2926.34 | 1.173 | — | 12.834 | 81.278 | 4.049 | 79.128 | 14.322 | 3.276 | 55.306 | evaluated | fresh continuous closure |
| prefill-decode (B=16, S=2048, O=80) | uniform | sparse_nvfp4 |  | 4589.68 | 0.791 | — | 0.804 | 76.729 | 0.126 | 78.438 | 0.110 | 0.014 | 113.600 | evaluated | frozen legacy runner* |
| prefill-decode (B=16, S=2048, O=80) | ours | point_000 | identity / dense reference | 3432.16 | 1.000 | — | 19.487 | 83.600 | 13.784 | 78.187 | 28.146 | 10.680 | 0.000 | same policy as dense BF16 | fresh continuous closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_002 | recommended: high-quality | 3132.48 | 1.096 | — | 20.274 | 83.245 | 13.473 | 78.400 | 28.098 | 10.586 | 0.387 | evaluated on all three tasks | fresh continuous closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_004 | recommended: primary balanced | 2708.39 | 1.267 | — | 18.747 | 83.837 | 13.266 | 78.585 | 27.893 | 10.654 | 0.543 | evaluated on all three tasks | fresh continuous closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_006 | recommended: fast task-validated | 2312.23 | 1.484 | — | 16.675 | 84.399 | 11.122 | 80.727 | 28.405 | 10.582 | 1.638 | evaluated on all three tasks | fresh continuous closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_008 | recommended: high-speed task-validated | 2165.02 | 1.585 | — | 16.274 | 84.040 | 8.957 | 81.619 | 28.343 | 10.546 | 2.821 | evaluated on all three tasks | fresh continuous closure |
| prefill-decode (B=16, S=2048, O=80) | ours | point_009_max_speed | recommended: max-speed endpoint | 2028.53 | 1.692 | — | 16.840 | 84.047 | 9.085 | 81.463 | 28.846 | 10.570 | 2.882 | pre-existing max-speed task run | fresh continuous closure |

## Notes

- Prefill-only uses ARC-Challenge normalized accuracy over 1172 examples.
- Prefill-decode retains both measured metrics per dataset: ROUGE-L/BERTScore for CNN/DM and DialogSum, ROUGE-L/SacreBLEU for IWSLT.
- `fresh continuous closure` denotes the 6-warmup / 5-measurement phase-continuous protocol.  `frozen legacy runner*` is retained because no continuous remeasurement exists for those two uniform methods; they are visibly distinguished in the task figures and should not be used for a fine-grained speed claim.
- Recommended paper candidates: prefill-only `ours_point_6` (near-lossless) and `ours_point_8` (dense-NVFP4 coverage); prefill-decode `point_002` (high quality), `point_004` (balanced primary), `point_006`/`point_008` (task-validated fast points), and `point_009_max_speed` (endpoint).
