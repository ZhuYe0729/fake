# Llama-3.1-8B-Instruct prefill-decode downstream Pareto

All horizontal coordinates are freshly measured continuous phase-heterogeneous E2E speedups from `closure/summary.csv`. Uniform task quality is read-only from the frozen baseline artifacts; point_009 uses the pre-existing max-speed task run.

## cnn_dm_1000 (ROUGE-L)

| kind | policy | speedup vs dense | score |
|---|---|---:|---:|
| uniform | dense_bf16 | 0.999 | 19.487 |
| uniform | dense_nvfp4 | 1.058 | 16.345 |
| ours | point_002 | 1.096 | 20.274 |
| uniform | sparse_bf16 | 1.173 | 12.834 |
| ours | point_004 | 1.267 | 18.747 |
| ours | point_009_max_speed | 1.692 | 16.840 |

## dsum (ROUGE-L)

| kind | policy | speedup vs dense | score |
|---|---|---:|---:|
| uniform | dense_bf16 | 0.999 | 13.784 |
| uniform | dense_nvfp4 | 1.058 | 8.203 |
| ours | point_002 | 1.096 | 13.473 |
| uniform | sparse_bf16 | 1.173 | 4.049 |
| ours | point_004 | 1.267 | 13.266 |
| ours | point_009_max_speed | 1.692 | 9.085 |

## IWSLT (SacreBLEU)

| kind | policy | speedup vs dense | score |
|---|---|---:|---:|
| uniform | dense_bf16 | 0.999 | 10.680 |
| uniform | dense_nvfp4 | 1.058 | 10.312 |
| ours | point_002 | 1.096 | 10.586 |
| uniform | sparse_bf16 | 1.173 | 3.276 |
| ours | point_004 | 1.267 | 10.654 |
| ours | point_009_max_speed | 1.692 | 10.570 |

