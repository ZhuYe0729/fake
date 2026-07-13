# Llama2-7B Broad Grid vLLM Benchmark

Latency cells are median wall-clock latency in milliseconds. Failed cells keep their status label.
Speedup cells are relative to `dense_bf16` for the same `(batch,input_seq,output_seq)`.

## Grid

- Batch values: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
- Input seq values: 128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536
- Output seq values: 1, 16, 64, 128
- Config rows: 468
- Raw method-config rows: 2808

## Method Status Counts

| method | OK | PRECHECK_OOM | INIT_OOM | OOM | OOM_SKIPPED | ERROR/MISSING |
|---|---:|---:|---:|---:|---:|---:|
| dense_bf16 | 206 | 236 | 0 | 0 | 0 | 26 |
| dense_nvfp4 | 216 | 236 | 0 | 0 | 0 | 16 |
| sparse_bf16 | 216 | 236 | 0 | 0 | 0 | 16 |
| sparse_nvfp4 | 216 | 236 | 0 | 0 | 0 | 16 |
| marlin_nvfp4 | 216 | 236 | 0 | 0 | 0 | 16 |
| hetero | 216 | 236 | 0 | 0 | 0 | 16 |

## Average Speedup Over Comparable OK Cells

| method | comparable_configs | avg_speedup |
|---|---:|---:|
| dense_bf16 | 0 | pending |
| dense_nvfp4 | 204 | 1.263 |
| sparse_bf16 | 204 | 1.392 |
| sparse_nvfp4 | 204 | 1.299 |
| marlin_nvfp4 | 204 | 1.322 |
| hetero | 204 | 1.337 |

## Outputs

- `results/summary_long.csv`: long-form method-config summary.
- `results/iterations.csv`: raw timed iterations.
- `summary/broad_grid_latency_table.csv`: requested wide latency table.
- `summary/broad_grid_speedup_table.csv`: wide speedup table.
- `summary/promising_scenarios_modeling.md`: selected high-potential scenarios plus kernel-model best-mixed analysis.
