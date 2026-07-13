# Llama2-7B Broad Grid vLLM Benchmark

Latency cells are median wall-clock latency in milliseconds. Failed cells keep their status label.
Speedup cells are relative to `dense_bf16` for the same `(batch,input_seq,output_seq)`.

## Grid

- Batch values: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
- Input seq values: 128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536
- Output seq values: 1, 16, 64, 128
- Config rows: 468
- Raw method-config rows: 0

## Method Status Counts

| method | OK | PRECHECK_OOM | INIT_OOM | OOM | OOM_SKIPPED | ERROR/MISSING |
|---|---:|---:|---:|---:|---:|---:|
| dense_bf16 | 0 | 0 | 0 | 0 | 0 | 0 |
| dense_nvfp4 | 0 | 0 | 0 | 0 | 0 | 0 |
| sparse_bf16 | 0 | 0 | 0 | 0 | 0 | 0 |
| sparse_nvfp4 | 0 | 0 | 0 | 0 | 0 | 0 |
| marlin_nvfp4 | 0 | 0 | 0 | 0 | 0 | 0 |
| hetero | 0 | 0 | 0 | 0 | 0 | 0 |

## Average Speedup Over Comparable OK Cells

| method | comparable_configs | avg_speedup |
|---|---:|---:|
| dense_bf16 | 0 | pending |
| dense_nvfp4 | 0 | pending |
| sparse_bf16 | 0 | pending |
| sparse_nvfp4 | 0 | pending |
| marlin_nvfp4 | 0 | pending |
| hetero | 0 | pending |

## Outputs

- `results/summary_long.csv`: long-form method-config summary.
- `results/iterations.csv`: raw timed iterations.
- `summary/broad_grid_latency_table.csv`: requested wide latency table.
- `summary/broad_grid_speedup_table.csv`: wide speedup table.
