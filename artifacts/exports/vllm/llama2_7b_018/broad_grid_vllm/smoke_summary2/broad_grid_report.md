# Llama2-7B Broad Grid vLLM Benchmark

Latency cells are median wall-clock latency in milliseconds. Failed cells keep their status label.
Speedup cells are relative to `dense_bf16` for the same `(batch,input_seq,output_seq)`.

## Grid

- Batch values: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
- Input seq values: 128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536
- Output seq values: 1, 16, 64, 128
- Config rows: 468
- Raw method-config rows: 12

## Method Status Counts

| method | OK | PRECHECK_OOM | INIT_OOM | OOM | OOM_SKIPPED | ERROR/MISSING |
|---|---:|---:|---:|---:|---:|---:|
| dense_bf16 | 0 | 0 | 0 | 0 | 0 | 2 |
| dense_nvfp4 | 2 | 0 | 0 | 0 | 0 | 0 |
| sparse_bf16 | 2 | 0 | 0 | 0 | 0 | 0 |
| sparse_nvfp4 | 2 | 0 | 0 | 0 | 0 | 0 |
| marlin_nvfp4 | 2 | 0 | 0 | 0 | 0 | 0 |
| hetero | 2 | 0 | 0 | 0 | 0 | 0 |

## Average Speedup Over Comparable OK Cells

| method | comparable_configs | avg_speedup |
|---|---:|---:|
| dense_bf16 | 0 | pending |
| dense_nvfp4 | 2 | 126.380 |
| sparse_bf16 | 2 | 194.551 |
| sparse_nvfp4 | 2 | 127.744 |
| marlin_nvfp4 | 2 | 94.215 |
| hetero | 2 | 106.444 |

## Outputs

- `results/summary_long.csv`: long-form method-config summary.
- `results/iterations.csv`: raw timed iterations.
- `summary/broad_grid_latency_table.csv`: requested wide latency table.
- `summary/broad_grid_speedup_table.csv`: wide speedup table.
