# CUDA-event overlap profile

| Mechanism metric | Median |
|---|---:|
| Total decode-state transfer work | 354.94 ms |
| Exposed copy stall | 1.92 ms |
| Hidden transfer work | 353.02 ms |
| Hidden ratio | 99.38% |

These are medians of 3 independent CUDA-event diagnostic runs on the
same Llama3.1 `point_011` B=8/S=2048/O=64 workload. `Total transfer work` sums
the per-layer decode-state materialization intervals on the copy stream.
`Exposed copy stall` sums the first-use copy-event stalls observed on the
compute stream. Profiling is diagnostic-only and is not used for formal E2E
latency.
