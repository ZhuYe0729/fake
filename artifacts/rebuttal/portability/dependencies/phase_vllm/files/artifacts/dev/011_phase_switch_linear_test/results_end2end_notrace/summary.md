# No-Trace End-to-End Benchmark

scenario: `end2end_notrace_i2048_o80_bs16`

| Method | Prefill | Decoding | Total | Total speedup vs BF16 |
|---|---:|---:|---:|---:|
| dense BF16 | 2.1733s | 2.7893s | 4.9626s | 1.00x |
| dense NVFP4 | 1.3634s | 2.9571s | 4.3205s | 1.15x |
| W4A16 Marlin | 2.2848s | 1.4486s | 3.7334s | 1.33x |
| dense NVFP4 + W4A16 | 1.3916s | 1.4777s | 2.8693s | 1.73x |
