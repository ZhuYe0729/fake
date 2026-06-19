# DINOv3 Layerwise Max Speed

This directory contains speed-model-selected per-layer CUTLASS policies and real DINOv3 forward benchmarks.

- Batch sizes: `32`
- Candidate kernels: `dense_bf16 dense_nvfp4 sparse_bf16 sparse_nvfp4`
- Input size: `3 256 256`
- Warmup/iters: `5/20`

Key files after a full run: `speed.csv`, `summary.csv`, `policy_summary.csv`, and per-batch policy/candidate files.
