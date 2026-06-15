# 053 DINOv3 Layerwise Max Speed Plan

## Summary
Implement a DINOv3 ViT-7B/16 speed-model-driven layerwise policy that selects the fastest realizable backend per Linear layer for each batch size in `1 2 4 8 16 32 64 128`, then applies that policy to the real DINO model and benchmarks forward speed.

All new experiment code, scripts, policy JSON/CSV, benchmark CSV, and summaries write under `artifacts/debug/019_dinov3_layerwise_max_speed/`.

## Key Changes
- Add `artifacts/debug/019_dinov3_layerwise_max_speed/code/dinov3_layerwise_policy.py`, a DINO policy loader/replacer that supports actual per-module backends:
  - `dense_bf16`: leave the selected `nn.Linear` in BF16.
  - `dense_nvfp4`: replace with existing CUTLASS NVFP4 wrapper.
  - `sparse_bf16`: replace with existing CUTLASS sparse BF16 wrapper.
  - `sparse_nvfp4`: replace with existing CUTLASS sparse NVFP4 wrapper.
- Add `artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.py`, a runner script that:
  - enumerates DINOv3 compressible Linear modules;
  - computes the DINO transformer GEMM `m` from batch size and inferred image-token count;
  - calls `KernelLatencyPredictor` for each `(m, n, k)` and candidate backend;
  - selects the lowest predicted supported latency per module;
  - writes policy JSON/CSV, candidate tables, aggregate predicted summaries, real forward speed rows, and baseline comparisons.
- Add `artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.sh` using the existing CUDA 12.8 / `wja-cospaq` / offline HF pattern.

## Test Plan
- Static verification:
  - `python3 -m py_compile` for new Python files.
  - `bash -n` for the new Slurm script.
  - `PYTHONPATH=. python <runner> --help` without local DINO weights.
- GPU-node smoke test:
  - run batch `1` only with `WARMUP=1 ITERS=2`;
  - verify policy files are produced and selected modules total 280.
- Full supercomputer run:
  - run the default batch sweep;
  - confirm `speed.csv` has one row per batch;
  - confirm `summary.csv` reports predicted best, measured latency, images/sec, and speedup versus available uniform baselines.

## Assumptions
- "Maximum speed" means no accuracy constraint for this run.
- Candidate methods are limited to currently realizable DINO CUTLASS paths: `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`.
- Benchmark scope stays consistent with prior DINO speed tests: random-input classifier forward only.
- The final authoritative number is measured full-model forward speed; predicted latency is used to choose the layerwise policy and explain the selection.
