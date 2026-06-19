# 057 Llama2 4090 Prefill Speed Plan

## Summary
- Create `artifacts/debug/025_llama2_4090_prefill_speed/` for a portable 4090 prefill-only speed test.
- Target model: Llama-2-7B at `/data/home/scxj523/run/wja/data/models/LLM-Research/llama-2-7b`.
- Target runtime: SLURM `gpu_4090`, CUDA 12.8, conda env `wja-cospaq`, project path `/data/home/scxj523/run/wja/project/my/fake`.
- Compare `dense_bf16`, `sparse_bf16`, and `marlin_nvfp4` as the repo backend for Marlin W4A16.
- Use prefill-only `batch_size=16`, `input_tokens=1024`, `output_tokens=0`, so `M=16384`.

## Key Changes
- Add a self-contained benchmark script under the debug directory.
- Measure full-model prefill latency as the primary metric.
- Also emit linear aggregate latency for the same methods as supporting evidence.
- Generate uniform method policies for compressed methods using the existing offline policy schema and Llama replacement path.
- Add a `gpu_4090` SLURM launcher that activates the offline supercomputer environment and writes all outputs into the debug directory.

## Test Plan
- Static validation:
  - `python -m py_compile artifacts/debug/025_llama2_4090_prefill_speed/scripts/bench_llama2_4090_prefill_speed.py`
  - `bash -n artifacts/debug/025_llama2_4090_prefill_speed/run_llama2_4090_prefill_speed_4090.sh`
  - `python artifacts/debug/025_llama2_4090_prefill_speed/scripts/bench_llama2_4090_prefill_speed.py --help`
- Supercomputer run:
  - Submit `sbatch artifacts/debug/025_llama2_4090_prefill_speed/run_llama2_4090_prefill_speed_4090.sh`.
  - Confirm full-model and linear CSV summaries contain all three methods.
  - Treat `speedup_vs_dense_bf16` in `results/full_model_prefill_summary.csv` as the main comparison.

## Assumptions
- `marlin w4a16` means the existing `marlin_nvfp4` backend.
- Unsupported compressed replacements are reported through skipped/fallback counts rather than hidden.
- The local machine does not need GPU execution; only syntax and shell checks are run locally.
