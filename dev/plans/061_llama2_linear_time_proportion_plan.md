# 061 Llama2 Linear Time Proportion Plan

## Summary
- Add a Llama2-7B focused linear time proportion study under `artifacts/debug/022_linear_time_proportion_study/llama2_7b/`.
- Keep the existing Qwen3.5 022 artifacts unchanged.
- Run dense BF16 full-model speed and hook-based coarse breakdown on local GPUs `7,6,5,4`.
- Use conda env `cospaq`; do not use SLURM/supercomputer launch assumptions.

## Key Changes
- Add a Llama2-specific benchmark script that loads `AutoModelForCausalLM` with `local_files_only=True`, BF16, and `sdpa` attention.
- Test matrix:
  - batch size: `1,4,16,32,64`
  - input tokens: `16,64,256,1024,4096,8192`
  - speed output tokens: `1,32,128,256`
  - breakdown output tokens: `1,32`
- Add a local multi-GPU launcher that shards configs across GPU `7,6,5,4`, with one worker process per GPU.
- Add analysis outputs:
  - `summary/llama2_linear_proportion_summary.csv`
  - `summary/analysis_report.md`
  - `summary/qwen_vs_llama_context.md`

## Test Plan
- Static validation:
  - `python -m py_compile artifacts/debug/022_linear_time_proportion_study/llama2_7b/run_study.py`
  - `python -m py_compile artifacts/debug/022_linear_time_proportion_study/llama2_7b/analyze.py`
  - `bash -n artifacts/debug/022_linear_time_proportion_study/llama2_7b/run_parallel.sh`
- Runtime validation:
  - `bash artifacts/debug/022_linear_time_proportion_study/llama2_7b/run_parallel.sh`
  - Confirm each GPU has at most one active worker.
  - Confirm non-OOM configs produce speed rows and breakdown rows.
  - Confirm summary reports are generated.

## Assumptions
- If the sandbox cannot see GPUs, the scripts are still valid for the target local GPU machine.
- OOM rows are expected for large batch/long input combinations and are retained in CSVs.
- The study only covers dense BF16 Llama2-7B.
