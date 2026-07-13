# Llama2-7B-Chat vLLM Baseline Evaluation Plan

## Summary
- Create `artifacts/exports/vllm/baselines/llama2-7b-chat/` for dense and global-uniform baseline scripts.
- Evaluate two scenarios: prefill-only `b=8,in=2048,out=1` and prefill-decode `b=16,in=2048,out=80`.
- Use the chat checkpoint at `/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf`.
- Generate compressed baselines through real calibration/preparation before vLLM export.

## Key Changes
- Add scripts for calibrated compression preparation, fused vLLM export, speed benchmarking, PMPD-style vLLM generation evaluation, and result summarization.
- Baseline methods: `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`, `marlin_nvfp4`.
- PMPD quality uses `cnn_dm_1000`, `dsum`, and `IWSLT`; the CNN/DM result is explicitly a fixed 1000-example subset.

## Test Plan
- Run `python -m py_compile` on all new Python scripts.
- Run export dry-run after prepared artifacts exist.
- Smoke speed with a small batch/input before running the two full scenarios.
- Smoke PMPD with `cnn_dm_1000 --question-end 2` before full quality runs.

## Assumptions
- vLLM execution uses the local vLLM environment.
- Compression preparation uses the project environment with CUDA/CUTLASS kernels available.
- If `marlin_nvfp4` export/load fails, record the failure and keep the other baselines intact.
