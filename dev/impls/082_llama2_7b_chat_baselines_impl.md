# Llama2-7B-Chat Baseline Implementation Log

## 2026-07-09 - Baseline Script Scaffold
- Development purpose: add reproducible scripts for llama2-7b-chat dense and global uniform compressed vLLM baselines.
- Changes: created baseline directory, compression/export/benchmark/eval/summarize scripts, and documentation.
- Affected files: `artifacts/exports/vllm/baselines/llama2-7b-chat/`.
- Follow-up: run real compression/export and then the full speed and PMPD quality jobs on an available GPU.

## 2026-07-10 - Complete Baseline Measurements
- Development purpose: complete calibrated global-uniform compression baselines for the two target vLLM scenarios and three generation-quality datasets.
- Changes: ran dense BF16, dense NVFP4, sparse BF16, sparse NVFP4, and Marlin NVFP4 throughput/latency measurements; completed CNN/DM 1000-example subset, DialogSum, and IWSLT generation evaluations; generated CSV/Markdown summaries.
- Affected files: `artifacts/exports/vllm/baselines/llama2-7b-chat/results/`, `artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/summarize_results.py`.
- Follow-up: IWSLT used the Llama-2-chat tokenizer as a local fallback for PMPD length filtering because the required Vicuna tokenizer was unavailable, so label it as non-strict PMPD. CNN/DM is the fixed 1000-example subset.
