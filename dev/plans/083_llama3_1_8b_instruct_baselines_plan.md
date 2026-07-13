# Llama-3.1-8B-Instruct Baseline Plan

## Summary
- Create an independent vLLM baseline suite for `/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- Preserve the Llama2 baseline methods, calibration configuration, two serving scenarios, and PMPD Claude-style generation prompt.

## Key Changes
- Add calibrated preparation and fused vLLM export for `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`, and `marlin_nvfp4`, alongside the dense BF16 reference.
- Measure `prefill_only` (`b=8,in=2048,out=1`) and `prefill_decode` (`b=16,in=2048,out=80`) on one GPU per vLLM process.
- Evaluate `cnn_dm_1000`, `dsum`, and `IWSLT` with the same PMPD Claude-style prompt and generation settings as Llama2.
- Document that the prompt is selected for Llama2/Llama3 comparability and is not the Llama-3 native chat-template deployment protocol.

## Test Plan
- Compile all scripts, export after real calibrated preparation, and smoke-load every checkpoint in vLLM.
- Run full speed and quality jobs, verify all 10 speed rows and 15 quality rows, then generate CSV/Markdown summaries.

## Assumptions
- The Llama-3.1 checkpoint remains a standard Llama causal LM compatible with the existing fused QKV/gate-up export path.
- IWSLT length filtering uses the same local Llama-2-chat tokenizer fallback as the Llama2 baseline because the PMPD Vicuna tokenizer is unavailable.
