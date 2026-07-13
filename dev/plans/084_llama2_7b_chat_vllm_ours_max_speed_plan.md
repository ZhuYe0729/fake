# 084 Llama2-7B-Chat vLLM Ours Max-Speed Plan

## Summary
- For Llama-2-7B-Chat, generate one predictor-only layer-heterogeneous max-speed policy per vLLM workload, export it to the existing phase-heterogeneous vLLM checkpoint format, then measure E2E speed and PMPD quality.
- Cover `prefill_only` (`b=8, input=2048, output=1`) and `prefill_decode` (`b=16, input=2048, output=80`).
- Keep Pareto optimization as an explicit TODO; do not introduce a quality model in this plan.

## Decisions
- Kernel routing uses `KernelLatencyPredictor`, not per-layer microbench lookup tables.
- Candidate methods are `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`, and predictor `marlin_nvfp4`, mapped to vLLM `w4a16_ours`.
- Predictor operates on vLLM's fused Llama projections (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`), independently for every decoder layer.
- Pure prefill optimization uses zero decode steps, while vLLM benchmarking retains the baseline request with one generated token.
- Quality evaluation uses `cnn_dm_1000`, `dsum`, and `IWSLT` for each exported scenario checkpoint.

## Verification
- Static-compile new Python scripts and shell-syntax-check launchers.
- Validate policy module count and supported runtime methods before export.
- Smoke-generate one request through each exported checkpoint before full speed/quality runs.
- Summaries compare ours with the existing dense-BF16 and uniform baselines.
