# Predictor Hybrid Prefill-Only Summary

## Scenario

**batch_size=16, input_tokens=1024, output_tokens=0**

- Prefill M = batch_size x input_tokens = 16384
- Predictor hybrid = use `KernelLatencyPredictor` offline to choose one compatible strategy per linear layer
- GPU module check = instantiate the predictor-selected kernel modules on RTX 5090 and sum over layer counts
- Manual reference = existing offline-measured manual hybrid results in `manual/prefill_only`

## GPU Module Result

| Model | Predictor module ms | Manual hybrid ms | Delta vs manual | Strategy match |
|---|---:|---:|---:|---|
| Llama-2-7B | 418.65 | 413.90 | +1.15% | Yes |
| Llama-3.1-8B | 409.12 | 405.37 | +0.93% | Yes |
| Qwen3.5-9B | 426.18 | 427.24 | -0.25% | Mostly |

## Strategy Summary

| Model | Manual strategy | Predictor strategy | Layers changed |
|---|---|---|---:|
| Llama-2-7B | sparse_bf16(160), sparse_nvfp4(64) | sparse_bf16(160), sparse_nvfp4(64) | 0 |
| Llama-3.1-8B | sparse_bf16(160), sparse_nvfp4(64) | sparse_bf16(160), sparse_nvfp4(64) | 0 |
| Qwen3.5-9B | sparse_bf16(136), sparse_nvfp4(96), marlin_nvfp4(16) | sparse_bf16(152), sparse_nvfp4(96) | 16 |

## Notes

- Llama prefill-only is the cleanest case: predictor and manual choose the same layer policy, and the real GPU module timing is within about 1% of the manual reference.
- Qwen3.5 differs only on the 16 `k_proj`/`v_proj` layers where manual selected `marlin_nvfp4`; predictor selected `sparse_bf16`. The measured module total is slightly faster than manual, so this difference is acceptable for this isolated prefill-only workload.
- The Llama full-model prefill rows in `llama_predictor_hybrid_full_e2e.csv` include attention, norm, cache setup, and Hugging Face model overhead. They are not directly comparable to the manual prefill-only table, which is module-level linear timing.

## Files

| File | Description |
|---|---|
| `gpu_policy_module_summary.csv` | Scenario-level GPU module totals |
| `strategy_diff_summary.csv` | Manual vs predictor strategy counts |
| `predictor_vs_manual_summary.csv` | Predictor-estimated linear latency vs manual reference |
| `*_prefill_only_policy.json` | Offline predictor policies |
| `llama_predictor_hybrid_full_e2e.csv` | Full Llama forward check for reference |
