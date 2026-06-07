# Predictor Hybrid Normal Scenario Summary

## Scenario

**batch_size=1, input_tokens=16384, output_tokens=32**

- Prefill M = 16384
- Decode M = 1, repeated for 32 output tokens
- Predictor hybrid chooses one compatible strategy per linear layer using offline predictor latency
- W4A4/W4A16 compatible pairs include the predicted conversion cost

## Full Model E2E

| Model | Predictor full E2E ms | Manual full E2E ms | Delta vs manual | Predictor policy |
|---|---:|---:|---:|---|
| Llama-2-7B | 2254.38 | 1930.00 | +16.81% | dense_nvfp4/marlin_nvfp4(224) |
| Llama-3.1-8B | 2160.83 | 2002.00 | +7.93% | dense_nvfp4/marlin_nvfp4(160), bf16(64) |
| Qwen3.5-9B | 4204.76 | 3308.00 | +27.11% | dense_nvfp4/marlin_nvfp4(184), bf16(64) |

## GPU Module Check

This table times only the predictor-selected linear kernel modules, then sums by layer count. It is useful for validating the policy's local kernel choices, but it is not a full-model E2E measurement.

| Model | Predictor module prefill ms | Predictor module decode x n ms | Predictor module E2E ms | Manual full E2E ms |
|---|---:|---:|---:|---:|
| Llama-2-7B | 597.93 | 266.74 | 864.67 | 1930.00 |
| Llama-3.1-8B | 592.14 | 353.58 | 945.72 | 2002.00 |
| Qwen3.5-9B | 584.49 | 321.70 | 906.19 | 3308.00 |

## Strategy Summary

| Model | Manual strategy | Predictor strategy | Layers changed |
|---|---|---|---:|
| Llama-2-7B | dense_nvfp4/marlin_nvfp4(224) | dense_nvfp4/marlin_nvfp4(224) | 0 |
| Llama-3.1-8B | dense_nvfp4/marlin_nvfp4(224) | dense_nvfp4/marlin_nvfp4(160), dense_bf16(64) | 64 |
| Qwen3.5-9B | dense_bf16(64), sparse_bf16(184) | dense_bf16(64), dense_nvfp4/marlin_nvfp4(184) | 184 |

## Analysis

- Llama-2 policy matches manual exactly, so the 16.81% slower full E2E result is not a strategy-selection difference. It points to benchmark/runtime differences such as full replacement path overhead, attention implementation differences, or first-use conversion/materialization cost inside the timed path.
- Llama-3.1 changes 64 `k_proj`/`v_proj` layers to `dense_bf16`. That reduces predicted decode cost, but the full model still trails manual by 7.93%; this needs a matched benchmark configuration before treating the predictor policy as better or worse.
- Qwen3.5 has the largest strategy divergence: predictor chooses W4A4/W4A16 compatible NVFP4 for 184 layers, while manual E2E selected `sparse_bf16` for those layers. The full E2E run replaced all 248 linear layers successfully, but predictor hybrid is 27.11% slower than manual. The gap is mainly in decode: the first decode step paid a large lazy materialization/conversion cost, and the steady decode path is still not enough to offset it.

## Files

| File | Description |
|---|---|
| `predictor_hybrid_full_e2e.csv` | Combined full model benchmark for Llama and Qwen3.5 |
| `llama_predictor_hybrid_full_e2e.csv` | Full Llama model benchmark |
| `qwen3_5_predictor_hybrid_full_e2e.csv` | Full Qwen3.5 predictor-hybrid benchmark |
| `gpu_policy_module_summary.csv` | Scenario-level GPU module totals |
| `strategy_diff_summary.csv` | Manual vs predictor strategy counts |
| `*_normal_01_policy.json` | Offline predictor policies |
| `predictor_vs_manual_summary.csv` | Predictor-estimated linear latency vs manual reference |
