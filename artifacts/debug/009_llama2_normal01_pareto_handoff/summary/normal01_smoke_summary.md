# Llama2 Normal-01 Pareto Smoke Summary

## Scenario

- Model: llama2-7b
- Scenario: normal_01 (batch_size=1, input_tokens=16384, output_tokens=32)
- Methods: dense_bf16, dense_nvfp4, sparse_bf16, sparse_nvfp4, marlin_nvfp4, dense_nvfp4_prefill_marlin_decode
- Quality formula: local_rel_mse_log_numel_layer_family

## Inputs

- Candidate rows: 1344 (224 modules × 6 methods)
- Latency source: policy JSON (artifacts/results/benchmarks/hybrid/pred/normal_01/llama_2_7b_normal_01_policy.json)
- Quality source: 007_llama2_quality_modeling/sensitivity/module_method_errors.csv

## 1. Script Generalization

The core pipeline (common_pareto.py, build_cost_table.py, optimize_pareto.py, summarize_pareto.py, validate_pareto_e2e.py) was successfully adapted from prefill_only to normal_01. Key changes:

- Added decode_ms and conversion_ms to per-module latency computation
- Added 6th method: dense_nvfp4_prefill_marlin_decode
- Updated policy format to include decode backend and conversion fields
- E2E benchmark extended from prefill-only to prefill+32-step decode
- All scripts compile and run without errors

## 2. Pareto Frontier

8 unique points generated from 9 budget levels. Frontier progression:

| Point | Quality | Latency (pred) | Speedup | Key Methods |
|-------|---------|---------------|---------|-------------|
| 0 | 0.000 | 1383.7 | 1.00x | dense_bf16 ×224 |
| 1 | 1.608 | 1284.6 | 1.08x | dense_bf16 ×203, hybrid ×21 |
| 2 | 3.152 | 1185.6 | 1.17x | dense_bf16 ×182, hybrid ×42 |
| 3 | 7.351 | 1019.5 | 1.36x | dense_bf16 ×139, hybrid ×81, sparse_bf16 ×4 |
| 4 | 17.937 | 934.0 | 1.48x | dense_bf16 ×82, hybrid ×92, sparse_bf16 ×50 |
| 5 | 45.028 | 883.9 | 1.57x | dense_bf16 ×35, hybrid ×97, sparse_bf16 ×92 |
| 6 | 106.083 | 831.8 | 1.66x | dense_bf16 ×1, hybrid ×84, sparse_bf16 ×139 |
| 7 | 136.545 | 814.7 | 1.70x | hybrid ×64, sparse_bf16 ×160 |

The frontier is sensible: hybrid (dense_nvfp4_prefill_marlin_decode) is the first upgrade path, replacing dense_bf16. As budget increases, sparse_bf16 takes over. No pure dense_nvfp4 or marlin_nvfp4 is selected — hybrid dominates both. No sparse_nvfp4 is selected — sparse_bf16 has better total latency for this scenario.

## 3. E2E Validation Results

3 points validated on GPU7 (RTX 5090, 32GB):

| Point | Pred Total | E2E Total | E2E Prefill | E2E Decode Avg | E2E Speedup | Pred/E2E Ratio | replaced | skipped |
|-------|-----------|-----------|-------------|----------------|-------------|----------------|----------|---------|
| 0 | 1383.7 | 2437.9 | 1504.3 | 29.18 | 1.00x | 0.568 | 224 | 0 |
| 3 | 1019.5 | 2577.7 | 1246.7 | 41.59 | 0.95x | 0.395 | 224 | 0 |
| 7 | 814.7 | 2331.6 | 1055.6 | 39.87 | 1.05x | 0.349 | 224 | 0 |

**E2E ranking**: Point 7 (2332ms) < Point 0 (2438ms) < Point 3 (2578ms)

**Key observations:**
- Point 7 (fastest predicted) IS fastest in E2E (2332ms vs 2438ms dense), confirming the optimizer direction
- Point 3 is SLOWER than dense in E2E (2578ms vs 2438ms) despite being predicted 1.36x faster — the decode slowdown from NVFP4 kernels overwhelms prefill gains
- Predicted/E2E ratio drops from 0.57 (all-dense) to 0.35 (sparse+hybrid) — the linear-only model captures less of the total E2E time as more aggressive methods are used
- replaced_linear_count = 224, skipped_linear_count = 0 for all validated points
- All 3 points needed PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid OOM

**Decode slowdown analysis:**
- Point 0 decode: 29.18ms/step (pure dense_bf16)
- Point 3 decode: 41.59ms/step (+43%) — hybrid NVFP4 modules add marlin-to-cutlass conversion overhead
- Point 7 decode: 39.87ms/step (+37%) — sparse_bf16 prefill with dense_bf16 decode is faster than hybrid's NVFP4 decode path

The linear-only latency model under-predicts decode cost because it excludes: attention operations, KV cache management, residual connections, layer norms, and kernel dispatch overhead for mixed-backend configurations.

## 4. Hybrid Method (dense_nvfp4_prefill_marlin_decode)

Yes, the hybrid appeared and was selected on the Pareto frontier:
- Available: Yes (injected as 6th method, reuses dense_nvfp4 quality)
- Selected: Yes — it's the primary upgrade from dense_bf16 at points 1-6
- At point 7, hybrid count drops to 64 (from 97 at point 5) as sparse_bf16 takes over
- Hybrid uses dense_nvfp4 prefill + marlin_nvfp4 decode + canonical_to_marlin conversion (~0.45-0.56ms per module)
- In E2E, hybrid modules add conversion overhead that the linear model partially captures

**Conversion latency assumption**: Conversion cost is modeled as 0.0 for sparse methods (dense_bf16 decode uses pre-stored weights). For hybrid and marlin_nvfp4, conversion is extracted from the policy JSON. This matches the framework's own modeling. If conversion cost is not available, it would be 0.0 and clearly marked — but in this case it was available.

## 5. Review Before Larger Validation

Items to review before scaling to 11-point validation:

1. **Decode latency model is too optimistic**: Linear-only predicted decode is ~40% of real E2E decode. A decode overhead factor or inclusion of attention operations is needed.
2. **Point 3 regression**: The middle point is slower than dense in E2E — the optimizer needs a decode-aware cost model to avoid selecting methods whose decode penalty outweighs prefill gains.
3. **Memory management**: E2E validation requires `expandable_segments:True` to avoid OOM on 32GB GPUs for normal_01.
4. **Sparse decode pairing**: Sparse methods pair with dense_bf16 decode (not sparse decode, which is unsupported at M=1). This pairing works in E2E but adds a "hidden" decode cost that the linear model overlooks.
5. **Quality validation**: NLL and ARC evaluation were not run (as specified — wait for smoke review). These will add important data points for the quality-speed tradeoff.

## 6. Failed Commands and Issues

- First E2E run: points 0 and 7 hit CUDA OOM during benchmark. Fixed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- GPU7 used for all validation (high-numbered, free per instructions).

## Review Checklist

1. Path: `fake/artifacts/debug/009_llama2_normal01_pareto_handoff/`
2. Candidate rows: 1344
3. Unique Pareto points: 8
4. E2E validated points: 0, 3, 7
5. replaced_linear_count=224, skipped_linear_count=0 for all 3 points ✓
6. Predicted vs E2E:
   - Point 0: pred=1383.7, e2e=2437.9
   - Point 3: pred=1019.5, e2e=2577.7
   - Point 7: pred=814.7, e2e=2331.6
7. E2E ranking: 7 < 0 < 3. Predicted ranking: 7 < 3 < 0. Point 7 is fastest in both. Point 3 is out of order.
8. dense_nvfp4_prefill_marlin_decode: Available ✓, Selected ✓ (64-97 modules on frontier)
9. Conversion latency: Extracted from policy JSON. canonical_to_cutlass ~0.035ms, canonical_to_marlin ~0.45-0.56ms per module.
10. Failed commands: OOM on first run without expandable_segments. Resolved.
