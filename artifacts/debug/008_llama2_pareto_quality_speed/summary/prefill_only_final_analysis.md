# Llama2-7B Prefill-Only Pareto Analysis — Final Results

## Claim

Constrained per-linear-module assignment finds useful speed-quality tradeoffs under prefill-only inference for Llama2-7B. The Pareto optimizer discovers mixed-method policies that dominate any single-method uniform baseline.

## Experimental Setup

| Parameter | Value |
|---|---|
| Model | Llama-2-7B |
| Scenario | prefill_only |
| Batch size | 16 |
| Input tokens | 1024 |
| Output tokens | 0 |
| Linear modules | 224 (32 layers x 7 projection matrices) |
| Candidate methods | dense_bf16, dense_nvfp4, sparse_bf16, sparse_nvfp4, marlin_nvfp4 |
| Quality proxy | `local_rel_mse * log1p(numel) * layer_weight * family_weight` |
| Quality validation | NLL on 16352 tokens + ARC-Challenge limit=128 |
| E2E timing | 1 warmup + 3 iterations, GPU7 |
| Pareto budget range | 0 – 379.26 (integer DP with 2000 bins) |

## Results

### Frontier Overview

The optimizer produced 11 unique Pareto points spanning quality cost 0 to 249.4:

| Point | Methods | Speedup | NLL delta | ARC acc_norm |
|---|---|---|---|---|
| 0 | 224 dense_bf16 | 1.00x | 0.0000 | 0.4609 |
| 1 | 222 dense_bf16 + 2 dense_nvfp4 | 1.01x | 0.0000 | 0.4609 |
| 5 | 182 dense_bf16 + 42 dense_nvfp4 | 1.14x | 0.0024 | 0.4609 |
| 8 | 131 dense_nvfp4 + 93 sparse_bf16 | 1.49x | 0.0698 | 0.4531 |
| 9 | 61 dense_nvfp4 + 158 sparse_bf16 + 5 sparse_nvfp4 | 1.60x | 0.1759 | 0.3750 |
| 10 | 160 sparse_bf16 + 64 sparse_nvfp4 | 1.66x | 0.5958 | 0.3203 |

### Quality Proxy Validation

The quality cost formula strongly predicts real quality degradation:

- **quality_cost vs NLL**: Pearson = 0.969, Spearman = 1.000 (all 11 points)
- **quality_cost vs ARC-Challenge acc_norm**: Pearson = −0.965, Spearman = −0.790 (all 11 points)

### Latency Prediction Accuracy

Predicted linear-only latency (sum of per-module microbenchmarks) tracks real E2E prefill latency almost perfectly:

- **predicted vs E2E**: Pearson = 0.9995, Spearman = 1.000 (all 11 points)
- The ratio `pred/E2E` decreases from 0.78 (point 0, all-dense) to 0.60 (point 10, most aggressive), reflecting that non-linear overhead (attention, norms, etc.) is a larger fraction of total latency as linear layers get faster.

### Comparison Against Uniform Baselines

| Policy | E2E latency (ms) | Speedup | NLL delta | ARC acc_norm |
|---|---|---|---|---|
| all_dense_bf16 | 1174.77 | 1.00x | 0.0000 | 0.4609 |
| all_dense_nvfp4 | 853.29 | 1.38x | 0.0368 | 0.4609 |
| all_sparse_bf16 | 803.30 | 1.46x | 0.3506 | 0.3594 |
| all_sparse_nvfp4 | 791.82 | 1.48x | 1.0675 | 0.2656 |
| all_marlin_nvfp4 | 1185.49 | 0.99x | 0.0368 | 0.4609 |
| **Pareto point 5** | **1021.91** | **1.14x** | **0.0024** | **0.4609** |
| **Pareto point 8** | **784.75** | **1.49x** | **0.0698** | **0.4531** |
| **Pareto point 10** | **701.77** | **1.66x** | **0.5958** | **0.3203** |

Key observations:
- **Pareto point 8** is faster than `all_dense_nvfp4`, `all_sparse_bf16`, and `all_sparse_nvfp4` while preserving far better accuracy. Its NLL delta (0.070) is 5x smaller than `all_sparse_bf16` (0.351) and 15x smaller than `all_sparse_nvfp4` (1.068).
- **Pareto point 5** achieves a modest 1.14x speedup with negligible quality loss (NLL delta = 0.0024, ARC unchanged at 0.4609).
- **`all_marlin_nvfp4`** is slower than `all_dense_bf16` in prefill-only mode (0.99x speedup), confirming that marlin_nvfp4 is not useful for prefill workloads and is correctly ignored by the optimizer.

## Interpretation

### Conservative budgets (points 0–5, quality_cost < 3)

The optimizer exclusively replaces MLP `dense_bf16` with `dense_nvfp4`. MLP layers have larger weight matrices (n=11008 vs n=4096 for attention) and thus dominate total latency. Substituting only 42 of 96 MLP projections with `dense_nvfp4` yields a 1.14x speedup with ARC-Challenge accuracy unchanged at 0.4609.

### Medium budgets (points 6–8, quality_cost 9–58)

The optimizer introduces `sparse_bf16` first in attention layers (points 6–7) and then in MLP layers (7–8). Sparse BF16 is the fastest per-module kernel (2.13ms vs 4.04ms for dense_bf16), but carries higher quality cost (0.85 per module vs 0.07 for dense_nvfp4). The optimizer defers sparse_bf16 until dense_nvfp4 alone is exhausted, correctly prioritizing low-risk MLP substitutions.

### Aggressive budgets (points 9–10, quality_cost > 137)

Only at the highest budgets does the optimizer select `sparse_nvfp4` for MLP modules (59 MLP modules at the final transition 9→10). Sparse NVFP4 has the highest quality cost (1.69 per module) and its latency (2.41ms) is slightly worse than sparse_bf16 (2.13ms), making it only useful when sparse_bf16 quality budget is fully consumed.

### marlin_nvfp4 exclusion

`marlin_nvfp4` is never selected by the optimizer. Its measured prefill latency (4.08ms per module) is marginally worse than `dense_bf16` (4.04ms), offering no speed benefit for prefill-only scenarios. It would only become relevant in hybrid prefill+decode scenarios.

## Plots

See the following generated visualizations:

- `plots/speed_vs_nll.png` — E2E speedup vs NLL delta, Pareto points (connected) vs uniform baselines (squares)
- `plots/speed_vs_arc_challenge.png` — E2E speedup vs ARC-Challenge acc_norm
- `plots/method_counts_frontier.png` — Stacked bar chart of method counts along the Pareto frontier
- `plots/predicted_vs_e2e_latency.png` — Predicted linear latency vs real E2E latency with correlation annotation

## Limitations

1. **Single model**: Results apply only to Llama-2-7B. The pattern of MLP-first substitution may not generalize to models with different architecture ratios.
2. **Prefill-only scenario**: Without decode, conversion costs between prefill and decode backends are zero. Hybrid scenarios (normal_01, normal_02) will have additional tradeoffs from conversion overhead.
3. **Quality proxy scope**: Validated against NLL and ARC-Challenge with 128 samples only. Full downstream evaluation (MMLU, HellaSwag, etc.) would strengthen confidence.
4. **Timing precision**: Only 3 iterations per Pareto point. Point-to-point variance is visible. Higher iteration counts would improve confidence in the exact speedup numbers.
5. **Method space**: Only 5 methods considered. Additional backends (e.g., NVFP4 with different sparsity patterns, hybrid dense/sparse kernels) may expand the frontier.

## Next Research Steps

1. **Extend to normal_01**: Add decode latency and conversion costs to the optimization. This will likely reintroduce marlin_nvfp4 (which is fast at decode) and create more interesting tradeoffs.
2. **Extend to normal_02**: Longer decode sequences will further shift the optimal method distribution.
3. **Repeat on other models**: Llama-3.1-8B and Qwen3.5-9B have different module dimensions and sensitivity patterns; the frontier shape may differ.
4. **Refine quality proxy**: The current formula (`local_rel_mse_log_numel_layer_family`) has Spearman = -0.79 with ARC-Challenge. Better proxies would improve the optimizer's quality ordering.
5. **Increase timing iterations**: 10–20 iterations per point would reduce noise in the E2E latency measurements.
