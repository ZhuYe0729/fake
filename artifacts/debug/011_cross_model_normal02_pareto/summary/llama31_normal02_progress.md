# Llama3.1 Normal02 Pareto Progress

## Scope

This run extends the normal_02 quality-speed Pareto workflow beyond Llama2, using:

- Model: `llama31-8b`
- Scenario: `normal_02`, batch 1, prefill 16384, decode 256
- Quality source: module-level local output error on WikiText-2 calibration
- Latency source: kernel predictor normal_02 per-linear candidates
- Root: `fake/artifacts/debug/011_cross_model_normal02_pareto`

## Generated Assets

- `scripts/generate_pred_normal02.py`
  - Generates normal_02 predictor candidates for `llama31-8b` and `qwen35-9b` without E2E runs.
- `scripts/collect_sensitivity.py`
  - Parameterized copy of the 007 sensitivity collector.
  - Supports `llama2-7b`, `llama31-8b`, and `qwen35-9b`.
- `scripts/build_cost_table.py`
  - Joins module quality rows with normal_02 latency candidates.
- `scripts/optimize_pareto.py`
  - Solves quality-constrained latency minimization per model.
- `scripts/validate_pareto_e2e.py`
  - Runs real full-model E2E validation for Llama policies.
  - Fixed policy lookup to match the current `pareto_points.csv` budget exactly, avoiding stale smoke policies.

## Llama3.1 Quality Collection

Full sensitivity completed on GPU 7:

- Calibration: WikiText-2, 16 samples, seq_len 512, seed 0
- Dense calibration NLL: 2.2162
- Modules: 224
- Method-error rows: 672
- Output: `models/llama31-8b/sensitivity/module_method_errors.csv`

Local relative MSE summary from the cost table:

| method | mean local_rel_mse | max local_rel_mse |
|---|---:|---:|
| dense_nvfp4 | 0.00452 | 0.01528 |
| sparse_bf16 | 0.04009 | 0.08135 |
| sparse_nvfp4 | 0.07964 | 0.16339 |

This ordering is sensible: dense NVFP4 is the lowest quality-cost compressed method, sparse BF16 is higher, and sparse NVFP4 is highest.

## Llama3.1 Pareto Frontier

Cost table:

- Rows: 1344
- Missing latency rows: 0
- Output: `models/llama31-8b/costs/module_method_candidates.csv`

The full frontier has 10 unique points. Important points:

| point | proxy quality | predicted linear speedup | bf16 | marlin | hybrid dense-prefill/marlin-decode |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 1.000x | 224 | 0 | 0 |
| 7 | 5.7714 | 1.465x | 151 | 0 | 73 |
| 9 | 14.2241 | 1.710x | 64 | 64 | 96 |

The optimizer first selects MLP `gate_proj/up_proj/down_proj` with `dense_nvfp4` prefill and `marlin_nvfp4` decode. Attention `q_proj/o_proj` marlin choices appear mostly at high quality budgets. Pure `dense_nvfp4` is not selected on this normal_02 frontier because decode dominates and marlin decode wins in the per-linear latency model.

## Real E2E Validation

Ran real E2E on GPU 7 with `iters=3`, `warmup_iters=1`:

| point | predicted speedup | real E2E ms | real speedup | prefill ms | decode avg ms | first decode ms | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.000x | 7070.66 | 1.000x | 1588.31 | 21.42 | 34.16 | ok |
| 7 | 1.465x | 7325.00 | 0.965x | 1282.85 | 23.60 | 97.87 | ok |
| 9 | 1.710x | 7342.81 | 0.963x | 1229.33 | 23.88 | 50.81 | ok |

All three policies replaced 224 linears with `skipped=0`.

Key observation: the per-linear predictor frontier does not transfer to Llama3.1 real E2E under normal_02. Prefill improves substantially, but decode average gets slower enough to erase the gain. Point 7 also has a large first-decode cost. This is a useful negative/control result: the quality-constrained optimizer is working structurally, but its speed objective is not sufficiently E2E-calibrated for this model/scenario.

## Single-Method Baselines

Generated pseudo-Pareto baseline roots with `scripts/generate_baseline_policy_roots.py` and validated two key baselines:

| method | predicted ms | real E2E ms | real speedup vs dense | prefill ms | decode avg ms | first decode ms |
|---|---:|---:|---:|---:|---:|---:|
| dense_bf16 | 4482.02 | 7070.66 | 1.000x | 1588.31 | 21.42 | 34.16 |
| dense_nvfp4 all | 8797.29 | 11688.12 | 0.605x | 1256.27 | 40.75 | 52.50 |
| marlin_nvfp4 all | 3126.12 | 6448.07 | 1.097x | 1596.20 | 18.95 | 31.22 |
| dense_nvfp4_prefill_marlin_decode all | 2895.69 | 6576.94 | 1.075x | 1257.58 | 20.78 | 142.20 |

Combined CSV: `summary/llama31_e2e_baseline_and_pareto.csv`.

This narrows the issue. All-marlin and all-hybrid do produce real speedup, while all-dense-NVFP4 is much slower. The current Pareto mixed policies are slower than dense, so the mismatch is not simply "marlin is bad on Llama3.1"; it is more specifically a mixed-backend full-model overhead / transition-cost issue that the current linear-sum objective does not model. Dense NVFP4 should be treated as a prefill-only backend for this scenario unless a future kernel improves decode.

## Interpretation

For Llama2 normal_02, the same style of frontier produced a real speedup at the fast point. For Llama3.1 normal_02, the current linear-sum latency objective overestimates decode gains. Likely causes to investigate:

- Full-model decode overhead not captured by isolated Linear.forward latency.
- Hybrid dense-prefill/marlin-decode conversion or first-use behavior not fully captured.
- Llama3.1 GQA/KV-cache/runtime composition changes the share of non-linear and attention overhead relative to Llama2.
- Per-linear decode latency at `m=1` may be too optimistic when many modules switch kernels inside full autoregressive decode.

## Recommended Next Step

Do not validate more Pareto quality points for Llama3.1 before recalibrating speed. The next concrete task should be E2E-aware latency calibration and a mixed-backend penalty:

1. Run full-model single-method baselines for Llama3.1 normal_02:
   - dense bf16
   - marlin nvfp4
   - dense_nvfp4_prefill_marlin_decode
   - optionally dense nvfp4
2. Compare each method's predicted linear latency to real E2E prefill/decode components.
3. Add a penalty for mixed decode backend policies or backend transitions:
   - start with features such as number of backend groups used, number of modules not matching the dominant decode backend, and first-decode conversion cost.
   - calibrate this penalty from dense, all-marlin, all-hybrid, point7, and point9.
4. Fit a simple model/scenario calibration:
   - `real_prefill = a_prefill * predicted_prefill + b_prefill`
   - `real_decode = a_decode * predicted_decode + b_decode`
   - optionally separate coefficients for `bf16`, `marlin`, and hybrid decode.
5. Re-run Pareto using calibrated latency cost.
6. Validate points 0, mid, fast again.

Qwen3.5-9B should wait until after this calibration change, otherwise it may repeat the same predictor-to-E2E mismatch.
