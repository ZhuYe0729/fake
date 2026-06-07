# Predictor Hybrid Analysis

## Layout

The predictor results are now split by scenario, matching the manual result layout:

| Directory | Contents |
|---|---|
| `prefill_only/` | Prefill-only policies, strategy diffs, module timing summary, and summary markdown |
| `normal_01/` | Normal prefill+decode policies, strategy diffs, module timing summary, full E2E, and summary markdown |
| root `pred/` | Cross-scenario aggregate CSVs and README |

## Key Result

| Model | Scenario | Policy relation to manual | Best comparable measurement | Result |
|---|---|---|---|---|
| Llama-2-7B | prefill_only | Same policy | GPU module vs manual module | +1.15% slower |
| Llama-3.1-8B | prefill_only | Same policy | GPU module vs manual module | +0.93% slower |
| Qwen3.5-9B | prefill_only | 16 layers changed | GPU module vs manual module | -0.25% faster |
| Llama-2-7B | normal_01 | Same policy | Full model vs manual full model | +16.81% slower |
| Llama-3.1-8B | normal_01 | 64 layers changed | Full model vs manual full model | +7.93% slower |
| Qwen3.5-9B | normal_01 | 184 layers changed | Full model vs manual full model | +27.11% slower |

## Interpretation

Prefill-only is mostly validated. For both Llama models, predictor and manual choose identical policies, and the GPU module total is within about 1% of the manual module benchmark. Qwen3.5 changes the small `k_proj`/`v_proj` group from `marlin_nvfp4` to `sparse_bf16`, and the module total is slightly faster than manual.

The normal scenario is more mixed. Llama-2 selects exactly the same policy as manual, but the full model E2E run is slower than the manual result. That means the gap is not caused by predictor policy choice; it is likely due to benchmark path/configuration differences or runtime overhead in the current predictor-hybrid replacement path. Llama-3.1 additionally changes the 64 `k_proj`/`v_proj` layers to `dense_bf16`, but the full E2E result still trails manual.

Qwen3.5 normal is now confirmed as a bad predictor selection for full E2E. The predictor module model strongly prefers `dense_nvfp4/marlin_nvfp4` for 184 layers, while the previous manual full E2E result selected `sparse_bf16` for those layers. The full run replaced all 248 linear layers, but finished at 4204.76ms versus the manual 3308.00ms. This is a module-to-model mismatch: the local linear timing underestimates full-model decode and lazy conversion/materialization costs for the NVFP4 shared policy.

## Practical Conclusion

- Use predictor policy directly for prefill-only Llama and likely Qwen3.5 after one confirming run.
- For normal Llama, the policy selection is plausible, but the full E2E benchmark path must be aligned with the manual benchmark before judging speed.
- For normal Qwen3.5, keep the manual `sparse_bf16` policy as the production reference. The current predictor policy should not replace it without adding model-level calibration or stronger penalties for the W4A4/W4A16 conversion path.
