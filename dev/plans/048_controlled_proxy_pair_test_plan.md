# Controlled Proxy Pair Test Plan

## Summary
- Add a controlled policy-pair test under `016` to reduce the count/local-sum confound in the current ablation.
- For each method, generate pairs with the same selected module count and nearly matched raw local-error sum, but large final layer/type proxy gap.
- Run real loss on those controlled policies and compare local-only versus final layer/type pairwise prediction.

## Key Changes
- Add controlled policy generation script.
- Extend existing loss runners with optional policy CSV/output tag support.
- Add controlled analysis script that reports pairwise direction accuracy and prediction correlations.
- Store outputs under `artifacts/debug/016_llama2_sparse_bf16_precision_proxy/controlled/`.

## Test Plan
- Generate controlled policies for `sparse_bf16`, `dense_nvfp4`, and `sparse_nvfp4`.
- Smoke-run two controlled policies per method.
- Full-run controlled policies, then analyze pairwise results.

## Assumptions
- Use final layer/type coefficients without dense NVFP4 nonlinear calibration for pair construction.
- Default controlled set uses counts `32,64` with 8 pairs per count per method.
