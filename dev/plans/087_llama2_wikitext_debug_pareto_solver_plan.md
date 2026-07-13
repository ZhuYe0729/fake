# 087 Llama2 WikiText Debug Pareto Solver Plan

## Objective
- Produce a reproducible predicted Pareto frontier using the validated WikiText normalized pooled quality proxy and the existing raw kernel latency predictor.
- Keep all policies and results under a new debug directory; do not promote points to exported results.

## Decisions
- Freeze proxy coefficients by refitting the 54-policy WikiText training split and serialize the normalization/factor parameters.
- Candidate actions use vLLM fused modules and runtime-legal prefill/decode method pairs.
- Optimize additive predicted quality contribution against raw predicted linear latency; omit the fitted intercept because it is policy-invariant.
- Emit a finite grid of quality budgets and vLLM phase-policy JSONs. No vLLM E2E/quality validation runs in this plan.

## Verification
- Check all 128 modules have dense baseline and at least one legal compressed action.
- Check dense endpoint has zero incremental quality and max-speed endpoint matches independent greedy speed selection.
- Report predicted-only status and known speed/quality validation limits in the summary.
