# Prefill-decode E2E speed-model debug plan

## Goal

Improve the Llama2-7B prefill-decode speed surrogate used before Pareto solving. The current objective is a linear sum of kernel predictions and overestimates end-to-end vLLM gains for intermediate heterogeneous policies.

## Assumptions

- The historical phase-heterogeneous runner (`batch=16`, `input=2048`, `output=80`, `.9` memory) is the formal measurement protocol.
- Existing 10-repeat measurements are retained as anchors; additional calibration policies may use fewer repeats for model fitting, with held-out policies remeasured at 10 repeats.
- This is a debug experiment. New data stays under `artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/`.

## Plan

1. Export policy-level E2E features from the existing kernel model and phase assignments → verify deterministic feature table for every predicted candidate.
2. Measure a distributed set of feasible policies with the formal runner and record OOM as an explicit feasibility outcome → verify all completed runs have paired output-1/output-80 samples.
3. Fit a regularized E2E correction model and compare it against the raw linear predictor on held-out policies → verify error metrics improve and ranking is reported.
4. Keep Pareto re-solving as TODO until the corrected model and feasibility treatment pass validation.

