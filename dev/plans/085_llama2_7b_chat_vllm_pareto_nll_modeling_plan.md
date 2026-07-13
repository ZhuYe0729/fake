# 085 Llama2-7B-Chat vLLM Pareto NLL Modeling Plan

## Objective
- Build and validate the two predictive inputs needed before Pareto solving for Llama-2-7B-Chat vLLM: raw kernel-latency prediction and an NLL-based quality model.
- Keep the optimizer/frontier solver explicitly out of scope (TODO).

## Decisions
- Store the work in `artifacts/exports/vllm/ours/llama2-7b-chat/pareto/nll_modeling_v1/`.
- Use 30 fixed, heterogeneous calibration policies per scenario. 21 policies train the quality coefficients and 9 fixed policies are held out only to measure generalization.
- The data budget is 100 deterministic examples each from CNN/DM, DialogSum, and IWSLT (300 examples/scenario).
- Measure NLL with teacher forcing: prefill-only uses prompt-token NLL; prefill-decode uses `ΔNLL_prefill + 80 * ΔNLL_decode`.
- Candidate runtime methods remain dense BF16, dense NVFP4, sparse BF16, sparse NVFP4, and w4a16-ours; quality maps w4a16-ours to dense-NVFP4 weights.
- The v1 quality proxy is a positive additive aggregation of calibrated local output errors, with global/method/layer/type multipliers.
- Speed validation compares raw aggregate `KernelLatencyPredictor` estimates to fresh-process vLLM E2E measurements under the already adopted per-scenario protocols. It deliberately does not fit an E2E correction table.

## Verification
- Check policy completeness, deterministic split, and sample composition.
- Record train and holdout MAE/RMSE/Spearman for each scenario's NLL model.
- Record prediction-versus-vLLM error for selected speed-validation policies, separating raw linear prediction from end-to-end runtime.
- Produce one README/summary stating limitations and an explicit Pareto-solver TODO.
