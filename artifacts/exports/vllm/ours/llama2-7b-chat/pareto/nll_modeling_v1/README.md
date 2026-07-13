# Llama2-7B-Chat Pareto modeling v1

This directory prepares the inputs for a future Pareto solver; it does **not** solve a frontier yet.

- `policies/`: 30 deterministic heterogeneous phase policies per scenario; `p00`--`p20` train the NLL proxy and `p21`--`p29` are fixed holdouts.
- `samples/`: 100 examples each from CNN/DM, DialogSum, and IWSLT.
- `nll/`: teacher-forced policy NLL measurements. Prefill/decode uses `Δprefill + 80 × Δdecode`.
- `quality_model/`: local-error feature table, calibrated positive additive model, and holdout metrics.
- `speed_model/`: raw `KernelLatencyPredictor` aggregation and (after launch) vLLM E2E validation.

`w4a16_ours` uses the dense-NVFP4 weight proxy for NLL, because it is a runtime route of the same quantized representation. Local errors are measured on the corresponding unfused HF projections and aggregated into vLLM's qkv/gate-up groups.

## TODO

Use the validated per-module NLL cost and raw speed cost in a constrained multi-choice optimizer, then independently validate selected frontier points with vLLM.
