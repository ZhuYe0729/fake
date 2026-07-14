# Llama-3.1-8B-Instruct prefill-only Pareto experiment

This directory is an independent, gated experiment for `b=8`, `input=2048`,
`output=1`.  It does not reuse Llama2 fitted coefficients, policies, E2E
calibration, or measurements.

Run gates in order:

1. `scripts/audit_prefill_only.py` writes the architecture/action manifest.
2. `scripts/generate_inputs.py` freezes Llama3-tokenized WikiText blocks and
   the 72-policy (54 train / 18 holdout) quality design.
3. Collect `local_errors/`, then run the NLL shards and `fit_quality_proxy.py`.
4. Measure the fixed speed calibration policies, fit the monotone correction,
   and only then solve and close the measured frontier.

The final reported curve must use measured vLLM speed and measured WikiText
NLL, jointly non-dominated with the five uniform references.
