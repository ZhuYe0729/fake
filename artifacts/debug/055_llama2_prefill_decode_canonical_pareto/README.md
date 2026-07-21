# Llama2-7B-chat canonical prefill-decode Pareto

This isolated experiment replaces all historical direct-prune and mixed-runner
prefill-decode quality artifacts. It reuses only the verified canonical sparse
states from debug 054 and evaluates true vLLM phase switching at input 2048,
output 80.

Run `scripts/bootstrap.py`, then use the smoke policy with the canonical-aware
debug-044 `stream_phase_policy_nll.py`. The smoke gate must record decode phase
trace events and exporter provenance before any calibration jobs are launched.
