# Llama2-7B prefill-only Pareto rebuild

This debug root rebuilds the `batch=8`, `input=2048` prefill-only frontier.
It does not replace `034_llama2_7b_chat_wikitext_pareto_solver`.

## Frozen starting point

- Candidate policies and raw roofline/local-residual latency inputs are from
  `034_llama2_7b_chat_wikitext_pareto_solver/prefill_only/pareto/`.
- Existing real points are 0, 4, 8, 12 and 16. Their speed measurements use
  the established `benchmark_phase_baseline_one.py` prefill runner.
- Baseline anchors remain the existing vLLM prefill-only measurements until a
  protocol mismatch is demonstrated.

## Calibration split

The first E2E speed calibration sweep uses points 1, 3, 6, 9, 11, 13 and 15:
it fills the intervals between existing measured points without fitting only
the final displayed candidates. Points 0, 4, 8, 12 and 16 are reserved as
independent held-out speed checks initially.

## Status

Input audit complete; checkpoint export and fixed-protocol E2E measurements
are the next step. New GPU work is restricted to devices 0--4.
