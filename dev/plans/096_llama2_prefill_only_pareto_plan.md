# Llama2-7B Prefill-only Pareto Plan

## Objective

Build a validated Pareto frontier for Llama2-7B-Chat on vLLM prefill-only
(`batch=8`, `input=2048`) without promoting the preliminary 034 solver curve.
All new evidence remains under a new debug artifact directory until it is
independently validated.

## Starting evidence and known gaps

- 034 contains a useful discrete-policy solver and five real WikiText points,
  but policy selection was driven by the raw sum of kernel predictions.
- Its five points show a plausible quality/speed trade-off, but do not test the
  full candidate curve or a corrected E2E latency surrogate.
- The prefill-decode work established the required pattern: separate protocol,
  policy-level E2E calibration, real WikiText NLL for selected points, then
  task-level checks for representative Pareto points.

## Fixed protocol

- Model: `Llama-2-7b-chat-hf`; vLLM phase-heterogeneous runtime.
- Scenario: prefill-only, batch 8, input length 2048, one generated token only
  when required by the vLLM benchmark API.
- Baselines use the existing prefill baseline runner/protocol. New comparisons
  will never mix a different runner or memory-utilization setting silently.
- New GPU work uses only GPU 0--4.

## Plan

1. Audit and freeze inputs: reconstruct 034 action space, policy JSONs,
   existing measurements, and exact runner settings into a new debug folder.
   Verify that dense and uniform anchors can be reproduced from source files.
2. Build prefill-only policy-level latency features from the existing
   roofline-plus-local-residual kernel model. Measure a distributed calibration
   set with the same runner and fit a monotone E2E correction. Verify held-out
   latency error and rank agreement improve over raw kernel summation.
3. Refit/use the validated WikiText quality proxy for prefill-only and inspect
   held-out ranking/error against real NLL. If existing proxy data is adequate,
   reuse it; otherwise collect only the missing controlled calibration points.
4. Solve a discrete, layer-heterogeneous frontier using corrected predicted
   latency and additive quality cost. Emit policy assignments, feasibility
   checks, and dense/max-speed endpoint audits.
5. Measure real fixed-protocol E2E speed and pooled WikiText ΔNLL for the
   selected frontier and uniform anchors. Replot using measured axes and label
   any unstable measurement rather than filtering it silently.
6. Run the three existing generation tasks for two to four representative,
   nondominated policies only after the NLL/speed curve is credible; produce
   task-specific Pareto figures. Keep optional dense-grid refinement as TODO.

## Success criteria

- A new artifact root holds scripts, inputs, measurements, fitted speed model,
  policies, and a report with reproducible commands.
- Corrected speed surrogate is evaluated on held-out policies, not merely fit
  on the displayed frontier.
- Final plotted axes are actual E2E time and actual WikiText NLL for every
  displayed non-baseline policy.
- At least one heterogeneous point remains nondominated against measured
  uniform references, or the report explicitly records the negative outcome.
