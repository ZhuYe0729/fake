# vLLM phase-switch overlap ablation plan

## Goal

Produce a compact, paper-ready runtime ablation showing that asynchronously
prefetching decode representations during prefill reduces the exposed phase
switching overhead.

## Fixed scope and assumptions

- Model: Llama3.1-8B-Instruct, matching the other modeling ablations.
- Scenario: prefill-decode, batch 8, input length 2048, output length 64.
- Runtime: the same `phase_hetero_mytest` vLLM backend used by the main
  experiment.
- Policy: canonical solved `point_011` (the measured max-speed Pareto point),
  whose 128 Linear modules all use different prefill and decode methods. This
  exposes the runtime mechanism using an actual policy produced by our method,
  rather than a synthetic stress policy.
- Comparison changes only `PHASE_HETERO_SYNC_COPY`:
  - `1`: copy all decode representations synchronously at the phase boundary;
  - `0`: schedule per-layer H2D copies on the dedicated copy stream and overlap
    them with later prefill computation.
- Formal latency runs disable tracing. Both variants run alternately on the
  same idle GPU with identical checkpoint, prompts, vLLM configuration, and
  measurement count.

## Steps and verification

1. Create a self-contained debug artifact with a checkpoint materializer and
   paired benchmark runner.
   - Verify: dry-run/config audit confirms the policy has 128/128 phase-switched
     Linear modules and both commands differ only in the sync-copy flag.
2. Materialize and verify one canonical `point_011` checkpoint.
   - Verify: the existing canonical-checkpoint verifier passes.
3. Run paired formal latency measurements with trace disabled.
   - Verify: each variant has at least five successful measured samples, no OOM,
     and no concurrent GPU workload during retained samples.
4. Summarize raw samples, medians, relative latency reduction, and generate a
   compact table/figure plus a short paper-facing interpretation.
   - Verify: summary can be rebuilt from retained CSV/JSON data.
5. Run an isolated CUDA-event diagnostic that measures total decode-state
   transfer work and the copy-event stalls actually exposed on the compute
   stream. Derive the hidden-copy ratio from these two quantities.
   - Verify: profiling is enabled only by a diagnostic-process monkey patch,
     uses the same checkpoint/workload, reports three independent samples, and
     is not mixed into formal E2E timing.

## Non-goals

- No changes to the main Pareto results or modeling ablation table.
- No runtime redesign or new overlap mechanism.
- No broad sweep over models, policies, batches, or output lengths.
