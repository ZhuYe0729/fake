# Llama3.1 prefill warmed E2E speed revalidation plan

## Goal

Regenerate only the Llama3.1-8B-Instruct prefill-only E2E speed calibration
and measured closure under the verified timing protocol. Reuse the existing
canonical weights, local roofline predictor, quality model, and discrete
solver formulation.

## Fixed protocol

- B=8, input length=2048, output length=1.
- `phase_hetero_mytest` runtime for every method, including uniform baselines.
- `max_num_seqs=8`, `max_num_batched_tokens=16384`.
- eager execution, prefix cache disabled, chunked prefill disabled.
- One loaded vLLM engine per policy; one unrecorded warmup request; then five
  timed requests in that same engine.
- Explicitly request the prefill phase before each request.

## Steps

1. Create an isolated 061 experiment with a warmed benchmark and reusable
   anchor scheduler.
   - Verify: its JSON records all fixed protocol fields plus five timings.
2. Recreate the existing 12-policy anchor design and measure it only when GPUs
   are available.
   - Verify: every policy has exactly five same-engine timed samples.
3. Fit the existing monotone raw-linear-to-E2E calibration on the new anchor
   data and report holdout error.
   - Verify: no historical 058 speed measurements are read as labels.
4. Re-run the unchanged discrete quality-constrained solver against the new
   calibration, then measure representative/max-speed closure points with the
   same runner.
   - Verify: predicted and measured E2E values use the fixed protocol.
5. Rebuild the Llama3 prefill Pareto/table artifacts from the new closure.

## Non-goals

- Do not regenerate SparseGPT canonical weights.
- Do not retrain the real-vLLM NLL quality proxy.
- Do not modify or overwrite 058 historical artifacts.
