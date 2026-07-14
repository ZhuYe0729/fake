# Llama-3.1-8B-Instruct prefill-decode Pareto plan

## Objective

Construct and validate a measured Pareto frontier for the `batch=16`,
`input=2048`, `output=80` phase-heterogeneous vLLM workload.  The resulting
artifact must compare mixed policies with the existing, frozen uniform vLLM
baselines and evaluate the selected mixed points on all three PMPD tasks.

## Execution gates

1. Create a self-contained debug root and freeze the Llama-3.1 architecture,
   action support, fixed WikiText samples, and a 72-policy (54 train / 18
   holdout) phase-aware calibration design.
2. Collect prefill and decode local errors and 100-block phase NLL labels;
   fit and validate the positive local+global phase quality proxy with target
   `delta_prefill + 80 * delta_decode`.
3. Generate a stratified 12-policy speed-calibration set, measure TTFT and
   main/decode latency with the established continuous phase-heterogeneous
   vLLM protocol, and fit a train-only monotone E2E correction to the kernel
   predictor sum.
4. Solve the phase-pair multiple-choice knapsack for a quality-budget sweep,
   retaining uniform policies as explicit reference points.
5. Freshly measure selected policies' E2E latency and WikiText NLL, generate
   a paper-style measured frontier, and only display measured axes.
6. Run CNN/DM-1000, DialogSum-1500, and IWSLT-333 for the non-dominated mixed
   points using the continuous phase runner; merge them with the existing
   frozen uniform results and plot each task frontier.

## Protocol invariants

- Model: `/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- vLLM scenario: prefill-decode, batch 16, input length 2048, output 80.
- Quality target: `delta_prefill_nll + 80 * delta_decode_nll` on fixed
  WikiText blocks.  Training/holdout policy split stays frozen.
- Speed surrogate: CUTLASS roofline/kernel predictor aggregate plus a
  train-only monotone E2E correction.  The held-out calibration set is never
  used in the correction fit.
- Existing baseline artifacts are inputs only; do not overwrite them.
- GPU 1--7 may be used; temporary exported checkpoints are concurrency
  limited because each consumes approximately 15 GB of shared disk.

## Success evidence

The debug root contains quality and speed validation metrics, predicted policy
manifests, fresh closure CSVs, a measured NLL--speed plot/CSV, task summaries,
and three task Pareto plots.  Any unavailable or OOM policy is recorded as an
infeasible measured point rather than silently treated as fast.
