# 115 Llama2 canonical prefill-decode Pareto plan

## Objective

Build a fresh, canonical-sparse, phase-runtime-consistent Pareto result for
Llama2-7B-chat under `batch=16, input_len=2048, output_len=80`, including
separate prefill/decode policies, real-vLLM quality, E2E speed, and three
generation-task datasets.

## Fixed experimental contract

- Model/tokenizer: `/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf`.
- Runtime: vLLM `phase_hetero_mytest`, with prefill/decode method switching
  enabled in a single loaded engine.
- Sparse source: reuse the verified Llama2 canonical sparse BF16 and sparse
  NVFP4 states from debug 054. Canonical mode must reject `--prune`.
- Quality target: teacher-forced, real-vLLM sequence NLL over fixed WikiText
  samples with 2048-token prompt + 80 decode targets; quality loss includes
  the actual decode-phase switch.
- Speed target: five loaded-engine repeats using the pre-existing
  `run_prefill_decode_baseline_aligned.sh` semantics: output 1 for TTFT and
  output 80 for E2E; report TTFT, E2E and TPOT `(E2E - TTFT) / 79`.
- Final generation tasks: CNN/DM, DialogSum and IWSLT, retaining every metric
  emitted by the existing vLLM evaluators.

## Execution plan

1. **Bootstrap and smoke gate**
   - Create isolated `artifacts/debug/055_llama2_prefill_decode_canonical_pareto/`.
   - Link/copy only metadata references to 054 canonical states; do not mutate
     the prefill-only experiment.
   - Export one mixed policy with sparse prefill and distinct decode methods.
     Verify phase trace proves both phase maps and canonical sources are used.

2. **Quality calibration**
   - Adapt the debug 044 real-vLLM teacher-forced decode NLL evaluator to accept
     canonical sparse states and reject direct prune.
   - Reuse/generate fixed 2048+80 WikiText samples and evaluate uniform plus
     controlled/mixed calibration policies under the true phase switch.
   - Fit the existing local+global quality proxy with separate prefill and
     decode method features. Report train/holdout MAE, RMSE, rank correlation,
     residual plots, and uniform endpoint residuals.

3. **Speed calibration**
   - Reuse the roofline + kernel calibration predictor, but validate both
     phases against real phase-runtime TTFT/E2E runs.
   - Fit only a natural scenario-level calibration to the predicted prefill and
     decode components if required by measured residuals; do not turn it into
     a speed lookup table.
   - Measure uniform phase policies on the identical runner before comparing
     them with ours.

4. **Joint constrained solve**
   - Solve for a series of quality budgets over two independent layer-method
     maps (prefill and decode), optimizing predicted E2E `TTFT + 79*TPOT`.
   - Preserve raw predicted prefill/decode/E2E components and policy JSON for
     every point. Pareto points are candidates, not final results.

5. **Real-runtime closure**
   - Test endpoints and dense-NVFP4 neighborhoods first; add neighboring
     points whenever a uniform reference falls outside the measured mixed
     frontier.
   - For each selected point, measure canonical real-vLLM NLL and five-repeat
     TTFT/E2E/TPOT. Normalize uniform and ours to the same phase dense-BF16
     reference.

6. **Downstream closure and paper artifacts**
   - Select about six mixed points spanning quality to max speed and run all
     three generation datasets; run the canonical uniform baselines with the
     exact same evaluator and stage switching.
   - Generate combined CSV/Markdown, NLL and per-metric Pareto plots, marking
     measured versus predicted values clearly.

## Hard gates

- No NLL, downstream score, or Pareto plot may use a checkpoint exported with
  direct `--prune`.
- No speed comparison may mix the legacy uniform runner with phase runtime.
- The NLL evaluator must observe the prefill-to-decode switch, not merely a
  prefill proxy.
- Do not launch all task datasets before the uniform endpoints and a few mixed
  NLL/speed points demonstrate a credible measured frontier.

## Existing references

- Canonical contract and safeguards:
  `artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/POSTMORTEM.md`.
- Teacher-forced decode-NLL implementation:
  `artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/evaluate_runtime_decode_nll.py`.
- Earlier E2E calibration structure (reference only; do not reuse old direct-prune quality):
  `artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/`.
- Baseline-aligned E2E runner:
  `artifacts/exports/vllm/ours/llama2-7b-chat/scripts/run_prefill_decode_baseline_aligned.sh`.
