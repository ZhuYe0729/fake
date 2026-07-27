# Llama3.1-8B-Instruct / RTX Pro 6000 / prefill-only

Independent, restartable B=8, S=2048, O=1 paper workflow. All experiment-owned
inputs, code, intermediate data and results live under 067. Model files, patched
vLLM, CUTLASS and the shared Hugging Face cache are external read-only dependencies.

## Frozen protocol

- vLLM V1 `phase_hetero_mytest`, BF16 KV and eager execution.
- Prefix cache and chunked prefill disabled and asserted after engine creation.
- Uniform p00-p04 and mixed policies use the same exporter and runtime.
- Formal speed uses one exclusive GPU and one fresh process per policy with
  one warmup plus five measured requests.
- Formal NLL uses 72 Llama3 prefill-only policies and 100 deterministic
  WikiText blocks; p00-p53 are train and p54-p71 are holdout.
- Canonical sparse states are copied from the locally rebuilt 066 states and
  independently hash/structure verified inside 067.

## Run

```bash
source artifacts/debug/067_llama31_pro6000_prefill_only/config.current.env
RUNNER=artifacts/debug/067_llama31_pro6000_prefill_only/scripts/run_stage.py
$COSPAQ_COSPAQ_PYTHON $RUNNER preflight --no-gpu
$COSPAQ_COSPAQ_PYTHON $RUNNER bootstrap
$COSPAQ_COSPAQ_PYTHON $RUNNER canonical
$COSPAQ_COSPAQ_PYTHON $RUNNER prewarm --speed-gpu 0
$COSPAQ_COSPAQ_PYTHON $RUNNER local-errors --gpus 0,1
$COSPAQ_COSPAQ_PYTHON $RUNNER smoke --gpus 0,1
$COSPAQ_COSPAQ_PYTHON $RUNNER nll --gpus 0,1
$COSPAQ_COSPAQ_PYTHON $RUNNER fit
$COSPAQ_COSPAQ_PYTHON $RUNNER profile --speed-gpu 0
$COSPAQ_COSPAQ_PYTHON $RUNNER solve
$COSPAQ_COSPAQ_PYTHON $RUNNER closure --speed-gpu 0
$COSPAQ_COSPAQ_PYTHON $RUNNER select-tasks
$COSPAQ_COSPAQ_PYTHON $RUNNER task-data
$COSPAQ_COSPAQ_PYTHON $RUNNER tasks --gpus 0,1
$COSPAQ_COSPAQ_PYTHON $RUNNER consolidate
$COSPAQ_COSPAQ_PYTHON $RUNNER validate
```

Every stage is resumable and skips only outputs that pass protocol and
provenance checks. Full tasks are WikiText, WinoGrande, ARC-Easy,
ARC-Challenge and MMLU; smoke limits never enter the final table.

## Completed run

- `validation/all.json`: all checks pass.
- Canonical: both methods pass 224/224 module validation and retain the copied
  066 hashes.
- Calibration: 72/72 policies, 100 blocks each; holdout MAE 0.05456 and
  Spearman 0.9546.
- Hardware model: four exact Llama3 prefill shapes, profiled locally on Pro 6000.
- Closure: five uniform plus 24 unique solved points, each with 100-block NLL
  and single-process 1+5 E2E timing.
- Downstream: 11 selected policies times five full tasks, 55/55 complete.
- Final table: `results/complete_results.csv`; six plots are under
  `results/figures/`.
