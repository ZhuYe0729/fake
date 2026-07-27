# Llama3.1-8B-Instruct / RTX Pro 6000 / prefill-decode

Independent B=8, S=2048, O=64 paper experiment. Historical bundles define the
protocol and policy design only; all active code and outputs are under 066.

## Frozen protocol

- vLLM V1 `phase_hetero_mytest`, BF16 KV, eager mode.
- Prefix cache and chunked prefill disabled and asserted after engine creation.
- `gpu_memory_utilization=0.80`; teacher-forcing scheduler capacity 16896 tokens.
- Uniform p00-p04 and mixed policies share the same exporter and runtime.
- Formal speed uses one exclusive GPU, one fresh process per policy, 1 warmup + 5 measured runs.
- Predictor cost is prefill M=16384 plus 63 decode steps at M=8; final claims use measured E2E speed.
- Llama3 canonical sparse states are regenerated locally because the historical large states are unavailable.
- PMPD uses the user-selected Legacy/common Human/Assistant prompt protocol; this is recorded in every task manifest.

## Run

```bash
source artifacts/debug/066_llama31_pro6000_prefill_decode/config.current.env
RUNNER=artifacts/debug/066_llama31_pro6000_prefill_decode/scripts/run_stage.py
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

Every stage is resumable. A file is skipped only after its protocol and provenance checks pass.
Dataset payloads remain in shared machine caches; their hashes and selection metadata are stored here.

## Completed run

- `validation/all.json`: all checks pass.
- Canonical states: 224/224 modules for both sparse BF16 and sparse NVFP4.
- Calibration: 72/72 policies, 100 blocks per policy.
- Quality proxy holdout: MAE 0.100368, Spearman 0.9360.
- Hardware model: eight exact Llama3 shapes and 1280 audited actions.
- Closure: 5 uniform policies plus 21 predicted points, all with 100-block NLL and single-process 1+5 E2E timing.
- Downstream: 11 selected policies × 3 datasets = 33 complete Legacy-protocol metric rows.
- Consolidated table: `results/complete_results.csv`; paper-candidate plots are in `results/figures/`.
