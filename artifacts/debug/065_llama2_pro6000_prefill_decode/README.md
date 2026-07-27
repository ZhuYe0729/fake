# Llama2-7B-Chat / RTX Pro 6000 / prefill-decode

Independent B=8, S=2048, O=64 paper experiment. Historical bundles define the protocol only;
all experiment-owned runtime code and outputs are under this directory.

## Frozen protocol

- vLLM V1 `phase_hetero_mytest`, BF16 KV, eager mode.
- Prefix cache and chunked prefill disabled and asserted after engine creation.
- `gpu_memory_utilization=0.80`; teacher-forcing scheduler capacity 16896 tokens.
- Uniform p00-p04 and mixed policies share the same exporter and runtime.
- Formal speed uses one exclusive GPU, one fresh process per policy, 1 warmup + 5 measured runs.
- Predictor cost is prefill M=16384 plus 63 decode steps at M=8; final claims use measured E2E speed.

## Run

```bash
source artifacts/debug/065_llama2_pro6000_prefill_decode/config.current.env
RUNNER=artifacts/debug/065_llama2_pro6000_prefill_decode/scripts/run_stage.py
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
