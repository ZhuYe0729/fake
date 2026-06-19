# 055 FakeVLM Linear-Layer Hybrid Max-Speed Prefill Plan

## Summary
Implement a new debug experiment for FakeVLM prefill-only maximum speed under `artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/`, comparing two linear-layer hybrid selection paths:

- `manual_profile`: measure candidate backend latency per unique `(m,n,k)` shape, then choose the fastest backend for each individual FakeVLM language `nn.Linear`.
- `latency_model`: use `KernelLatencyPredictor` to choose the fastest backend for the same per-linear candidate set.

Run scope is prefill-only only: real FakeVLM prompt forward via `model(**inputs, use_cache=False)`, not `generate()`. Prefill-decode support is left as an explicit TODO.

## Key Changes
- Add debug-only code under `artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/code/`:
  - dataset/input helpers adapted from `020_fakevlm_uniform_accuracy` and `third_party/FakeVLM/scripts/eval.py`
  - per-linear policy builder for FakeVLM using `select_compressible_modules(model, "fakevlm")`
  - runtime replacer supporting `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`
  - benchmark runner that loads FakeVLM locally, applies a policy, and times prefill forward on real processed image/text batches
- Add launch/summarize scripts:
  - `run_prefill_speed.sh`: default `CUDA_VISIBLE_DEVICES=0,1`, conda env `cospaq`, batch sizes `1 2 4 8 16`, warmup/iters configurable
  - `summarize_prefill_speed.py`: joins manual/model policies and measured E2E prefill speed into summary CSV/Markdown
- Output layout:
  - `policies/manual_profile/batch_<bs>/policy.json|csv`
  - `policies/latency_model/batch_<bs>/policy.json|csv`
  - `candidates/manual_profile/batch_<bs>.csv`
  - `candidates/latency_model/batch_<bs>.csv`
  - `speed/prefill_speed.csv`
  - `summary/prefill_speed_summary.csv`
  - `summary/prefill_speed_summary.md`
  - `TODO_prefill_decode.md`

## Test Plan
- Static checks:
  - `python -m py_compile` for new Python files
  - `bash -n` for shell launchers
  - `conda run -n cospaq python ... --help`
- GPU smoke:
  - `CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZES=1 SAMPLE_LIMIT=2 WARMUP=1 ITERS=2 bash .../run_prefill_speed.sh`
  - verify candidate CSVs, policy JSON/CSV, speed CSV, and summary files are created
- Full run:
  - `CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZES="1 2 4 8 16" bash .../run_prefill_speed.sh`
  - verify both `manual_profile` and `latency_model` have measured rows for all batch sizes
  - verify all rows record device name, visible devices, replaced/skipped counts, backend counts, warmup/iters, and sample count

## Assumptions
- “最大速度” means fastest measured prefill-only runtime, with no accuracy constraint in this experiment.
- “linear layer 粒度” means policy entries are per module name, not only per layer type/group; identical shapes can share cached candidate measurements.
- Existing FakeVLM compression target stays language-model-only: vision tower and multimodal projector are not replaced.
- Prefill-decode hybrid routing is intentionally not implemented here; add only a TODO describing required decode timing and backend transition work.
