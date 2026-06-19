# FakeVLM Uniform Compression Accuracy Plan

## Summary
- Add a debug-only FakeVLM accuracy workflow under `artifacts/debug/020_fakevlm_uniform_accuracy`.
- Evaluate six uniform methods: dense BF16, sparse BF16, dense NVFP4, sparse NVFP4, Marlin weight-only, and dense NVFP4 prefill plus Marlin decode.
- Use `cospaq`, `/home/agent/wja/data/...` model/data paths, and one process per GPU in physical GPU order `7,6,5,4,3,2`.
- Test accuracy first; leave speed as TODO.

## Key Changes
- Add a `fakevlm`/`llava` compressible-module selector for language-model transformer linear layers only.
- Build a FakeVLM eval script based on `third_party/FakeVLM/scripts/eval.py`, saving predictions, accuracy, configs, logs, and compression metadata in the debug directory.
- Use calibrated pruning for sparse BF16 and sparse NVFP4: collect FakeClue activation Hessian/importance, prune weights, then install the real runtime wrapper with wrapper-side pruning disabled.
- Use real NVFP4 runtime inference for W4A4 methods; activation global scale is computed online per forward, not loaded from offline activation metadata.
- Set hybrid decode threshold to `m <= 8`.

## Test Plan
- Run `conda run -n cospaq python -m py_compile` for the new Python scripts.
- Run help/CPU-side checks and verify the launcher never assigns GPU 0 or 1.
- Run smoke accuracy with 5 samples on GPUs `7,6,5,4,3,2`.
- Run full 5000-sample accuracy in parallel, one method per GPU, then summarize all methods.

## Assumptions
- Vision tower, projector, and `lm_head` remain BF16.
- "Uniform" means every selected language-model linear uses the same method; the hybrid method differs only by prefill/decode backend.
- Static activation calibration and speed benchmarking are separate follow-up work.
