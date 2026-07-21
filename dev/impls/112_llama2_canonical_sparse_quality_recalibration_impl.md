# 112 Llama2 canonical sparse quality recalibration

## 2026-07-17 - Pipeline implementation started
- Development goal: replace direct magnitude-pruned mixed sparse policies with canonical SparseGPT-calibrated sparse sources.
- Planned changes: add pre-quantized canonical sparse preparation, canonical-state support in phase export, and an isolated 054 NLL/modeling bundle.
- Scope: Llama2-7B-chat prefill-only only; GPU evaluation is intentionally not launched in this implementation step.

## 2026-07-17 - Canonical sparse exporter and 054 bundle
- Added a pre-quant sparse-NVFP4 preparation option: SparseGPT produces pairwise-4:8 BF16 weights, while final NVFP4 conversion is deferred to phase export.
- Extended the vLLM phase exporter with canonical sparse state arguments. Sparse phases now select calibrated module weights and pack with `prune=False`; combining canonical states and `--prune` is rejected.
- Added the isolated 054 bootstrap, canonical materialization/validation, actual sparse-wrapper local error collection, restartable 72-policy NLL dispatcher, merge, and unchanged softplus quality-proxy fitting scripts.
- The new bundle has not run GPU compression or NLL evaluation yet; all output directories remain isolated from paper artifacts.

## 2026-07-17 - Multi-process extension prewarm
- Diagnosed stalled six-GPU sparse exports as concurrent `torch.utils.cpp_extension.load()` calls contending on the shared CUTLASS extension build lock.
- Added a vLLM-environment single-process prewarm script. It loads sparse BF16, sparse NVFP4, and dense NVFP4 converter extensions before any parallel phase exports.
- Future multi-GPU NLL dispatches must run this prewarm step first; completed 054 results remain reusable.

## 2026-07-17 - Canonical sparse NLL rebuild completed
- Generated and exhaustively validated 224/224 linear weights for both canonical sparse states.
- Collected actual wrapper local errors for 224 sparse-BF16 and 224 sparse-NVFP4 modules, then rebuilt the 672-row method-module feature table.
- Re-ran all 72 fixed policies with canonical phase export and real vLLM NLL; all completed with zero failures.
- Unchanged softplus proxy achieved holdout MAE 0.089391, RMSE 0.104758, Spearman 0.7523, versus the invalid direct-prune 053 holdout MAE 0.425113 and RMSE 0.545739.
