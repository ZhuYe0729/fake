# Qwen3.5-9B Linear Kernel Breakdown Debug Plan

## Summary
- Debug Qwen3.5-9B `language_model.layers.0.mlp.down_proj` under `batch_size=1,input_tokens=16384,output_tokens=32`.
- Compare full single-layer timing for `sparse_bf16` and `dense_nvfp4` prefill + `marlin_nvfp4` decode.
- Store scripts, results, and summary under `artifacts/debug/001_qwen35_linear_breakdown/`.

## Implementation
- Add a standalone debug script that loads the real Qwen3.5-9B model, clones the target linear layer, and times build/materialization/forward phases with CUDA synchronization.
- Report both steady runtime and first-call/lazy costs so kernel-only timing can be compared with E2E behavior.
- Save machine-readable JSON/CSV and a README summary.

## Verification
- Compile the debug script with `python -m py_compile`.
- Run the script on GPU and verify the target shape is `n=4096,k=12288`, outputs are finite, and all result files are generated.

## Assumptions
- The target layer is fixed to `language_model.layers.0.mlp.down_proj`.
- The task is debug-only and does not change predictor or production hybrid policy.
