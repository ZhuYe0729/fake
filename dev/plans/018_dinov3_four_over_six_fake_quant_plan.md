# DINOv3 NVFP4 4/6 Fake Quant Plan

## Summary
Add two independent DINOv3 ViT-7B/16 fake-quant compression paths using Four Over Six adaptive block scaling:

- `nvfp4_4over6_unstructured_sparse`
- `nvfp4_4over6_semi_structured_sparse`

Use MSE to choose each NVFP4 block's candidate scale (`amax/6` vs `amax/4`), keep existing naive NVFP4/pruning paths unchanged, and save checkpoints/results in separate directories.

## Key Changes
- Add `nvfp4_scale_rule` to fake NVFP4 config and checkpoint CSV metadata.
- Keep existing paths on `static_6`; use `four_over_six_mse` for the new 4/6 methods.
- Reuse existing unstructured pruning and NVFP4 pairwise 4:8 structured pruning before 4/6 fake quant.
- Add DINOv3-specific prepare/eval/bench wrappers and Slurm entries with independent result directories:
  - `artifacts/results/dinov3_vit7b16_4over6_unstructured_sparse/`
  - `artifacts/results/dinov3_vit7b16_4over6_semi_structured_sparse/`
- Update plotting helpers so new DINOv3 4/6 accuracy/speed rows are discoverable.

## Test Plan
- Compile changed Python files with `python -m py_compile`.
- Run CPU-level fake quant checks for shape/dtype, selector counts, and unchanged `static_6` behavior.
- GPU smoke via Slurm for checkpoint preparation and optional accuracy/speed runs.

## Assumptions
- "structured" maps to the existing `nvfp4_semi_structured_sparse` pairwise 4:8 pattern.
- 4/6 applies to weight fake quant only; activation quant kernels and CUTLASS runtime kernels are deferred.
- MaxViT is out of scope for this pass.
