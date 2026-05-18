# MaxViT NVFP4 4/6 Fake Quant Plan

## Goal

Enable MaxViT tiny/small/base/large to run the same two weight fake-quant 4/6 variants already used by DINOv3:

- `nvfp4_4over6_unstructured_sparse`
- `nvfp4_4over6_semi_structured_sparse`

The implementation should reuse the existing generic compression checkpoint, accuracy, and speed scripts, while keeping normal NVFP4 methods on `static_6`.

## Plan

1. Remove the DINO-only guard for `nvfp4_4over6_*` in the generic checkpoint preparation script.
2. Add a MaxViT 4/6 prepare Slurm wrapper that runs both methods for one or more variants.
3. Add MaxViT 4/6 accuracy and speed Slurm wrappers that write to variant-specific 4/6 result directories.
4. Document all MaxViT 4/6 test commands for tiny/small/base/large.
5. Run lightweight syntax checks.

## Notes

- This plan covers weight fake quantization only. MaxViT activation fake quant is not added here.
- Existing non-4/6 methods keep `static_6` as their default scale rule.
