# Checkpoint File Compression Report Plan

## Goal

Generate a separate report for real file-size compression ratios based on existing checkpoint files, distinct from metadata-estimated compression ratios used in plots.

## Plan

1. Compute dense source bytes for MaxViT variants and DINOv3.
2. Scan existing `artifacts/checkpoints/**/model.pt` files.
3. Compute `dense_source_bytes / checkpoint_bytes` for each checkpoint file.
4. Mark checkpoint type clearly:
   - real packed/runtime/storage checkpoint
   - fake dense state_dict checkpoint
5. Write both CSV and Markdown reports under `artifacts/results/`.

## Notes

- Rescale candidates currently do not have real packed storage checkpoints; their existing `model.pt` files are dense fake-quant state_dicts, so file-size ratio is close to 1x and should not be used as real storage compression.
- The real storage-compressed ratios are meaningful for existing CUTLASS packed checkpoint paths.
