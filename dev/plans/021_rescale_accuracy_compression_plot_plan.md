# Rescale Accuracy/Compression Plot Plan

## Goal

Create a presentation-ready figure summarizing accuracy and compression ratio for the original NVFP4 sparse methods versus the rescale method.

## Selection Rule

- Original non-rescale NVFP4 methods: choose worst seeded Top-1 accuracy.
- Rescale methods: choose best seeded Top-1 accuracy with activation quantization off.
- Labels should use `Rescale`, not `four over six`.

## Plan

1. Collect results for MaxViT tiny/small/base/large and DINOv3.
2. Build two panels: unstructured and 2:4 structured.
3. Plot Top-1 accuracy bars for original worst versus rescale best.
4. Add dense baseline markers, compression-ratio labels, and delta annotations.
5. Save CSV, PNG, and PDF outputs under `artifacts/results/`.

## Output

- `artifacts/results/rescale_accuracy_compression_summary.csv`
- `artifacts/results/rescale_accuracy_compression_summary.png`
- `artifacts/results/rescale_accuracy_compression_summary.pdf`
