# FakeVLM Sample Size Accuracy Analysis

Determines the minimum number of test samples needed to reliably estimate true accuracy.

## Approach

Statistical subsampling of the full 5000-sample predictions from `020_fakevlm_uniform_accuracy`. Randomly subsampling N predictions is mathematically equivalent to running the model on N randomly selected test samples (model is deterministic with fixed seed).

## Usage

```bash
conda run -n cospaq python artifacts/debug/024_fakevlm_sample_size_analysis/analyze_sample_size.py
```

## Outputs

- `outputs/<method>/sample_size_accuracy.csv` — raw per-seed results
- `outputs/<method>/sample_size_stats.csv` — aggregated statistics
- `outputs/<method>/accuracy_vs_samples.png` — convergence plot with confidence bands
- `summary/sample_size_summary.csv` — threshold table across methods
- `summary/sample_size_report.md` — human-readable report with recommendations
- `summary/all_methods_comparison.png` — all methods overlaid
- `summary/error_vs_samples.png` — error vs N for all methods