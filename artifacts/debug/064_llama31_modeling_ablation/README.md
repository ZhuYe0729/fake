# Llama3.1 modeling ablation

This directory contains the compact modeling ablation used by the paper.  It
is fully offline: all local errors, real-vLLM NLL labels, RTX 5090 E2E
latency labels, and the independent Llama3 exact-shape module profiles are
read from the frozen 057/059 experiments. Kernel-generalization labels come
from the existing CUTLASS benchmark corpus; its split is fixed before fitting.

Experiment:

- model: Llama3.1-8B-Instruct;
- workload: prefill-decode, B=8, input=2048, output=64;
- quality split: fixed feature-coverage 54 train / 18 holdout policies;
- speed split: fixed 7 train / 5 holdout policies;
- kernel split: three seeds, each holding out 25% of complete `(N,K)` groups;
- no GPU evaluation and no mutation of 057 or 060.

Run with the project environment:

```bash
conda run -n wja-cospaq python artifacts/debug/064_llama31_modeling_ablation/scripts/run_all.py
```

The paper-facing outputs are written to `report/`.  The original compact bar
figure is retained as `modeling_ablation.{png,pdf}`; the richer prediction
scatter and incremental-benefit figure is written as
`modeling_ablation_detailed.{png,pdf}`.  Raw per-policy predictions and fitted
metrics are retained under `quality/` and `speed/`. The simplest paper-facing
result is `report/ablation_tables.{md,tex}`.

The full implementation-level design description, formulas, ablation protocol,
and draft-ready paper text are in `report/modeling_design_and_ablation.md`.
