# Qwen3.5-9B E2E Linear Gap Debug

This directory contains scripts/results for comparing standalone linear-module policy selection with full-model Qwen3.5-9B E2E behavior under `normal_01`.

- `scripts/trace_qwen35_policy_gap.py`: loads the real model, applies single/manual/pred policies, measures no-hook E2E, then traces every compressible linear in the real model forward.
- `results/`: generated CSV and markdown outputs.
- `ANALYSIS.md`: summary of the identified mismatch.
