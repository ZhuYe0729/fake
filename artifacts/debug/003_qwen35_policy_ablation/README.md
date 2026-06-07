# Qwen3.5-9B Policy Ablation

This debug experiment compares full-model E2E latency for small policy edits around the Qwen3.5-9B `normal_01` manual/pred difference.

- `scripts/qwen35_policy_ablation.py`: builds manual/pred/sparse and swap variants, then runs repeated full-model E2E.
- `results/`: generated policy variants and timing CSVs.
- `ANALYSIS.md`: interpretation of the ablation result.
