# Llama3.1-8B-Instruct canonical prefill-only Pareto

This is the canonical prefill-only closure for batch 8 and input length 2048.
It deliberately reuses, rather than recreates, the validated Llama3 canonical
SparseGPT states and prefill local-error measurements in 057. It also reuses
the Llama3 kernel-level action support from 038, but remeasures final E2E
phase-runtime anchors.

Run order (all commands are resumable):

```bash
conda activate vllm
python scripts/bootstrap.py
python scripts/verify_reused_assets.py
python scripts/build_speed_design.py
python scripts/run_speed_anchors.py --gpus 1,2,3,4
python scripts/fit_speed_calibrator.py
python scripts/run_nll_shards.py --gpus 1,2,3,4,5,6,7
python scripts/merge_nll.py
python scripts/fit_quality.py
python scripts/solve_pareto.py
```

The final closure and task scripts are enabled only after the two proxy reports
have been inspected. Uniform and mixed policies always use the same
`phase_hetero_mytest` runtime. `--prune` is never accepted in this experiment.
