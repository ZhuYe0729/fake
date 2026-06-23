# FakeVLM Pareto Search Audit

Small-scale empirical audit for the FakeVLM prefill Pareto frontier.

This run validates searched mixed policies with:

- Real full-model prefill E2E speed.
- Fixed-random 20% FakeClue accuracy subset.

Reference frontier: `artifacts/debug/024_fakevlm_prefill_global_pareto`.

## Workflow

```bash
python artifacts/debug/025_fakevlm_pareto_search_audit/scripts/make_subset.py
python artifacts/debug/025_fakevlm_pareto_search_audit/scripts/generate_search_policies.py
python artifacts/debug/025_fakevlm_pareto_search_audit/scripts/launch_validation.py --gpus 0,1,2,3,4,5
python artifacts/debug/025_fakevlm_pareto_search_audit/scripts/summarize_search.py
```

Use `--max-policies` on the launcher for smoke tests.

On a Slurm GPU node:

```bash
sbatch artifacts/debug/025_fakevlm_pareto_search_audit/scripts/run_full_validation.sbatch
```

On an already allocated 6-GPU node:

```bash
GPUS=0,1,2,3,4,5 bash artifacts/debug/025_fakevlm_pareto_search_audit/scripts/run_full_validation.sh
```
