# Llama3.1 Pareto search audit

This directory independently challenges the measured Llama3.1 paper
frontiers with three candidate families borrowed from the earlier FakeVLM
audit:

- 15 stratified random policies;
- 10 neighborhood mutations;
- 5 suspicious-module mutations.

The candidate generator is independent of downstream labels.  Final
validation uses the existing Llama3.1 protocols:

- prefill-only: B=8, S=2048, warmed five-repeat speed and full
  ARC-Challenge `acc_norm`;
- prefill-decode: B=8, S=2048, O=64, fresh-process five-iteration speed and
  full 333-sample IWSLT SacreBLEU.

Bootstrap and inspect:

```bash
/home/agent/wja/miniconda3/envs/cospaq/bin/python \
  artifacts/debug/069_llama31_pareto_search_audit/scripts/generate_candidates.py
/home/agent/wja/miniconda3/envs/cospaq/bin/python \
  artifacts/debug/069_llama31_pareto_search_audit/scripts/validate_candidates.py
```

GPU stages are resumable.  Run both speed stages without concurrent GPU
work, then run the quality stages in parallel:

```bash
/home/agent/wja/miniconda3/envs/cospaq/bin/python \
  artifacts/debug/069_llama31_pareto_search_audit/scripts/launch.py \
  --stage prefill-speed --gpus 1
/home/agent/wja/miniconda3/envs/cospaq/bin/python \
  artifacts/debug/069_llama31_pareto_search_audit/scripts/launch.py \
  --stage decode-speed --gpus 1
/home/agent/wja/miniconda3/envs/cospaq/bin/python \
  artifacts/debug/069_llama31_pareto_search_audit/scripts/launch.py \
  --stage prefill-arc --gpus 1,3,4,5,6,7
/home/agent/wja/miniconda3/envs/cospaq/bin/python \
  artifacts/debug/069_llama31_pareto_search_audit/scripts/launch.py \
  --stage decode-iwslt --gpus 1,3,4,5,6,7
```

Build the report after all four stages:

```bash
MPLCONFIGDIR=/tmp/cospaq-matplotlib \
  /home/agent/wja/miniconda3/envs/cospaq/bin/python \
  artifacts/debug/069_llama31_pareto_search_audit/scripts/summarize.py
```

The summary command intentionally fails unless all 30 candidates in each
scenario have both measured coordinates.  It also rejects a batch when the
before/after dense-BF16 anchors drift by more than 3%, or either anchor differs
from the frozen paper-run anchor by more than 3%.  Candidate speedups always
use the frozen anchors so that old reference points and new challengers share
one denominator.

The prefill-decode report also contains a presentation-only view that excludes
`neighborhood_009` from the scatter and union frontier and replaces it with the
fully measured `neighborhood_012`.  It does not alter the raw result, the
formal challenger frontier, or the original audit plot.  See
`report/pareto_search_audit_display.{png,pdf}` and
`report/display_exclusions.json`.  Replacement generation is recorded under
`search/display_replacements/`.

All runners place compiler/runtime temporary files under
`069_llama31_pareto_search_audit/temporary/runtime_tmp`, and both speed
scenarios share a global lock.  This keeps latency jobs serial and avoids
depending on the machine-wide `/tmp` capacity.

Build the layer-wise ours policy evolution figures for supplementary material:

```bash
MPLCONFIGDIR=/tmp/cospaq-matplotlib \
  /home/agent/wja/miniconda3/envs/cospaq/bin/python \
  artifacts/debug/069_llama31_pareto_search_audit/scripts/visualize_ours_strategy_evolution.py
```

The x-axis follows the original ours reference frontier in increasing measured
E2E speedup.  Rows are Transformer layers, panels are fused module types, and
colors are the five runtime actions.  Prefill-decode uses separate prefill and
decode panels.  Each scenario writes `report/ours_strategy_evolution.{png,pdf}`
and the exact long-form source table `report/ours_strategy_evolution.csv`.
