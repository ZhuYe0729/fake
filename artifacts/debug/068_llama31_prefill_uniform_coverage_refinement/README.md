# Llama3.1 prefill-only uniform-coverage refinement

This debug experiment refines the measured Pareto closure between the warmed
061 `point_007` and `point_009` policies. It does not modify 058, 060, or 061.

The primary target is the final 061 uniform dense-NVFP4 reference:

- measured speedup: `1.7573628343x`;
- ARC-Challenge `acc_norm`: `0.5110921502`.

All candidates use the unchanged 058 quality model, 061 warmed speed protocol,
057 canonical sparse states, and the real `phase_hetero_mytest` vLLM runtime.
Downstream labels are used only for validation, never for constructing a
policy.

Bootstrap the immutable seed set:

```bash
python scripts/bootstrap.py
```

Measure one seed with the final warmed speed protocol:

```bash
/home/agent/wja/miniconda3/envs/cospaq/bin/python \
  scripts/run_speed_policy.py --policy bridge_dense_nvfp4_120 --gpu 1
```

Outputs are written under `llama31_8b_instruct/`. Checkpoints under
`temporary/` are deleted after each run.

Generate the fine-grained policies and rebuild the report:

```bash
python scripts/solve_fine_pareto.py
python scripts/solve_fine_pareto.py \
  --start 42 --stop 78 --stride 4 \
  --prefix low_q --output predicted_low_candidates.csv
python scripts/build_midrange_bridges.py
MPLCONFIGDIR=/tmp/cospaq-matplotlib python scripts/build_report.py
```

The measured winner for the pre-selected ARC-Challenge plot is `fine_q208`:

- E2E speedup: `1.79554x` (uniform dense-NVFP4: `1.75736x`);
- ARC-Challenge `acc_norm`: `0.52048` (uniform dense-NVFP4: `0.51109`).

Thus it strictly dominates dense-NVFP4 on this plot and the complete ours
frontier covers every uniform reference. The cross-task check also covers
dense-NVFP4 on WinoGrande and ARC-Easy, but not on WikiText perplexity or
MMLU. See `llama31_8b_instruct/report/summary.md` and
`all_metric_coverage.csv`; no unsuccessful candidate was removed.

The paper-facing prefill-only curve uses only quality results remeasured under
the current 068 runtime. In particular, the old 058 bridge task scores are not
mixed with new speeds: bridge-120 was remeasured as `1.73379x / 0.52986`, and
the low-budget closure adds non-dominated points at 1.252x, 1.286x, 1.437x,
and 1.504x.
