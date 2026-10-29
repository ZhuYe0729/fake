# DASO

DASO selects module-wise and phase-wise compression actions for a target model,
GPU, and serving workload under a quality budget. This directory is the clean,
standalone source tree distilled from the paper experiments; historical debug
directories are not runtime dependencies.

## What is included

- Canonical SparseGPT state preparation and provenance checks.
- Method-, structure-, depth-, and phase-aware quality calibration.
- External calibrated Roofline latency-model adapter and E2E phase composition.
- Budgeted dynamic-programming policy solver.
- Real phase-vLLM NLL, speed, and downstream-task runners.
- Paper configurations, structured measurements, policies, figures, tables,
  ablations, and appendix Pareto-audit data.
- Staging adapters for MaxViT, DINOv3, MIRROR, and FakeVLM.

The patched vLLM runtime and CUTLASS wrapper are deliberately external. Model
weights and datasets are also not redistributed.

## Layout

```text
src/daso/                 stable library: compression, models, solver, reporting
workflows/paper/          executable paper pipeline and runtime launchers
configs/                  portable model, workload, platform, experiment examples
results/paper/            final structured data, policies, figures, tables, hashes
extras/                   staged MaxViT, DINOv3, MIRROR, and FakeVLM support
docs/                     reproduction, backend, provenance, and extension guides
tests/                    CPU tests for the reusable core and workflow invariants
validation/               recorded standalone-copy validation evidence
```

## Quick validation

```bash
python -m pip install -e '.[test,plot]'
pytest -q
python workflows/paper/build_result_manifest.py
python workflows/paper/validate_archive.py
```

To rebuild data-driven paper figures:

```bash
MPLCONFIGDIR=/tmp/daso-mpl bash workflows/paper/rebuild_paper_artifacts.sh
```

For a full GPU reproduction, start with [docs/reproduction.md](docs/reproduction.md).
The exact scope and result limitations are in
[docs/result_provenance.md](docs/result_provenance.md).

This repository rebuilds the data-driven tables and figures.  It does not yet
build the complete paper PDF because the original paper export did not include
the publisher style, bibliography, or manually authored motivation/overview
figures.  See [docs/release_blockers.md](docs/release_blockers.md).
