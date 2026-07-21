# 063 validation report

Validation date: 2026-07-21. This report separates fresh execution from retained-result audit and static command checks. Smoke numbers below are diagnostic only and must not enter paper tables.

## Environment and preflight — fresh

- `cospaq`: Python 3.12.13, PyTorch 2.9.0+cu128, Transformers 5.9.0.
- `vllm`: Python 3.12.13, PyTorch 2.9.0+cu128, vLLM `0.11.1.dev0+gb8b302cde.d20260624`, Transformers 4.55.2.
- Hardware: 8 × NVIDIA GeForce RTX 5090, compute capability 12.0, 32607 MiB each.
- Model config: Llama, 32 layers, hidden size 4096, intermediate size 11008, 32 attention/KV heads.
- Patched runtime source contains `phase_hetero_mytest`, dense/sparse NVFP4, sparse BF16 and Marlin NVFP4 backends.
- Free disk at preflight: 1311 GiB.
- Machine-readable reports: `validation/preflight_no_gpu.json` and `validation/preflight_gpu.json`.

## Bootstrap — fresh

`bootstrap_repro.py` created a new run tree without copying measurements, fitted coefficients, or checkpoints. Both scenarios contain 72 policies. The fixed WikiText sample tensors, architecture manifest and policy files were copied with hashes; both copied manifests were rewritten to the new run root and all 144 policy paths were resolved successfully.

Machine-readable reports: `runs/llama2_7b_chat/bootstrap_provenance.json` and `validation/bootstrap_audit.json`.

## Canonical sparse — retained state, freshly audited

The expensive 27-GB SparseGPT states were not regenerated. The existing 054 states were reused only after manifest and size audit:

| state | bytes | SHA-256 |
|---|---:|---|
| sparse BF16 | 13,476,922,367 | `40da8456708d38b67b9bc5a6cdf488e1607d18295945d8e9e1edfc0d7e7b5672` |
| sparse NVFP4 prequant | 13,476,922,367 | `2d43771466bcbdcb64fdb79ff074e08fe5cceb5b3587090a089754bcc6715f85` |

The provenance semantics are SparseGPT 2:4 BF16 and SparseGPT pairwise 4:8 with one final phase-export NVFP4 conversion. `validation/canonical_reuse_audit.json` records the fresh audit.

## Four-policy phase-vLLM closure — fresh

p00/p01/p02/p71 were freshly exported from the clean policy tree. Every checkpoint passed exact policy/provenance verification (`prune=false`, both canonical state paths recorded). For each point, two 2048-token prefill-only blocks and two B=8/S=2048/O=64 teacher-forced blocks were executed, followed by 1 warmup + 2 measured prefill-decode smoke repeats.

| policy | coverage | prefill-only avg NLL | prefill-decode avg NLL | smoke E2E median ms |
|---|---|---:|---:|---:|
| p00 | dense BF16 | 2.064677 | 1.497777 | 2298.713 |
| p01 | dense NVFP4 | 2.101806 | 1.512239 | 2963.432 |
| p02 | canonical sparse BF16 | 2.337825 | 1.823407 | 2055.528 |
| p71 | comprehensive mixed phase/method pairs | 2.144525 | 1.699761 | 3854.742 |

All decode traces recorded `enter_decode=1`, `apply_decode=8064`, and `prepare_next_prefill=1`. The speed column is not a performance result: four cards ran concurrently, only two measurements were used, and extension initialization was intentionally exercised. It verifies successful runner completion and output schema only.

Machine-readable report: `validation/smoke_audit.json`; raw results/logs: `runs/llama2_7b_chat/validation/`.

### Fault found during fresh validation

p01 and p71 initially stayed at zero GPU utilization. The cause was a zero-byte CUTLASS extension lock left since 2026-07-20 17:37, while no `ninja`, `nvcc`, or `c++` process existed and the compiled `.so` was already present. After removing only that confirmed stale lock, both exporters resumed and all stages completed. The preflight now reports extension-lock age and fails for a lock older than 30 minutes; the runbook requires confirming that no compiler owns it before removal.

## Tiny downstream runners — fresh

- Prefill-only: dense-BF16 p00, ARC-Easy `limit=2`, produced a normal lm-eval result JSON (`acc` and `acc_norm` present).
- Prefill-decode: dense-BF16 p00, DialogSum examples `[0,2)`, produced exactly two PMPD generation JSONL rows with references and choices.

These validate offline dataset discovery, the lm-eval vLLM adapter, the PMPD generation runner and output persistence. Their two-example metrics have no scientific meaning.

## Full experiment closure — retained artifacts freshly audited

The following completed historical results were not rerun, but their required files and row counts were freshly audited:

- 054 prefill-only NLL: 72/72; local-error table, quality metrics and measured paper table present.
- 056 prefill-decode NLL: 72/72; coverage-holdout quality metrics present.
- 056 measured closure: 10/10 policies.
- 056 generation summary: 24/24 policy-dataset rows.
- 060 Llama2 prefill-only and prefill-decode complete result tables present.

Machine-readable report: `validation/retained_audit.json`.

## Statically checked, not fully rerun

The following were syntax/CLI/path checked and cross-checked against retained outputs, but intentionally not recomputed due cost:

- full SparseGPT canonical generation;
- all local-error collection jobs;
- both 72-policy, 100-block NLL sweeps;
- quality-model fitting, kernel profiling and both solvers;
- 1+5 exclusive formal speed runs for every displayed point;
- complete five-task prefill-only and three-dataset PMPD evaluation;
- final new-run paper aggregation.

Python compilation and Bash syntax checks passed for all new scripts and modified entrypoints. The modified 054/056 scenario paths, model path, vLLM root, canonical directory and both interpreter paths were exercised through environment-variable overrides. This is sufficient to establish that the documented stages connect correctly on the current platform while retaining a clear boundary around results that were not freshly regenerated.

