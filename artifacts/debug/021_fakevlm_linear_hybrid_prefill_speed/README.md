# FakeVLM Linear Hybrid Prefill Speed

Debug-only FakeVLM prefill speed experiment for per-linear hybrid backend selection.

## Scope

- Scenario: prefill-only, measured with `model(**inputs, use_cache=False)`.
- Selection paths: `manual_profile` and `latency_model`.
- Default batch sizes: `1 2 4 8 16`.
- Default GPUs: physical cards `0,1` only.
- Conda env: `cospaq`.

## Run

Smoke:

```bash
SAMPLE_LIMIT=2 BATCH_SIZES=1 WARMUP=1 ITERS=2 bash artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/code/run_prefill_speed.sh
```

Full default:

```bash
bash artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/code/run_prefill_speed.sh
```

## Outputs

- `configs/`: per-batch run configs.
- `candidates/`: candidate backend latency tables.
- `policies/`: per-linear policies for manual and latency-model selection.
- `speed/prefill_speed.csv`: measured FakeVLM prefill forward latency.
- `summary/prefill_speed_summary.csv`: best uniform baseline and hybrid speedups.
- `summary/prefill_speed_summary.md`: compact Markdown summary.
- `ANALYSIS.md`: full results, policy composition, and interpretation.

## TODO

Prefill-decode is intentionally out of scope for this run. See `TODO_prefill_decode.md`.
