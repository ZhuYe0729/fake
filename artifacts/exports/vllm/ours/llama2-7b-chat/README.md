# Llama2-7B-Chat vLLM Ours

`max_speed/` contains one predictor-only layer-heterogeneous policy per
workload. The predictor uses vLLM fused Llama Linear shapes and the trained
models under `fake/kernels/cutlass/cutlass_wrapper/modeling`; it never profiles
per-layer candidate latency during routing.

Scenarios:

- `prefill_only`: policy objective uses `M=8*2048`, zero decode steps; the
  benchmark remains baseline-compatible and requests one output token.
- `prefill_decode`: policy objective uses `M_prefill=16*2048`, `M_decode=16`,
  and 80 decode steps.

The candidate mapping is `marlin_nvfp4` in the predictor to `w4a16_ours` in
the vLLM phase-heterogeneous checkpoint. Outputs retain both names.

Run the full workflow. It uses `cospaq` for predictor/export and the local
`vllm` environment for vLLM benchmark/quality execution:

```bash
bash artifacts/exports/vllm/ours/llama2-7b-chat/scripts/run_max_speed.sh
```

`pareto/` is intentionally TODO. It will consume scenario-specific per-fused-
Linear quality costs and emit the same `phase_hetero_policy.json` format.

For official phase-runtime measurements, use `scripts/run_fresh_process_speed.sh`
for `prefill_decode`: it follows the existing vLLM phase-hetero artifact
convention of one warmup and ten independent processes per output length, and
summarizes only `generate_s`. `prefill_only` instead uses the baseline-aligned
runner because it has no phase transition. For PMPD quality,
use `scripts/run_isolated_pmpd.sh` once per scenario/dataset with its exact
sample count in `QUESTION_END`; it creates one process per four-sample batch.
