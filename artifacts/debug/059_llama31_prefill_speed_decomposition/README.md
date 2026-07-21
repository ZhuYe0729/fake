# Llama3.1 prefill speed decomposition debug

This debug experiment isolates per-linear latency prediction error from
phase-vLLM E2E composition error for the canonical 058 prefill-only scenario
(batch 8, input 2048). It reuses existing policies and never modifies either
the kernel model artifacts or the 058 experiment outputs.

Run:

```bash
conda run -n cospaq python scripts/profile_exact_shapes.py --gpu 1
conda run -n cospaq python scripts/summarize_decomposition.py
```

`exact_micro/targeted_profile.csv` contains real module-forward timings for
the four Llama3 fused shapes and all five actions. `report/` joins their sums
with 058's independently measured phase-vLLM E2E closure results.
