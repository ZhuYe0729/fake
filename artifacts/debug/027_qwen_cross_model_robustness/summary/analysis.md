# Qwen/Llama Cross-Model Robustness Analysis

- Models: `Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B, Llama-2-7B, Llama-3.1-8B`.
- Workloads: `prefill_only, normal_01, normal_02`.
- Main table: `summary/model_workload_method_table.md`.
- Model average table: `summary/model_average_table.md`.
- Transfer table: `summary/cross_model_transfer.md`.
- `our_linear_hybrid` overall geomean: `1.195x`.
- Best transfer strategy in current results: `our_linear_hybrid` at `1.195x`.

Interpretation target:
- Uniform methods should show model-size-dependent winners and weaker transferred averages.
- `our_linear_hybrid` should retain strong average speedup by selecting per-linear prefill/decode backends per model and workload.
