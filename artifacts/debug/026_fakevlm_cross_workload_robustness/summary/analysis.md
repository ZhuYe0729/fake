# FakeVLM Cross-Workload Robustness Analysis

- Workloads: `prefill_only, normal_01, normal_02`.
- Main table: `summary/workload_method_table.md`.
- Transfer table: `summary/cross_workload_transfer.md`.
- Best geomean strategy in current results: `our_linear_hybrid` at `1.262x`.

Interpretation target:
- Uniform methods should show workload-dependent winners and weaker transferred averages.
- `our_linear_hybrid` should retain strong average speedup by selecting per-linear prefill/decode backends for each workload.
