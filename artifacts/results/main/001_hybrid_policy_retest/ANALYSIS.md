# Main Hybrid Policy Retest Analysis

## Setup

- Models: Llama-2-7B, Llama-3.1-8B, Qwen3.5-9B.
- Scenarios:
  - `prefill_only`: `batch_size=16,input_tokens=1024,output_tokens=0`.
  - `normal_01`: `batch_size=1,input_tokens=16384,output_tokens=32`.
- Method families:
  - `single/*`: all compressible linear layers use one backend.
  - `manual`: per shape, benchmark 6 linear-module candidates and select the lowest measured linear latency.
  - `pred`: same candidate set, but select using the kernel latency predictor.
- `single/marlin_nvfp4` is the W4A16 Marlin path.

## Full-Model E2E Ranking

### prefill_only

| Model | Best | manual | pred | dense_bf16 | sparse_bf16 | dense_nvfp4 | sparse_nvfp4 | marlin_nvfp4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-2-7B | pred 756.917 | 791.136 | 756.917 | 1174.772 | 803.297 | 853.286 | 791.817 | 1185.490 |
| Llama-3.1-8B | manual 817.852 | 817.852 | 827.817 | 1289.500 | 933.767 | 943.967 | 879.495 | 1294.634 |
| Qwen3.5-9B | manual 1395.389 | 1395.389 | 1400.313 | 1855.521 | 1563.715 | 1542.141 | 1476.376 | 1873.199 |

### normal_01

| Model | Best | manual | pred | dense_bf16 | sparse_bf16 | dense_nvfp4 | sparse_nvfp4 | marlin_nvfp4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-2-7B | pred 2071.787 | 2102.645 | 2071.787 | 2457.865 | 2429.384 | 3338.795 | 3477.703 | 2273.406 |
| Llama-3.1-8B | manual 2058.601 | 2058.601 | 2074.224 | 2256.926 | 2098.866 | 2970.650 | 3258.166 | 2347.541 |
| Qwen3.5-9B | sparse_bf16 3646.444 | 4101.563 | 3682.884 | 4136.767 | 3646.444 | 5074.886 | 5452.643 | 4296.811 |

## Observations

- Manual/pred E2E now applies the generated policy. `*_full_e2e.csv` contains non-empty `replaced_linear_count` and `backend_counts` for both families.
- Predictor policy is strong in Llama normal_01 and Qwen normal_01, but Qwen normal_01 still slightly trails all-sparse-bf16 in full E2E.
- Linear-module aggregate latency and full-model E2E do not always rank methods identically. Qwen normal_01 is the clearest case: manual shape-level selection has reasonable linear totals, but full-model E2E is worse than both pred and single sparse_bf16.
- For prefill_only, manual/pred policies are consistently better than dense_bf16 and close to or better than the best single compressed backend.

## Files

- `comparison/full_e2e_summary.csv`: all 42 full-model E2E rows.
- `comparison/linear_latency_summary.csv`: total linear latency rows.
- `comparison/manual_vs_pred_policy_diff.csv`: per-linear policy differences.
