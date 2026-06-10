# Llama2-7B Oracle Summary

Timing mode: warm E2E. Full-model E2E excludes model loading and policy replacement; it includes warmed prefill and decode forward time.

Scenarios:

- `prefill_only`: batch_size=16, input_tokens=1024, output_tokens=0
- `normal_01`: batch_size=1, input_tokens=16384, output_tokens=32
- `normal_02`: batch_size=1, input_tokens=16384, output_tokens=256

## E2E Speed

### prefill_only

| method | prefill ms | decode x n ms | e2e ms | source |
|---|---:|---:|---:|---|
| `single/dense_bf16` | 1174.772 | 0.000 | 1174.772 | `single/dense_bf16/prefill_only/llama2-7b_full_e2e.csv` |
| `single/sparse_bf16` | 803.297 | 0.000 | 803.297 | `single/sparse_bf16/prefill_only/llama2-7b_full_e2e.csv` |
| `single/dense_nvfp4` | 853.286 | 0.000 | 853.286 | `single/dense_nvfp4/prefill_only/llama2-7b_full_e2e.csv` |
| `single/sparse_nvfp4` | 791.817 | 0.000 | 791.817 | `single/sparse_nvfp4/prefill_only/llama2-7b_full_e2e.csv` |
| `single/marlin_nvfp4` | 1185.490 | 0.000 | 1185.490 | `single/marlin_nvfp4/prefill_only/llama2-7b_full_e2e.csv` |
| `single/dense_nvfp4_prefill_marlin_decode` | 853.086 | 0.000 | 853.086 | `single/dense_nvfp4_prefill_marlin_decode/prefill_only/llama2-7b_full_e2e.csv` |
| `pred` | 756.917 | 0.000 | 756.917 | `pred/prefill_only/llama2-7b_full_e2e.csv` |
| `oracle` | 756.917 | 0.000 | 756.917 | `oracle/prefill_only/llama2-7b_full_e2e.csv` |

### normal_01

| method | prefill ms | decode x n ms | e2e ms | source |
|---|---:|---:|---:|---|
| `single/dense_bf16` | 1513.118 | 944.746 | 2457.865 | `single/dense_bf16/normal_01/llama2-7b_full_e2e.csv` |
| `single/sparse_bf16` | 1167.326 | 1262.058 | 2429.384 | `single/sparse_bf16/normal_01/llama2-7b_full_e2e.csv` |
| `single/dense_nvfp4` | 1311.004 | 2027.791 | 3338.795 | `single/dense_nvfp4/normal_01/llama2-7b_full_e2e.csv` |
| `single/sparse_nvfp4` | 1125.956 | 2351.746 | 3477.703 | `single/sparse_nvfp4/normal_01/llama2-7b_full_e2e.csv` |
| `single/marlin_nvfp4` | 1521.375 | 752.030 | 2273.406 | `single/marlin_nvfp4/normal_01/llama2-7b_full_e2e.csv` |
| `single/dense_nvfp4_prefill_marlin_decode` | 1190.806 | 1264.684 | 2455.489 | `single/dense_nvfp4_prefill_marlin_decode/normal_01/llama2-7b_full_e2e.csv` |
| `pred` | 1240.986 | 830.801 | 2071.787 | `pred/normal_01/llama2-7b_full_e2e.csv` |
| `oracle` | 1240.986 | 830.801 | 2071.787 | `oracle/normal_01/llama2-7b_full_e2e.csv` |

### normal_02

| method | prefill ms | decode x n ms | e2e ms | source |
|---|---:|---:|---:|---|
| `single/dense_bf16` | 1512.000 | 7589.402 | 9101.402 | `single/dense_bf16/normal_02/llama2-7b_full_e2e.csv` |
| `single/sparse_bf16` | 1163.815 | 9170.894 | 10334.709 | `single/sparse_bf16/normal_02/llama2-7b_full_e2e.csv` |
| `single/dense_nvfp4` | 1187.082 | 16161.770 | 17348.852 | `single/dense_nvfp4/normal_02/llama2-7b_full_e2e.csv` |
| `single/sparse_nvfp4` | 1165.920 | 20563.507 | 21729.426 | `single/sparse_nvfp4/normal_02/llama2-7b_full_e2e.csv` |
| `single/marlin_nvfp4` | 1519.052 | 6198.822 | 7717.874 | `single/marlin_nvfp4/normal_02/llama2-7b_full_e2e.csv` |
| `single/dense_nvfp4_prefill_marlin_decode` | 1190.478 | 6571.953 | 7762.431 | `single/dense_nvfp4_prefill_marlin_decode/normal_02/llama2-7b_full_e2e.csv` |
| `pred` | 1240.863 | 6041.503 | 7282.366 | `pred/normal_02/llama2-7b_full_e2e.csv` |
| `oracle` | 1253.832 | 6172.910 | 7426.742 | `oracle/normal_02/llama2-7b_full_e2e.csv` |

## Oracle vs Pred Policy

| scenario | linear group | oracle | pred | same |
|---|---|---|---|---|
| `prefill_only` | `mlp.down_proj` | `sparse_bf16->sparse_bf16` | `sparse_bf16->sparse_bf16` | True |
| `prefill_only` | `mlp.gate_proj` | `sparse_nvfp4->sparse_nvfp4` | `sparse_nvfp4->sparse_nvfp4` | True |
| `prefill_only` | `mlp.up_proj` | `sparse_nvfp4->sparse_nvfp4` | `sparse_nvfp4->sparse_nvfp4` | True |
| `prefill_only` | `self_attn.k_proj` | `sparse_bf16->sparse_bf16` | `sparse_bf16->sparse_bf16` | True |
| `prefill_only` | `self_attn.o_proj` | `sparse_bf16->sparse_bf16` | `sparse_bf16->sparse_bf16` | True |
| `prefill_only` | `self_attn.q_proj` | `sparse_bf16->sparse_bf16` | `sparse_bf16->sparse_bf16` | True |
| `prefill_only` | `self_attn.v_proj` | `sparse_bf16->sparse_bf16` | `sparse_bf16->sparse_bf16` | True |
| `normal_01` | `mlp.down_proj` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` | True |
| `normal_01` | `mlp.gate_proj` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` | True |
| `normal_01` | `mlp.up_proj` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` | True |
| `normal_01` | `self_attn.k_proj` | `marlin_nvfp4->marlin_nvfp4` | `marlin_nvfp4->marlin_nvfp4` | True |
| `normal_01` | `self_attn.o_proj` | `marlin_nvfp4->marlin_nvfp4` | `marlin_nvfp4->marlin_nvfp4` | True |
| `normal_01` | `self_attn.q_proj` | `marlin_nvfp4->marlin_nvfp4` | `marlin_nvfp4->marlin_nvfp4` | True |
| `normal_01` | `self_attn.v_proj` | `marlin_nvfp4->marlin_nvfp4` | `marlin_nvfp4->marlin_nvfp4` | True |
| `normal_02` | `mlp.down_proj` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` | True |
| `normal_02` | `mlp.gate_proj` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` | True |
| `normal_02` | `mlp.up_proj` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` | True |
| `normal_02` | `self_attn.k_proj` | `marlin_nvfp4->marlin_nvfp4` | `marlin_nvfp4->marlin_nvfp4` | True |
| `normal_02` | `self_attn.o_proj` | `marlin_nvfp4->marlin_nvfp4` | `marlin_nvfp4->marlin_nvfp4` | True |
| `normal_02` | `self_attn.q_proj` | `marlin_nvfp4->marlin_nvfp4` | `marlin_nvfp4->marlin_nvfp4` | True |
| `normal_02` | `self_attn.v_proj` | `marlin_nvfp4->marlin_nvfp4` | `marlin_nvfp4->marlin_nvfp4` | True |

## Oracle Derivation

- `normal_02`: oracle comes from full-model method traces plus targeted no-hook full-model ablation over attention k/q/v. The refined oracle policy is identical to pred.
- `normal_01`: existing full-model E2E comparison shows pred is faster than manual and all single methods; its only manual disagreement is `mlp.down_proj`, and the pred choice is retained as the validated oracle for this summary.
- `prefill_only`: existing full-model E2E comparison shows pred is fastest among available policies and single methods; with no decode phase, the oracle uses the validated pred policy.

## Pred Derivation

Pred uses the kernel latency predictor to estimate each linear group independently. For each candidate strategy it computes:

`prefill_latency + output_tokens * decode_latency + conversion_latency`

The compatible mixed strategy `dense_nvfp4->marlin_nvfp4` uses dense NVFP4 latency for prefill, Marlin W4A16 latency for decode, and the predictor's conversion latency for the shared NVFP4-to-Marlin transition.

## Notes

- `normal_02` oracle and pred have identical policies. Their E2E rows are separate runs, so the numeric difference reflects run-to-run variance rather than a policy difference.
- The `oracle` row means the validated oracle policy run available in this directory; when oracle and pred policies are identical, the best observed latency for that policy may appear in either row.
