# Llama2 vLLM Prefill-Only Workload Search

Quality budget uses Pareto point P024: `0.29584014634`.

## Existing measured prefill-only proxy

| scenario | best_single | optimized_vs_best_single | optimized_acc_norm | max_speed_vs_best_single | max_speed_acc_norm |
|---|---|---:|---:|---:|---:|
| b4_in1024_out1 | sparse_nvfp4 | 0.818 | 0.4308873720136519 | 0.882 | 0.431740614334471 |
| b2_in1024_out1 | sparse_nvfp4 | 0.882 | 0.4308873720136519 | 0.925 | 0.431740614334471 |
| b4_in512_out1 | sparse_nvfp4 | 0.847 | 0.4308873720136519 | 0.891 | 0.431740614334471 |
| b2_in4096_out1 | sparse_nvfp4 | 0.830 | 0.4308873720136519 | 0.898 | 0.431740614334471 |
| b1_in4096_out1 | sparse_nvfp4 | 0.825 | 0.4308873720136519 | 0.894 | 0.431740614334471 |
| b8_in512_out1 | sparse_nvfp4 | 0.884 | 0.43686006825938567 | 1.020 | 0.4087030716723549 |

## Top pure-prefill predicted candidates

| scenario | prefill_m | best_uniform | optimized_vs_best_uniform | optimized_counts | max_speed_vs_best_uniform | max_speed_counts |
|---|---:|---|---:|---|---:|---|
| b1_in128_out1 | 128 | marlin_nvfp4 | 1.186026 | dense_bf16:73,sparse_bf16:55 | 1.242473 | dense_bf16:64,sparse_bf16:64 |
| b1_in256_out1 | 256 | sparse_bf16 | 0.978547 | dense_bf16:72,dense_nvfp4:2,sparse_bf16:54 | 1.033766 | dense_bf16:32,sparse_bf16:96 |
| b2_in128_out1 | 256 | sparse_bf16 | 0.978547 | dense_bf16:72,dense_nvfp4:2,sparse_bf16:54 | 1.033766 | dense_bf16:32,sparse_bf16:96 |
| b1_in512_out1 | 512 | sparse_bf16 | 0.963541 | dense_bf16:35,dense_nvfp4:66,sparse_bf16:27 | 1.125332 | sparse_bf16:96,sparse_nvfp4:32 |
| b2_in256_out1 | 512 | sparse_bf16 | 0.963541 | dense_bf16:35,dense_nvfp4:66,sparse_bf16:27 | 1.125332 | sparse_bf16:96,sparse_nvfp4:32 |

## Interpretation

- `optimized` is the P024 quality-constrained fused hetero policy.
- `max_speed` is the unconstrained fastest fused hetero policy and is useful as a speed upper bound, but may have unacceptable accuracy loss.
- Pure-prefill prediction excludes the decode `M=batch` call that vLLM `output_seq=1` still executes.
- Final claims should use the generated focused retest commands and full quality results.

## Generated files

- `prefill_only_prediction_candidates.csv`
- `prefill_only_prediction_details.csv`
- `prefill_only_policy_rows.csv`
- `existing_prefill_only_measured.csv`
- `focused_retest_scenarios.csv`
- `run_focused_retest.sh`
