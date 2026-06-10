# Llama2 Prefill-Only Pareto Analysis

## Inputs

- Candidate rows: 1120
- Pareto budget points: 12
- Unique frontier points: 11

## Method Cost Summary

- dense_bf16: latency_sum_ms=905.191118, quality_sum=0.000000, latency_gain_vs_dense=0.000000
- dense_nvfp4: latency_sum_ms=598.606849, quality_sum=16.530083, latency_gain_vs_dense=306.584269
- sparse_bf16: latency_sum_ms=477.013607, quality_sum=191.284474, latency_gain_vs_dense=428.177511
- sparse_nvfp4: latency_sum_ms=538.899970, quality_sum=379.264602, latency_gain_vs_dense=366.291148
- marlin_nvfp4: latency_sum_ms=913.749609, quality_sum=16.530083, latency_gain_vs_dense=-8.558492

## Frontier Endpoints

- Conservative endpoint: quality=0.000000, latency_ms=905.191118, speedup=1.000000
- Speed endpoint: quality=249.354585, latency_ms=419.982644, speedup=2.155306

## Policy Explanation

- point 5->6: 50 mlp modules dense_bf16->dense_nvfp4, latency_delta_ms=-108.423425, quality_delta=5.460205
- point 4->5: 21 mlp modules dense_bf16->dense_nvfp4, latency_delta_ms=-69.353139, quality_delta=1.466757
- point 6->7: 45 attention modules dense_bf16->sparse_bf16, latency_delta_ms=-50.446654, quality_delta=10.614141
- point 8->9: 56 attention modules dense_nvfp4->sparse_bf16, latency_delta_ms=-39.100861, quality_delta=61.562729
- point 7->8: 19 mlp modules dense_nvfp4->sparse_bf16, latency_delta_ms=-35.593475, quality_delta=24.943686
- point 3->4: 10 mlp modules dense_bf16->dense_nvfp4, latency_delta_ms=-33.082112, quality_delta=0.592909
- point 9->10: 59 mlp modules dense_nvfp4->sparse_nvfp4, latency_delta_ms=-26.949738, quality_delta=109.213596
- point 2->3: 7 mlp modules dense_bf16->dense_nvfp4, latency_delta_ms=-23.157478, quality_delta=0.550388
