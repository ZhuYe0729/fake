# Llama2 Prefill-Only Pareto Analysis

## Inputs

- Candidate rows: 896
- Pareto budget points: 31
- Unique frontier points: 29

## Method Cost Summary

- dense_bf16: latency_sum_ms=905.191118, quality_sum=0.000000, latency_gain_vs_dense=0.000000
- dense_nvfp4: latency_sum_ms=598.606849, quality_sum=0.321325, latency_gain_vs_dense=306.584269
- sparse_bf16: latency_sum_ms=477.013607, quality_sum=0.530134, latency_gain_vs_dense=428.177511
- sparse_nvfp4: latency_sum_ms=538.899970, quality_sum=1.235227, latency_gain_vs_dense=366.291148

## Frontier Endpoints

- Conservative endpoint: quality=0.000000, latency_ms=905.191118, speedup=1.000000
- Speed endpoint: quality=0.814352, latency_ms=419.982644, speedup=2.155306
