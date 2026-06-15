# Llama2 Normal-01 Pareto Analysis

Scenario: batch_size=1, input_tokens=16384, output_tokens=32 (prefill M=16384, decode M=1)

## Inputs

- Candidate rows: 1344
- Pareto budget points: 9
- Unique frontier points: 8

## Method Cost Summary

| Method | latency_sum_ms | quality_sum | prefill_sum_ms | decode_sum_ms | conv_sum_ms | gain_vs_dense |
|--------|---------------|-------------|----------------|---------------|-------------|---------------|
| dense_bf16 | 1383.66 | 0.0000 | 984.68 | 12.4682 | 0.0000 | 0.00 |
| dense_nvfp4 | 1612.00 | 16.5301 | 606.02 | 31.1866 | 8.0041 | -228.33 |
| sparse_bf16 | 885.92 | 191.2845 | 486.94 | 12.4682 | 0.0000 | 497.74 |
| sparse_nvfp4 | 941.46 | 379.2646 | 542.48 | 12.4682 | 0.0000 | 442.20 |
| marlin_nvfp4 | 1314.35 | 16.5301 | 938.82 | 8.2450 | 111.6884 | 69.31 |
| dense_nvfp4_prefill_marlin_decode | 981.55 | 16.5301 | 606.02 | 8.2450 | 111.6884 | 402.11 |

## Frontier Endpoints

- **Conservative**: quality=0.000000, latency_ms=1383.66, speedup=1.0000, methods={'dense_bf16': 224}

- **Speed**: quality=136.544546, latency_ms=814.71, speedup=1.6984, methods={'sparse_bf16': 160, 'dense_nvfp4_prefill_marlin_decode': 64}

## Frontier Progression

- Point 0: quality=0.0000, latency=1383.66ms, speedup=1.0000, prefill=984.7ms, decode=399.0ms, conv=0.0ms, counts={'dense_bf16': 224}
- Point 1: quality=1.6085, latency=1284.64ms, speedup=1.0771, prefill=902.1ms, decode=370.7ms, conv=11.8ms, counts={'dense_bf16': 203, 'dense_nvfp4_prefill_marlin_decode': 21}
- Point 2: quality=3.1520, latency=1185.61ms, speedup=1.1670, prefill=819.6ms, decode=342.4ms, conv=23.7ms, counts={'dense_bf16': 182, 'dense_nvfp4_prefill_marlin_decode': 42}
- Point 3: quality=7.3508, latency=1019.47ms, speedup=1.3572, prefill=683.2ms, decode=290.7ms, conv=45.6ms, counts={'dense_bf16': 139, 'sparse_bf16': 4, 'dense_nvfp4_prefill_marlin_decode': 81}
- Point 4: quality=17.9369, latency=934.02ms, speedup=1.4814, prefill=605.8ms, decode=276.4ms, conv=51.8ms, counts={'dense_bf16': 82, 'sparse_bf16': 50, 'dense_nvfp4_prefill_marlin_decode': 92}
- Point 5: quality=45.0282, latency=883.92ms, speedup=1.5654, prefill=553.8ms, decode=276.1ms, conv=54.1ms, counts={'dense_bf16': 35, 'sparse_bf16': 92, 'dense_nvfp4_prefill_marlin_decode': 97}
- Point 6: quality=106.0825, latency=831.80ms, speedup=1.6635, prefill=497.7ms, decode=286.8ms, conv=47.3ms, counts={'dense_bf16': 1, 'sparse_bf16': 139, 'dense_nvfp4_prefill_marlin_decode': 84}
- Point 7: quality=136.5445, latency=814.71ms, speedup=1.6984, prefill=465.9ms, decode=312.7ms, conv=36.0ms, counts={'sparse_bf16': 160, 'dense_nvfp4_prefill_marlin_decode': 64}
