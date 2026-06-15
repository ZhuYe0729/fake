# Llama2 Normal-02 Pareto Analysis

Scenario: batch_size=1, input_tokens=16384, output_tokens=256 (prefill M=16384, decode M=1)

## Inputs

- Candidate rows: 1344
- Pareto budget points: 10
- Unique frontier points: 10

## Method Cost Summary

| Method | rows | latency_sum_ms | quality_sum | prefill_sum_ms | decode_sum_ms | conv_sum_ms | gain_vs_dense |
|--------|------|---------------|-------------|----------------|---------------|-------------|---------------|
| dense_bf16 | 224 | 4176.54 | 0.0000 | 984.68 | 12.4682 | 0.0000 | 0.00 |
| dense_nvfp4 | 224 | 8589.79 | 16.5301 | 606.02 | 31.1866 | 0.0000 | -4413.24 |
| sparse_bf16 | 224 | - | - | - | - | - | all rows unsupported |
| sparse_nvfp4 | 224 | - | - | - | - | - | all rows unsupported |
| marlin_nvfp4 | 224 | 3049.53 | 16.5301 | 938.82 | 8.2450 | 0.0000 | 1127.01 |
| dense_nvfp4_prefill_marlin_decode | 224 | 2836.42 | 16.5301 | 606.02 | 8.2450 | 119.6925 | 1340.12 |

## Frontier Endpoints

- **Conservative (dense)**: quality=0.000000, latency_ms=4176.54, speedup=1.0000, methods={'dense_bf16': 224}

- **Speed**: quality=16.530083, latency_ms=2829.12, speedup=1.4763, methods={'marlin_nvfp4': 128, 'dense_nvfp4_prefill_marlin_decode': 96}

## Frontier Progression

- Point 0: quality=0.0000, latency=4176.54ms, speedup=1.0000, prefill=984.7ms, decode=3191.9ms, conv=0.0ms, counts={'dense_bf16': 224}
- Point 1: quality=0.0794, latency=4135.94ms, speedup=1.0098, prefill=974.7ms, decode=3159.5ms, conv=1.8ms, counts={'dense_bf16': 220, 'marlin_nvfp4': 1, 'dense_nvfp4_prefill_marlin_decode': 3}
- Point 2: quality=0.1584, latency=4109.46ms, speedup=1.0163, prefill=968.6ms, decode=3137.8ms, conv=3.0ms, counts={'dense_bf16': 217, 'marlin_nvfp4': 2, 'dense_nvfp4_prefill_marlin_decode': 5}
- Point 3: quality=0.3259, latency=4056.25ms, speedup=1.0297, prefill=954.8ms, decode=3096.0ms, conv=5.4ms, counts={'dense_bf16': 215, 'dense_nvfp4_prefill_marlin_decode': 9}
- Point 4: quality=0.6481, latency=3957.47ms, speedup=1.0554, prefill=927.3ms, decode=3020.6ms, conv=9.6ms, counts={'dense_bf16': 208, 'dense_nvfp4_prefill_marlin_decode': 16}
- Point 5: quality=1.3055, latency=3800.00ms, speedup=1.0991, prefill=882.2ms, decode=2901.6ms, conv=16.2ms, counts={'dense_bf16': 197, 'dense_nvfp4_prefill_marlin_decode': 27}
- Point 6: quality=2.5973, latency=3562.06ms, speedup=1.1725, prefill=818.9ms, decode=2716.8ms, conv=26.4ms, counts={'dense_bf16': 175, 'marlin_nvfp4': 5, 'dense_nvfp4_prefill_marlin_decode': 44}
- Point 7: quality=5.2974, latency=3190.26ms, speedup=1.3092, prefill=718.5ms, decode=2429.2ms, conv=42.6ms, counts={'dense_bf16': 153, 'dense_nvfp4_prefill_marlin_decode': 71}
- Point 8: quality=10.4523, latency=2857.25ms, speedup=1.4617, prefill=663.3ms, decode=2136.4ms, conv=57.5ms, counts={'dense_bf16': 56, 'marlin_nvfp4': 72, 'dense_nvfp4_prefill_marlin_decode': 96}
- Point 9: quality=16.5301, latency=2829.12ms, speedup=1.4763, prefill=660.9ms, decode=2110.7ms, conv=57.5ms, counts={'marlin_nvfp4': 128, 'dense_nvfp4_prefill_marlin_decode': 96}
