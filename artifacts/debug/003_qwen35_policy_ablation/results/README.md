# Qwen3.5-9B Policy Ablation

- Scenario: `normal_01` -> `{'batch_size': 1, 'input_tokens': 16384, 'output_tokens': 32}`
- Repeats per variant: `3`

| Variant | E2E mean ms | E2E min/max ms | Prefill mean ms | Decode x n mean ms | Backend counts |
| --- | ---: | ---: | ---: | ---: | --- |
| `pred_down_kv_to_manual` | 4024.2347 | 4009.2530/4045.9041 | 2560.2844 | 1463.9503 | `{'marlin_nvfp4': 104, 'dense_nvfp4/marlin_nvfp4': 96, 'bf16': 48}` |
| `manual_down_to_pred` | 4028.4844 | 4006.6018/4045.0289 | 2527.9702 | 1500.5142 | `{'marlin_nvfp4': 72, 'dense_nvfp4/marlin_nvfp4': 128, 'bf16': 48}` |
| `manual_down_kv_to_pred` | 4031.6597 | 4024.5194/4041.0555 | 2526.5356 | 1505.1241 | `{'marlin_nvfp4': 56, 'dense_nvfp4/marlin_nvfp4': 128, 'bf16': 64}` |
| `pred` | 4032.9041 | 4018.7266/4043.4851 | 2533.7224 | 1499.1817 | `{'marlin_nvfp4': 56, 'dense_nvfp4/marlin_nvfp4': 128, 'bf16': 64}` |
| `manual_kv_to_pred` | 4036.5612 | 4006.2529/4061.2787 | 2563.8367 | 1472.7245 | `{'marlin_nvfp4': 88, 'dense_nvfp4/marlin_nvfp4': 96, 'bf16': 64}` |
| `pred_down_to_manual` | 4038.5416 | 4008.5978/4064.5527 | 2568.7132 | 1469.8284 | `{'marlin_nvfp4': 88, 'dense_nvfp4/marlin_nvfp4': 96, 'bf16': 64}` |
| `pred_kv_to_manual` | 4048.1034 | 4014.5322/4114.2393 | 2556.9006 | 1491.2028 | `{'marlin_nvfp4': 72, 'dense_nvfp4/marlin_nvfp4': 128, 'bf16': 48}` |
| `manual` | 4051.3494 | 3713.2341/4394.6005 | 2536.6023 | 1514.7471 | `{'marlin_nvfp4': 104, 'dense_nvfp4/marlin_nvfp4': 96, 'bf16': 48}` |
| `single_sparse_bf16` | 4337.9648 | 3650.7295/4681.6334 | 2508.5806 | 1829.3843 | `{'sparse_bf16': 248}` |
