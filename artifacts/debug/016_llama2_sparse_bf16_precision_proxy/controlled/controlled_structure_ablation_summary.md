# Controlled Structure Ablation

Leave-one-pair-out results on raw-local-matched controlled pairs. The target is `loss(high_final) - loss(low_final)`.

| method | variant | pairs | Pearson | Spearman | MAE | direction acc |
|---|---|---:|---:|---:|---:|---:|
| sparse_bf16 | raw_only | 16 | 0.2047 | 0.2059 | 0.006810 | 0.8125 |
| sparse_bf16 | layer_depth | 16 | -0.1045 | -0.0676 | 0.012986 | 0.3750 |
| sparse_bf16 | linear_type | 16 | 0.4986 | 0.5265 | 0.008150 | 0.6875 |
| sparse_bf16 | layer_type | 16 | 0.1768 | 0.1618 | 0.008424 | 0.7500 |
| dense_nvfp4 | raw_only | 16 | -0.9961 | -0.9853 | 0.006824 | 0.9375 |
| dense_nvfp4 | layer_depth | 16 | 0.2710 | 0.1412 | 0.009667 | 0.9375 |
| dense_nvfp4 | linear_type | 16 | 0.0468 | -0.2118 | 0.010606 | 0.9375 |
| dense_nvfp4 | layer_type | 16 | -0.0795 | -0.1529 | 0.014667 | 0.9375 |
| sparse_nvfp4 | raw_only | 16 | -0.5944 | -0.5676 | 0.032617 | 0.8125 |
| sparse_nvfp4 | layer_depth | 16 | 0.6197 | 0.6794 | 0.021538 | 0.8750 |
| sparse_nvfp4 | linear_type | 16 | -0.5307 | -0.4118 | 0.049512 | 0.6875 |
| sparse_nvfp4 | layer_type | 16 | 0.1877 | 0.1353 | 0.029313 | 0.7500 |

## Plot

- `/root/wja/project/my/cospaq/fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy/controlled/controlled_structure_ablation_pair_delta.png`
