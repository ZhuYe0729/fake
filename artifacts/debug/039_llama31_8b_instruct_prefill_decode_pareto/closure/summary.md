# Llama-3.1-8B-Instruct prefill-decode measured closure

| kind | policy | E2E ms | speedup vs dense | actual ΔNLL | predicted ΔNLL |
|---|---|---:|---:|---:|---:|
| ours | point_000 | 3432.16 | 1.000 | 0.0000 | 0.0000 |
| ours | point_002 | 3132.48 | 1.096 | 0.3871 | 0.0439 |
| ours | point_004 | 2708.39 | 1.267 | 0.5428 | 0.1702 |
| ours | point_006 | 2312.23 | 1.484 | 1.6379 | 0.3259 |
| ours | point_008 | 2165.02 | 1.585 | 2.8214 | 0.5803 |
| ours | point_009 | 2028.53 | 1.692 | 2.8819 | 0.9302 |
| uniform | dense_bf16 | 3436.91 | 0.999 | 0.0000 |  |
| uniform | dense_nvfp4 | 3242.68 | 1.058 | 2.8819 |  |
| uniform | sparse_bf16 | 2926.34 | 1.173 | 55.3057 |  |
| uniform | w4a16_ours | 2922.91 | 1.174 | 2.8819 |  |
