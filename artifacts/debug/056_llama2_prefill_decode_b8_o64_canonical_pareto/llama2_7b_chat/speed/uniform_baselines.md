# Uniform speed baselines (same runtime)

All rows use B=8, input=2048, output=64, `phase_hetero_mytest`, VLLM V1, BF16 KV, and disabled chunked prefill.

| method | E2E median ms | speedup | TTFT ms | TPOT ms | note |
|---|---:|---:|---:|---:|---|
| dense_bf16 | 2291.843 | 1.000 | 1096.902 | 18.967 | dense BF16 in both phases |
| dense_nvfp4 | 2853.994 | 0.803 | 668.066 | 34.697 | dense NVFP4 in both phases |
| sparse_bf16 | 2020.771 | 1.134 | 701.848 | 20.935 | canonical sparse BF16 in both phases |
| sparse_nvfp4_legal_projection | 2991.496 | 0.766 | 679.030 | 36.706 | prefill sparse NVFP4; decode dense NVFP4 because sparse NVFP4 is unsupported at M=8 |
| w4a16_marlin | 1925.835 | 1.190 | 1128.222 | 12.661 | W4A16/Marlin NVFP4 in both phases |
