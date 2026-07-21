# Uniform speed baselines (same runtime)

All rows use B=8, input=2048, output=64, `phase_hetero_mytest`, VLLM V1, BF16 KV, and disabled chunked prefill.

| method | E2E median ms | speedup | TTFT ms | TPOT ms | note |
|---|---:|---:|---:|---:|---|
| dense_bf16 | 2193.379 | 1.000 | 1150.701 | 16.550 | dense BF16 in both phases |
| dense_nvfp4 | 2641.574 | 0.830 | 674.719 | 31.220 | dense NVFP4 in both phases |
| sparse_bf16 | 1937.578 | 1.132 | 753.193 | 18.800 | canonical sparse BF16 in both phases |
| sparse_nvfp4_legal_projection | 2770.355 | 0.792 | 635.219 | 33.891 | prefill sparse NVFP4; decode dense NVFP4 because sparse NVFP4 is unsupported at M=8 |
| w4a16_marlin | 1992.982 | 1.101 | 1171.951 | 13.032 | W4A16/Marlin NVFP4 in both phases |
