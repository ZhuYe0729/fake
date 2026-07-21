# Llama3 prefill high-sparse speed diagnosis

## Result

The Llama3 high-sparse mismatch is primarily an E2E composition/runtime effect,
with a smaller but non-zero per-linear predictor contribution.

| policy | predicted local | exact local | measured E2E | local prediction error | E2E minus exact local |
|---|---:|---:|---:|---:|---:|
| uniform dense-NVFP4 (`p01`) | 527.73 ms | 501.88 ms | 700.13 ms | +5.2% | 198.26 ms |
| uniform sparse-BF16 (`p02`) | 529.67 ms | 578.77 ms | 1026.50 ms | -8.5% | 447.73 ms |
| uniform sparse-NVFP4 (`p03`) | 453.71 ms | 446.33 ms | 644.43 ms | +1.7% | 198.10 ms |
| ours max predicted speed (`p014`) | 380.32 ms | 362.05 ms | 739.67 ms | +5.0% | 377.62 ms |

For the dense-NVFP4 versus sparse-BF16 uniform comparison, observed E2E
difference is 326.37 ms. Exact local latency explains 76.89 ms of it (about
24%); the E2E residual explains 249.47 ms (about 76%). Therefore local model
error alone cannot explain the gap.

The Llama3 gate/up shape (`M=16384, N=28672, K=4096`) has a relevant local
error: sparse-BF16 was predicted at 8.98 ms but measured at 10.68 ms (-15.9%).
This should be incorporated when the local predictor is next refreshed, but
the p014 failure remains after replacing every local prediction with exact
microbenchmark values.

## Implication

Do not add a generic scalar speed scale. The next E2E calibrator should retain
the exact/predicted local sum and add natural composition features: sparse-BF16
local-time share, sparse-NVFP4 local-time share, and the number of method
transitions across the ordered module execution sequence. It must be calibrated
with high-sparsity mixed anchors, including the `96 sparse-BF16 + 32
sparse-NVFP4` endpoint. The local predictor refresh is a separate, smaller
fix using the new exact Llama3 shape rows.
