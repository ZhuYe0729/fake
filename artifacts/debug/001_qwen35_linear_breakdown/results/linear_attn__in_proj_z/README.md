# Qwen3.5-9B Linear Kernel Breakdown

## Scenario

- Model: `Qwen3.5-9B`
- Layer: `language_model.layers.0.linear_attn.in_proj_z`
- Shape: `N=4096, K=4096`
- Workload: `batch_size=1, input_tokens=16384, output_tokens=32`
- M: `prefill=16384, decode=1`
- GPU: `NVIDIA GeForce RTX 5090`

## Breakdown

| Path | Build ms | Conversion ms | Prefill first ms | Prefill steady ms | Decode first ms | Decode steady ms | Decode x32 steady ms | Runtime steady ms | E2E steady with build/conversion ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sparse_bf16` | 88.719905 | 0.000000 | 39.706913 | 1.294922 | 38.257824 | 0.125690 | 4.022067 | 5.316989 | 94.036894 |
| `dense_nvfp4_prefill_marlin_decode_explicit` | 0.931180 | 1.568600 | 2.058880 | 1.997859 | 0.175936 | 0.043571 | 1.394278 | 3.392138 | 5.891917 |
| `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 0.973126 | 0.000000 | 2.169280 | 1.999427 | 1.031456 | 0.051232 | 1.639424 | 3.638851 | 4.611977 |

## Observations

- Runtime-only steady latency: sparse_bf16 is `5.316989ms`, explicit dense_nvfp4+marlin is `3.392138ms`; sparse is `1.924851ms` slower for this layer and workload.
- Offline preparation is very different: sparse_bf16 build/pack is `88.719905ms`, while explicit canonical+CUTLASS+Marlin preparation is `2.499780ms` in this run.
- Lazy wrapper steady runtime is close to explicit dense_nvfp4+marlin, but the first prefill/decode calls include lazy materialization; first-inclusive runtime is `4.788928ms`, `1.150077ms` above steady runtime.
- These numbers are single-layer debug timings. Build/materialization costs are offline costs unless the wrapper leaves them lazy and pays them during the first timed forward.

## Notes

- `first` includes any first-call runtime initialization that remains after explicit build/materialization.
- The lazy wrapper row intentionally leaves CUTLASS/Marlin materialization inside the first prefill/decode calls.
- `runtime steady` excludes build/conversion and uses warmed forward latency.
- `E2E steady with build/conversion` adds build/conversion once to steady prefill + 32 decode steps.

## Files

- `results/breakdown.json`: full structured result.
- `results/breakdown.csv`: flat comparison table.
- `scripts/qwen35_linear_breakdown.py`: reproduction script.
