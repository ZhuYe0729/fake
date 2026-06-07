# Qwen3.5-9B Linear Kernel Breakdown

## Scenario

- Model: `Qwen3.5-9B`
- Layer: `language_model.layers.0.mlp.gate_proj`
- Shape: `N=12288, K=4096`
- Workload: `batch_size=1, input_tokens=16384, output_tokens=32`
- M: `prefill=16384, decode=1`
- GPU: `NVIDIA GeForce RTX 5090`

## Breakdown

| Path | Build ms | Conversion ms | Prefill first ms | Prefill steady ms | Decode first ms | Decode steady ms | Decode x32 steady ms | Runtime steady ms | E2E steady with build/conversion ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sparse_bf16` | 94.342838 | 0.000000 | 41.593281 | 3.630326 | 37.634079 | 0.126051 | 4.033638 | 7.663965 | 102.006803 |
| `dense_nvfp4_prefill_marlin_decode_explicit` | 2.994834 | 1.291999 | 2.987360 | 2.949155 | 0.107104 | 0.044710 | 1.430733 | 4.379888 | 8.666721 |
| `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 3.075346 | 0.000000 | 3.141664 | 2.954416 | 1.103776 | 0.051446 | 1.646285 | 4.600701 | 7.676047 |

## Observations

- Runtime-only steady latency: sparse_bf16 is `7.663965ms`, explicit dense_nvfp4+marlin is `4.379888ms`; sparse is `3.284077ms` slower for this layer and workload.
- Offline preparation is very different: sparse_bf16 build/pack is `94.342838ms`, while explicit canonical+CUTLASS+Marlin preparation is `4.286833ms` in this run.
- Lazy wrapper steady runtime is close to explicit dense_nvfp4+marlin, but the first prefill/decode calls include lazy materialization; first-inclusive runtime is `5.840278ms`, `1.239578ms` above steady runtime.
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
