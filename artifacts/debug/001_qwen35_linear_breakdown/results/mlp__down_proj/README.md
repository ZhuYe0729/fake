# Qwen3.5-9B Linear Kernel Breakdown

## Scenario

- Model: `Qwen3.5-9B`
- Layer: `language_model.layers.0.mlp.down_proj`
- Shape: `N=4096, K=12288`
- Workload: `batch_size=1, input_tokens=16384, output_tokens=32`
- M: `prefill=16384, decode=1`
- GPU: `NVIDIA GeForce RTX 5090`

## Breakdown

| Path | Build ms | Conversion ms | Prefill first ms | Prefill steady ms | Decode first ms | Decode steady ms | Decode x32 steady ms | Runtime steady ms | E2E steady with build/conversion ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sparse_bf16` | 93.394190 | 0.000000 | 41.581375 | 3.618214 | 37.271809 | 0.217629 | 6.964122 | 10.582336 | 103.976526 |
| `dense_nvfp4_prefill_marlin_decode_explicit` | 3.155568 | 1.270629 | 6.068832 | 5.729645 | 0.116576 | 0.043133 | 1.380250 | 7.109894 | 11.536092 |
| `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 3.119632 | 0.000000 | 5.897568 | 5.740291 | 1.055392 | 0.051798 | 1.657549 | 7.397840 | 10.517472 |

## Observations

- Runtime-only steady latency: sparse_bf16 is `10.582336ms`, explicit dense_nvfp4+marlin is `7.109894ms`; sparse is `3.472441ms` slower for this layer and workload.
- Offline preparation is very different: sparse_bf16 build/pack is `93.394190ms`, while explicit canonical+CUTLASS+Marlin preparation is `4.426197ms` in this run.
- Lazy wrapper steady runtime is close to explicit dense_nvfp4+marlin, but the first prefill/decode calls include lazy materialization; first-inclusive runtime is `8.558711ms`, `1.160871ms` above steady runtime.
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
