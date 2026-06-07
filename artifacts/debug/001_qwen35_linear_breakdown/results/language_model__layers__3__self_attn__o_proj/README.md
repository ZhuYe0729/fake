# Qwen3.5-9B Linear Kernel Breakdown

## Scenario

- Model: `Qwen3.5-9B`
- Layer: `language_model.layers.3.self_attn.o_proj`
- Shape: `N=4096, K=4096`
- Workload: `batch_size=1, input_tokens=16384, output_tokens=32`
- M: `prefill=16384, decode=1`
- GPU: `NVIDIA GeForce RTX 5090`

## Breakdown

| Path | Build ms | Conversion ms | Prefill first ms | Prefill steady ms | Decode first ms | Decode steady ms | Decode x32 steady ms | Runtime steady ms | E2E steady with build/conversion ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sparse_bf16` | 84.412912 | 0.000000 | 39.477856 | 1.296390 | 38.230751 | 0.128234 | 4.103475 | 5.399866 | 89.812777 |
| `dense_nvfp4_prefill_marlin_decode_explicit` | 0.917702 | 1.555404 | 2.043584 | 1.995757 | 0.098016 | 0.043955 | 1.406566 | 3.402323 | 5.875429 |
| `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 0.944726 | 0.000000 | 2.189344 | 2.000797 | 1.038880 | 0.051488 | 1.647616 | 3.648413 | 4.593139 |

## Observations

- Runtime-only steady latency: sparse_bf16 is `5.399866ms`, explicit dense_nvfp4+marlin is `3.402323ms`; sparse is `1.997542ms` slower for this layer and workload.
- Offline preparation is very different: sparse_bf16 build/pack is `84.412912ms`, while explicit canonical+CUTLASS+Marlin preparation is `2.473106ms` in this run.
- Lazy wrapper steady runtime is close to explicit dense_nvfp4+marlin, but the first prefill/decode calls include lazy materialization; first-inclusive runtime is `4.824352ms`, `1.175939ms` above steady runtime.
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
