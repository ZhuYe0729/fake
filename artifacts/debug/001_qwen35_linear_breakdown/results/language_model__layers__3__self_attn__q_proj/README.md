# Qwen3.5-9B Linear Kernel Breakdown

## Scenario

- Model: `Qwen3.5-9B`
- Layer: `language_model.layers.3.self_attn.q_proj`
- Shape: `N=8192, K=4096`
- Workload: `batch_size=1, input_tokens=16384, output_tokens=32`
- M: `prefill=16384, decode=1`
- GPU: `NVIDIA GeForce RTX 5090`

## Breakdown

| Path | Build ms | Conversion ms | Prefill first ms | Prefill steady ms | Decode first ms | Decode steady ms | Decode x32 steady ms | Runtime steady ms | E2E steady with build/conversion ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sparse_bf16` | 88.277384 | 0.000000 | 40.734142 | 2.466842 | 38.221409 | 0.124397 | 3.980698 | 6.447539 | 94.724923 |
| `dense_nvfp4_prefill_marlin_decode_explicit` | 1.971500 | 1.220197 | 2.512608 | 2.472032 | 0.099200 | 0.043930 | 1.405747 | 3.877779 | 7.069477 |
| `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 2.076549 | 0.000000 | 2.674432 | 2.479318 | 1.088224 | 0.050896 | 1.628672 | 4.107990 | 6.184539 |

## Observations

- Runtime-only steady latency: sparse_bf16 is `6.447539ms`, explicit dense_nvfp4+marlin is `3.877779ms`; sparse is `2.569760ms` slower for this layer and workload.
- Offline preparation is very different: sparse_bf16 build/pack is `88.277384ms`, while explicit canonical+CUTLASS+Marlin preparation is `3.191697ms` in this run.
- Lazy wrapper steady runtime is close to explicit dense_nvfp4+marlin, but the first prefill/decode calls include lazy materialization; first-inclusive runtime is `5.340432ms`, `1.232442ms` above steady runtime.
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
