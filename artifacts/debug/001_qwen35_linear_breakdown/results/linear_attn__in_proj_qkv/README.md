# Qwen3.5-9B Linear Kernel Breakdown

## Scenario

- Model: `Qwen3.5-9B`
- Layer: `language_model.layers.0.linear_attn.in_proj_qkv`
- Shape: `N=8192, K=4096`
- Workload: `batch_size=1, input_tokens=16384, output_tokens=32`
- M: `prefill=16384, decode=1`
- GPU: `NVIDIA GeForce RTX 5090`

## Breakdown

| Path | Build ms | Conversion ms | Prefill first ms | Prefill steady ms | Decode first ms | Decode steady ms | Decode x32 steady ms | Runtime steady ms | E2E steady with build/conversion ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sparse_bf16` | 941.216203 | 0.000000 | 41.656673 | 2.503875 | 45.697186 | 0.128890 | 4.124467 | 6.628342 | 947.844546 |
| `dense_nvfp4_prefill_marlin_decode_explicit` | 117.098284 | 281.859175 | 2.965472 | 2.492605 | 2.041792 | 0.047126 | 1.508045 | 4.000650 | 402.958109 |
| `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 2.160860 | 0.000000 | 2.714496 | 2.478624 | 1.245536 | 0.052310 | 1.673933 | 4.152557 | 6.313417 |

## Observations

- Runtime-only steady latency: sparse_bf16 is `6.628342ms`, explicit dense_nvfp4+marlin is `4.000650ms`; sparse is `2.627693ms` slower for this layer and workload.
- Offline preparation is very different: sparse_bf16 build/pack is `941.216203ms`, while explicit canonical+CUTLASS+Marlin preparation is `398.957459ms` in this run.
- Lazy wrapper steady runtime is close to explicit dense_nvfp4+marlin, but the first prefill/decode calls include lazy materialization; first-inclusive runtime is `5.581654ms`, `1.429097ms` above steady runtime.
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
