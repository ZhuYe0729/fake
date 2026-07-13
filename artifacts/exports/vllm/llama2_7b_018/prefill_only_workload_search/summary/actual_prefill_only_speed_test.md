# Actual Prefill-Only Speed Test

## Setup

- Backend: vLLM environment `/home/agent/wja/miniconda3/envs/vllm`, vLLM `0.11.1.dev0+gb8b302cde.d20260624`.
- Metric: median latency over 5 measured iterations after 1 warmup.
- Workload proxy: `output_seq=1`.
- GPUs: idle RTX 5090 devices, excluding busy GPU 1.
- Hetero checkpoints reused existing max-speed exports; no new checkpoint was exported.

## Results

| scenario | best single | best single median ms | max-speed hetero policy | hetero median ms | hetero vs best single |
|---|---|---:|---|---:|---:|
| `b8_in2048_out1` | `sparse_nvfp4` | 515.656 | `maxspeed_004_f2600ffcfc` | 507.450 | 1.016x |
| `b1_in512_out1` | `sparse_bf16` | 25.379 | `maxspeed_005_4746310a30` | 20.769 | 1.222x |

## Interpretation

- A meaningful larger prefill case, `batch=8,input_seq=2048`, does not show a clear practical win in vLLM: only `1.016x` over best single.
- A small prefill-only case, `batch=1,input_seq=512`, does show a clear speed win: `1.222x` over best single.
- Therefore, if accuracy is ignored and a prefill-only speed-only example is acceptable, `batch=1,input_seq=512,output_seq=1` is currently the best measured point.
- If the workload needs to be a substantial batched prefill scenario, current measured evidence still does not support a strong prefill-only advantage over best single.

## Source Outputs

- Uniform `b8_in2048`: `prefill_only_workload_search/b8_in2048_uniform_vllm_env/summary_long.csv`
- Hetero `b8_in2048`: `prefill_only_workload_search/b8_in2048_maxspeed_hetero_vllm/max_speed_hetero_summary.csv`
- Uniform `b1_in512`: `prefill_only_workload_search/b1_in512_uniform_vllm_env/summary_long.csv`
- Hetero `b1_in512`: `prefill_only_workload_search/b1_in512_maxspeed_hetero_vllm/max_speed_hetero_summary.csv`
