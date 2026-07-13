# Llama2-7B-Chat vLLM Max-Speed Summary

## Workloads and Protocols

| Scenario | Workload | Ours speed protocol |
| --- | --- | --- |
| `prefill_only` | batch 8, input 2048, output 1 | Baseline-aligned phase one-shot runner: fixed `TokensPrompt`, `max_model_len=2049`, prefix cache disabled, 1 warmup + 5 measured fresh processes. |
| `prefill_decode` | batch 16, input 2048, output 80 | Existing vLLM phase-heterogeneous fresh-process protocol: 1 warmup + 10 measured fresh processes per output length. |

The uniform baseline speed data uses the existing baseline runner. The two ours
rows above use the scenario-specific protocols required by the phase runtime;
do not treat values from different protocols as a strict same-runner ranking.

## Speed

| Method | Prefill-only E2E ms | Prefill-decode E2E ms | Prefill-decode TTFT ms | Prefill-decode TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| dense BF16 | 1079.536 | 4868.068 | 2151.099 | 34.392 |
| dense NVFP4 | 578.163 | 4293.725 | 1157.180 | 39.703 |
| Marlin NVFP4 | 1034.992 | 3495.347 | 2062.112 | 18.142 |
| sparse BF16 | 660.948 | 3424.870 | 1299.185 | 26.907 |
| sparse NVFP4 | **520.632** | 4723.443 | **1051.052** | 46.486 |
| ours max-speed | 527.083 | **3079.879** | 1562.697 | 19.205 |

For `prefill_only`, ours is 1.24% slower than uniform sparse-NVFP4. For
`prefill_decode`, ours has the lowest measured E2E under its phase-specific
protocol and a TPOT close to uniform Marlin NVFP4.

## PMPD Quality

Baseline quality is shared across the two speed workloads. Ours quality is
evaluated separately for each scenario's exported checkpoint.

| Method | CNN/DM Rouge-L / BERTScore | DSum Rouge-L / BERTScore | IWSLT Rouge-L / SacreBLEU |
| --- | ---: | ---: | ---: |
| dense BF16 | 23.671 / 87.185 | 21.688 / 87.176 | 46.701 / 19.296 |
| dense NVFP4 | 24.273 / 87.204 | 20.592 / 86.691 | 44.969 / 16.835 |
| Marlin NVFP4 | **24.579 / 87.251** | 21.368 / 87.053 | **46.772 / 18.200** |
| sparse BF16 | 15.350 / 82.131 | 13.539 / 75.392 | 15.798 / 3.900 |
| sparse NVFP4 | 2.047 / 9.626 | 9.388 / 61.906 | 1.133 / 0.238 |
| ours prefill-only | 0.803 / 74.022 | 1.178 / 76.702 | 0.835 / 0.023 |
| ours prefill-decode | 23.544 / 87.082 | 21.581 / 87.154 | 45.309 / 18.301 |

## Conclusions

- `prefill_decode` is the usable max-speed policy: it retains quality close to
  dense/Marlin baselines while delivering the best reported phase-policy E2E.
- `prefill_only` is not a usable quality-speed point: its mixed sparse policy
  is nearly as fast as uniform sparse-NVFP4 but has severe PMPD quality loss.
- Pareto policy search and quality-budget optimization remain TODO; this report
  covers only unconstrained predictor max-speed policies.

## Source Tables

- Baseline speed: `artifacts/exports/vllm/baselines/llama2-7b-chat/results/summary/speed_summary.csv`
- Baseline quality: `artifacts/exports/vllm/baselines/llama2-7b-chat/results/summary/quality_summary.csv`
- Ours speed: `artifacts/exports/vllm/ours/llama2-7b-chat/max_speed/summary/speed_summary.csv`
- Ours quality: `artifacts/exports/vllm/ours/llama2-7b-chat/max_speed/summary/quality_summary.csv`
