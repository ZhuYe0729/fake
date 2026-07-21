# Llama3.1-8B-Instruct canonical prefill-only results

All methods use the same `phase_hetero_mytest` vLLM runtime. Speeds are independently measured E2E prefill medians (five runs, B=8, input=2048). NLL and downstream task values are real phase-vLLM measurements; task evaluation uses conservative runtime allocation solely for stability and is not a timing measurement.

## All measured policies

| family | policy | speed (ms) | speedup | ΔNLL | WikiText PPL | WinoGrande | ARC-Easy | ARC-Challenge | MMLU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | dense BF16 | 1294.20 | 1.000 | 0.0000 | 9.4250 | 0.7348 | 0.7971 | 0.5563 | 0.6840 |
| uniform | dense NVFP4 | 700.13 | 1.849 | 0.0965 | 10.3757 | 0.7206 | 0.7626 | 0.5111 | 0.6350 |
| uniform | sparse BF16 | 1026.50 | 1.261 | 0.6792 | 21.4360 | 0.6606 | 0.6334 | 0.3874 | 0.4296 |
| uniform | sparse NVFP4 | 644.43 | 2.008 | 1.5595 | 65.0020 | 0.5328 | 0.4327 | 0.2637 | 0.2388 |
| uniform | Marlin W4A16 | 1184.64 | 1.092 | 0.0498 | 9.9351 | 0.7309 | 0.7782 | 0.5367 | 0.6620 |
| ours | p003 | 1184.74 | 1.092 | 0.0141 | 9.5668 | 0.7411 | 0.7988 | 0.5597 | 0.6828 |
| ours | p005 | 947.78 | 1.366 | 0.0300 | 9.7420 | 0.7372 | 0.7828 | 0.5461 | 0.6727 |
| ours | p007 | 825.91 | 1.567 | 0.0572 | 9.9990 | 0.7167 | 0.7635 | 0.5290 | 0.6637 |
| ours | bridge-072 | 902.92 | 1.433 | 0.0559 | 9.9978 | 0.7230 | 0.7769 | 0.5350 | 0.6655 |
| ours | bridge-088 | 858.34 | 1.508 | 0.0702 | 10.1183 | 0.7143 | 0.7774 | 0.5486 | 0.6647 |
| ours | bridge-104 | 742.66 | 1.743 | 0.0809 | 10.2129 | 0.7230 | 0.7706 | 0.5299 | 0.6555 |
| ours | bridge-120 | 705.21 | 1.835 | 0.0931 | 10.3553 | 0.7230 | 0.7580 | 0.5384 | 0.6399 |
| ours | p009 | 788.60 | 1.641 | 0.2255 | 12.1336 | 0.7198 | 0.7525 | 0.4915 | 0.5689 |
| ours | p011 | 768.46 | 1.684 | 0.3901 | 15.1288 | 0.6748 | 0.6755 | 0.4172 | 0.4254 |
| ours | p014 | 739.67 | 1.750 | 1.0074 | 31.9439 | 0.6109 | 0.5787 | 0.3302 | 0.3114 |

## Artifacts

- `all_policy_task_results.csv`: machine-readable result table.
- `pareto_speed_vs_real_nll.png` and `pareto_speed_vs_{wikitext,winogrande,arc_easy,arc_challenge,mmlu}.png`: paper-facing Pareto plots.
