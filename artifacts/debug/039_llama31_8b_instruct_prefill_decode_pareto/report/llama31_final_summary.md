# Llama-3.1-8B-Instruct: final measured summary

## Scope and protocol

- Model: `Meta-Llama-3.1-8B-Instruct`.
- Prefill-only: batch 8, input 2048; actual quality is full ARC-Challenge
  (1172 samples), and speed/NLL axes are freshly measured.
- Prefill-decode (PMPD): batch 16, input 2048, output 80; E2E speedups and
  WikiText ΔNLL are freshly measured using the continuous phase runtime.
  CNN/DM-1000, DialogSum-1500, and IWSLT-333 are independently generated and
  scored with the existing PMPD evaluator.
- Frozen uniform artifacts are read-only inputs.  The two selected mixed PMPD
  points were generated in a persistent vLLM instance per shard, using
  `prepare_next_prefill()` between batches rather than reloading weights per
  request.

## Prefill-only: ARC-Challenge

| family | point | measured speedup | ARC normalized accuracy |
|---|---|---:|---:|
| Uniform | dense BF16 | 1.000x | 0.5538 |
| Ours | point 6 | 1.560x | 0.5486 |
| Uniform | dense NVFP4 | 1.728x | 0.5410 |
| Ours | point 8 | 1.788x | 0.5265 |
| Ours | point 9 | 1.856x | 0.4940 |
| Ours | point 11 | 2.035x | 0.4454 |
| Ours | point 13 | 2.168x | 0.3311 |

The mixed frontier includes the dense-NVFP4 quality region and also offers a
near-dense-BF16 point at 1.560x.  Full source data and figure:
`../../038_llama31_8b_instruct_prefill_only_pareto/arc_challenge/report/`.

## Prefill-decode: measured NLL frontier

| family | point | measured speedup | actual WikiText ΔNLL |
|---|---|---:|---:|
| Uniform | dense BF16 | 1.000x | 0.000 |
| Ours | point 002 | 1.096x | 0.387 |
| Ours | point 004 | 1.267x | 0.543 |
| Uniform | dense NVFP4 | 1.058x | 2.882 |
| Uniform | sparse BF16 | 1.173x | 55.306 |
| Ours | point 009 / max speed | 1.692x | 2.882 |

At 1.096x and 1.267x, the selected mixed policies have substantially smaller
measured NLL loss than dense-NVFP4 while being faster; point 004 is also both
faster and much higher quality than sparse-BF16.

## Prefill-decode: real downstream quality

| policy | speedup | CNN/DM ROUGE-L | DialogSum ROUGE-L | IWSLT SacreBLEU |
|---|---:|---:|---:|---:|
| Uniform dense BF16 | 1.000x | 19.487 | 13.784 | 10.680 |
| Uniform dense NVFP4 | 1.058x | 16.345 | 8.203 | 10.312 |
| Uniform sparse BF16 | 1.173x | 12.834 | 4.049 | 3.276 |
| Ours high-quality (point 002) | 1.096x | 20.274 | 13.473 | 10.586 |
| Ours middle (point 004) | 1.267x | 18.747 | 13.266 | 10.654 |
| Ours max speed (point 009) | 1.692x | 16.840 | 9.085 | 10.570 |

The high-quality point nearly preserves dense-BF16 on DialogSum/IWSLT while
improving CNN/DM and attaining a measured 1.096x E2E speedup.  The middle
point preserves strong scores on all three tasks while occupying a speed
region where both uniform dense-NVFP4 and sparse-BF16 are markedly worse.

## Paper-ready artifacts

- Measured NLL frontier: `../closure/report/pareto_measured_speed_nll.png`.
- Prefill-only ARC frontier:
  `../../038_llama31_8b_instruct_prefill_only_pareto/arc_challenge/report/pareto_speedup_vs_arc_challenge.png`.
- PMPD task frontiers: `../closure/tasks/report/pareto_speed_vs_{cnn_dm_1000,dsum,IWSLT}.png`.
- All PMPD task rows and metric paths: `../closure/tasks/report/downstream_summary.csv`.

## Caveat

The speedup column is the formal batch-16/input-2048/output-80 E2E benchmark,
not the generation throughput reported by the evaluator for its longer
task-specific continuations.  The latter is retained in each task `metrics.json`
only as a run diagnostic.
