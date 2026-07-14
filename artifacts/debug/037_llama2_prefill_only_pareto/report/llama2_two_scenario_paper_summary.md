# Llama2-7B-Chat：两场景论文汇总

本表只使用已经完成的实测结果。速度均为相对该场景 dense BF16 的
端到端加速比；ours 选择位于对应实测任务质量并集前沿、且最适合展示
速度--质量折中的代表点。不同场景有不同 workload 和 runner，**不应在
两个场景之间直接比较加速比或绝对分数**。

## 场景与指标

| 场景 | 服务 workload | 任务质量 |
|---|---|---|
| prefill-only | batch=8, input=2048, API output=1 | ARC-Challenge 0-shot `acc_norm`，1,172 examples |
| prefill-decode | batch=16, input=2048, output=80 | CNN/DM-1000 ROUGE-L、DialogSum ROUGE-L、IWSLT SacreBLEU |

## Prefill-only

| 方法 | E2E speedup | ARC-Challenge `acc_norm` |
|---|---:|---:|
| Dense BF16 | 1.000x | 0.4334 |
| Marlin NVFP4 | 1.043x | 0.4283 |
| Sparse BF16 | 1.633x | 0.3532 |
| Dense NVFP4 | 1.867x | 0.4283 |
| Sparse NVFP4 | 2.074x | 0.2398 |
| **Ours (point 12)** | **1.726x** | **0.4420** |

`point 12` is selected because it lies on the measured ARC union frontier: it
provides a 1.726x speedup while matching/exceeding the measured task score of
the uniform alternatives. ARC is evaluated once on a fixed 1,172-example set;
the result should be described as a measured trade-off, not as a claim of
statistical superiority without confidence intervals.

## Prefill-decode

| 方法 | E2E speedup | CNN/DM ROUGE-L | DialogSum ROUGE-L | IWSLT BLEU |
|---|---:|---:|---:|---:|
| Dense BF16 | 1.000x | 23.67 | 21.69 | 19.30 |
| Dense NVFP4 | 1.188x | 24.27 | 20.59 | 16.84 |
| Marlin NVFP4 | 1.387x | 24.58 | 21.37 | 18.20 |
| Sparse BF16 | 1.442x | 15.35 | 13.54 | 3.90 |
| Sparse NVFP4 | 1.185x | 2.05 | 9.39 | 0.24 |
| **Ours (point 11)** | **1.714x** | **23.54** | **21.58** | **18.30** |

`point 11` is selected as the high-speed representative: it is 1.714x faster
than dense BF16 while retaining near-dense scores on all three generation tasks
and avoiding the severe degradation of sparse uniform references. It also
trades off against Marlin NVFP4 rather than claiming to dominate every metric:
Marlin has higher CNN/DM ROUGE-L, while ours is substantially faster and has
higher DialogSum/IWSLT scores.

## Paper-facing interpretation

- The method should be presented as learning a **heterogeneous Pareto trade-
  off**, not as universally dominating every uniform compression method.
- Uniform methods are included in the measured union frontier. This is why
  dense-NVFP4 and sparse-NVFP4 remain visible endpoints where appropriate.
- The strongest concise prefill-only claim is: “ours reaches 1.726x speedup
  with 0.442 ARC `acc_norm`, while uniform dense-NVFP4 reaches 1.867x with
  0.428.”
- The strongest concise prefill-decode claim is: “ours reaches 1.714x E2E
  speedup with near-dense quality on all three generation tasks.”

## Result sources

- Prefill-only all-point summary:
  `artifacts/debug/037_llama2_prefill_only_pareto/arc_challenge/report/arc_challenge_speed_summary.csv`.
- Prefill-only plots:
  `arc_challenge/report/pareto_speedup_vs_arc_challenge.png` and
  `arc_challenge/report/pareto_speedup_vs_arc_challenge_quality_plateau.png`.
- Prefill-decode ours task scores:
  `artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/task_quality_all/summary.csv`.
- Prefill-decode baseline/task comparison values:
  `artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/task_quality_all/report/all_task_pareto_points.csv`.
- Prefill-decode measured NLL/speed curve:
  `artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/report/formal_util085_actual_nll_summary.csv` and
  `artifacts/debug/036_llama2_prefill_decode_intermediate_points/report/pareto_speedup_vs_wikitext_with_intermediates.png`.
