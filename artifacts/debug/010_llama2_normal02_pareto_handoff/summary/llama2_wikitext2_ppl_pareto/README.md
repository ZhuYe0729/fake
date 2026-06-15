# Llama2-7B WikiText-2 PPL Pareto Summary

This report replaces ARC-Challenge as the current final quality check for the normal02 prefill+decode scenario. Quality is measured by teacher-forced WikiText-2 test perplexity; speed remains the existing normal02 prefill+decode latency model and stable E2E timing where available.

## Main Observations

- Dense bf16 P0: PPL 5.4951, NLL 1.703854.
- Fastest measured Pareto point P9: measured speedup 1.2207x, PPL 5.6291, relative PPL delta 2.44%.
- The Pareto policies increase PPL smoothly as predicted speed improves; P9 matches the all-nvfp4 quality endpoint but is faster than uniform marlin/dense-nvfp4 in the normal02 cost model.
- dense_nvfp4, marlin_nvfp4, and dense_nvfp4_prefill_marlin_decode have identical PPL here because all three use the same dense nvfp4 compressed weights for quality; their difference is only the runtime backend assignment.
- P1-P3 now have PPL but still do not have stable measured E2E repeats; plots include a predicted-speed curve for all points and a measured-speed curve for the already measured subset.

## Pareto Points

| point | pred speedup | measured speedup | PPL | PPL delta % | replaced modules |
|---:|---:|---:|---:|---:|---:|
| P0 | 1.0000x | 1.0000x | 5.4951 | 0.00% | 0 |
| P1 | 1.0098x |  | 5.5387 | 0.79% | 4 |
| P2 | 1.0163x |  | 5.5402 | 0.82% | 7 |
| P3 | 1.0297x |  | 5.5440 | 0.89% | 9 |
| P4 | 1.0554x | 1.0058x | 5.5476 | 0.96% | 16 |
| P5 | 1.0991x | 1.0175x | 5.5509 | 1.02% | 27 |
| P6 | 1.1725x | 1.0588x | 5.5593 | 1.17% | 49 |
| P7 | 1.3092x | 1.0821x | 5.5766 | 1.48% | 71 |
| P8 | 1.4617x | 1.1553x | 5.6037 | 1.98% | 168 |
| P9 | 1.4763x | 1.2207x | 5.6291 | 2.44% | 224 |

## Uniform Baselines

| method | predicted speedup | PPL | PPL delta % | replaced modules |
|---|---:|---:|---:|---:|
| dense_bf16 | 1.0000x | 5.4951 | 0.00% | 0 |
| dense_nvfp4 | 0.4862x | 5.6291 | 2.44% | 224 |
| marlin_nvfp4 | 1.3696x | 5.6291 | 2.44% | 224 |
| dense_nvfp4_prefill_marlin_decode | 1.4725x | 5.6291 | 2.44% | 224 |

## Files

- `pareto_wikitext2_ppl_summary.csv`: joined Pareto PPL plus stable E2E timing where available.
- `uniform_wikitext2_ppl_summary.csv`: uniform method PPL baselines.
- `pareto_predicted_speed_vs_wikitext2_ppl.png`: all Pareto points and uniform baselines on predicted speed.
- `pareto_measured_speed_vs_wikitext2_ppl.png`: measured E2E subset, with uniform baselines shown using predicted speed for reference.

## Next Steps

1. Run stable E2E repeats for P1-P3, so the measured-speed curve covers every PPL-validated point.
2. Add measured E2E timings for uniform marlin_nvfp4 and dense_nvfp4_prefill_marlin_decode, because they are the important speed baselines under the same final PPL.
3. Use WikiText-2 PPL as the primary quality axis for the next optimization audit; ARC can remain optional downstream task context, not the final metric for this prefill+decode scenario.
