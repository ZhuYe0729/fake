# 008 Llama2 Pareto Quality-Speed

Experimental workspace for quality-constrained, speed-optimized per-linear method assignment.

This directory is intentionally isolated from the main framework. It uses:

- quality inputs from `../007_llama2_quality_modeling`
- prepared compressed artifacts from `../../results/main/003_llama2_7b_arc_easy_accuracy`
- prefill-only latency for `batch_size=16,input_tokens=1024,output_tokens=0`

## Workflow

Build a candidate table using the existing oracle-summary latency as a quick first pass:

```bash
python scripts/build_cost_table.py --latency-source existing
python scripts/analyze_quality_formulas.py
python scripts/optimize_pareto.py --budgets 0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.0
python scripts/summarize_pareto.py
```

Collect fresh real-kernel prefill-only microbench latency, then rebuild:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/bench_prefill_latency.py --gpu 7
python scripts/compare_latency_sources.py
python scripts/build_cost_table.py --latency-source fresh
python scripts/optimize_pareto.py
python scripts/summarize_pareto.py
```

The first validation target is a small set of representative Pareto policies:

```bash
python scripts/select_validation_policies.py
```

Validate selected policies with real compressed weights:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/validate_pareto_quality.py --gpu 7 --points validation
```

Validate selected policies with real full-model prefill-only E2E latency:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/validate_pareto_e2e.py --gpu 7 --points validation
```

## Outputs

- `latency/prefill_latency.csv`: measured or imported per-group latency.
- `quality/formula_correlation.csv`: candidate quality proxy correlation against existing NLL/ARC ablations.
- `costs/module_method_candidates.csv`: per-module method choices with quality and latency costs.
- `pareto/pareto_points.csv`: quality-budget constrained optimum points.
- `pareto/policies/*.json`: selected per-linear policies.
- `summary/analysis.md`: compact quality-speed Pareto analysis.
- `validation/pareto_quality_validation.csv`: NLL and optional ARC results for selected Pareto policies.
- `validation/pareto_e2e_validation.csv`: real full-model prefill-only latency for selected Pareto policies.
- `validation/pareto_validation_joined.csv`: joined E2E, NLL, and ARC validation summary.
- `summary/policy_*_method_counts.csv`: policy explanation tables by family/type/layer and frontier transition.

## Scope

This pass only handles `llama2-7b` and `prefill_only`. `normal_01`, `normal_02`, decode latency, and dense-nvfp4-prefill plus marlin-decode conversion costs are TODO.

## Completed Run Notes

The completed first pass used fresh real-kernel prefill-only microbench latency on physical GPU7. The scenario is `batch_size=16,input_tokens=1024,output_tokens=0`.

Generated tables:

- `latency/prefill_latency.csv`: 35 rows, 7 Llama2 linear groups x 5 methods.
- `costs/module_method_candidates.csv`: 1120 rows, 224 modules x 5 methods.
- `pareto/pareto_unique_points.csv`: 11 unique frontier points.
- `validation/pareto_quality_validation.csv`: 5 selected policies with NLL and ARC-Challenge limit=128.
- `validation/pareto_e2e_validation.csv`: 5 selected policies with real full-model prefill-only E2E latency.
- `validation/pareto_quality_validation.csv`: expanded to all 11 unique Pareto points with NLL and ARC-Challenge limit=128.

Fresh latency method totals:

- dense_bf16: latency_sum_ms=905.1911178588867, quality_sum=0.0
- dense_nvfp4: latency_sum_ms=598.6068492889405, quality_sum=16.530083108272944
- sparse_bf16: latency_sum_ms=477.0136070251465, quality_sum=191.28447364745372
- sparse_nvfp4: latency_sum_ms=538.8999702453614, quality_sum=379.2646015372682
- marlin_nvfp4: latency_sum_ms=913.749609375, quality_sum=16.530083108272944

Pareto endpoints:

- conservative: quality=0.0, latency_ms=905.1911178588867, speedup=1.0
- speed-optimal under the constrained candidate table: quality=249.35458457377698, latency_ms=419.98264389038087, speedup=2.1553060133007533

Selected validation points:

- point 0: quality=0.0, latency_ms=905.1911178588867, nll=2.039499126068533, arc_acc=0.40625, arc_acc_norm=0.4609375
- point 2: quality=0.23199262343139426, latency_ms=891.9582732200622, nll=2.0397004465300976, arc_acc=0.4140625, arc_acc_norm=0.453125
- point 5: quality=2.842046757666721, latency_ms=766.3655449867249, nll=2.0418942035294325, arc_acc=0.3984375, arc_acc_norm=0.4609375
- point 8: quality=58.21201590943091, latency_ms=506.5709192276001, nll=2.1093044766241325, arc_acc=0.3984375, arc_acc_norm=0.453125
- point 10: quality=249.35458457377698, latency_ms=419.98264389038087, nll=2.6352587399417406, arc_acc=0.28125, arc_acc_norm=0.3203125

Real full-model E2E validation for selected points:

- point 0: predicted_linear_ms=905.1911178588867, e2e_prefill_mean_ms=1165.6009114583333, speedup=1.0
- point 2: predicted_linear_ms=891.9582732200622, e2e_prefill_mean_ms=1146.7749837239583, speedup=1.0164164094975643
- point 5: predicted_linear_ms=766.3655449867249, e2e_prefill_mean_ms=1021.3096313476562, speedup=1.141280641719083
- point 8: predicted_linear_ms=506.5709192276001, e2e_prefill_mean_ms=783.983642578125, speedup=1.4867668764430626
- point 10: predicted_linear_ms=419.98264389038087, e2e_prefill_mean_ms=694.4718424479166, speedup=1.6783990944107283

Validation correlations:

- predicted linear latency vs real E2E prefill latency on selected 5 points: pearson=0.9997408220927038, spearman=1.0
- quality cost vs NLL on all 11 quality points: pearson=0.9691908137987784, spearman=1.0
- quality cost vs ARC-Challenge acc_norm on all 11 quality points: pearson=-0.9652633710759109, spearman=-0.789776981742655

Policy transition highlights:

- Conservative budgets first choose low-risk MLP `dense_bf16 -> dense_nvfp4`.
- Mid budgets introduce `sparse_bf16`, first mostly attention and then mixed MLP/attention.
- Speed endpoint uses `160 sparse_bf16 + 64 sparse_nvfp4`; `marlin_nvfp4` is not selected for prefill-only because it is slower than dense_bf16 in the measured cost table.
