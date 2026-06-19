# FakeVLM Sample Size Accuracy Analysis

## Overview

This analysis determines how many test samples are needed to reliably estimate the true accuracy of FakeVLM models. It uses statistical subsampling of the full 5000-sample predictions from `020_fakevlm_uniform_accuracy`, with 30 random seeds per sample size.

## Full Accuracy (5000 samples)

| Method | Accuracy | Correct | Wrong |
| --- | ---: | ---: | ---: |
| `dense_bf16` | 0.986400 | 4932 | 68 |
| `sparse_bf16` | 0.985200 | 4926 | 74 |
| `dense_nvfp4` | 0.987000 | 4935 | 65 |
| `sparse_nvfp4` | 0.768600 | 3842 | 1158 |
| `marlin_weight_only` | 0.987600 | 4938 | 62 |
| `dense_nvfp4_prefill_marlin_decode` | 0.986800 | 4934 | 66 |

## Recommended Sample Sizes

The table below shows the minimum sample size N where the error metric drops below the threshold. "Max error" is the worst-case across all 30 seeds; "mean error" is the average.

| Method | Full Acc | N for max err ≤0.5% | N for max err ≤1% | N for max err ≤2% | N for mean err ≤0.5% | N for mean err ≤1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_bf16` | 0.9864 | 1500 | 750 | 150 | 300 | 75 |
| `sparse_bf16` | 0.9852 | 1500 | 500 | 150 | 300 | 150 |
| `dense_nvfp4` | 0.9870 | 3000 | 750 | 150 | 300 | 75 |
| `sparse_nvfp4` | 0.7686 | >4000 | 4000 | 1500 | 3000 | 1000 |
| `marlin_weight_only` | 0.9876 | 3000 | 750 | 150 | 300 | 75 |
| `dense_nvfp4_prefill_marlin_decode` | 0.9868 | 3000 | 500 | 150 | 300 | 75 |

## Detailed Stats (dense_bf16)

Full table for the primary baseline method:

| N | Mean Acc | Std | Min | Max | Mean Abs Err | Max Abs Err |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.986667 | 0.018856 | 0.960000 | 1.000000 | 0.017867 | 0.026400 |
| 50 | 0.984000 | 0.016653 | 0.940000 | 1.000000 | 0.014187 | 0.046400 |
| 75 | 0.985333 | 0.010527 | 0.960000 | 1.000000 | 0.007662 | 0.026400 |
| 100 | 0.986333 | 0.009481 | 0.960000 | 1.000000 | 0.007720 | 0.026400 |
| 150 | 0.988889 | 0.007166 | 0.973333 | 1.000000 | 0.005938 | 0.013600 |
| 200 | 0.987833 | 0.006283 | 0.975000 | 1.000000 | 0.005460 | 0.013600 |
| 300 | 0.986444 | 0.005769 | 0.976667 | 0.996667 | 0.004756 | 0.010267 |
| 500 | 0.985733 | 0.004343 | 0.976000 | 0.994000 | 0.003387 | 0.010400 |
| 750 | 0.986311 | 0.002731 | 0.981333 | 0.990667 | 0.002364 | 0.005067 |
| 1000 | 0.985967 | 0.002938 | 0.979000 | 0.992000 | 0.002353 | 0.007400 |
| 1500 | 0.986244 | 0.002133 | 0.982000 | 0.991333 | 0.001658 | 0.004933 |
| 2000 | 0.986133 | 0.001821 | 0.982500 | 0.990000 | 0.001400 | 0.003900 |
| 3000 | 0.985978 | 0.001145 | 0.984333 | 0.989000 | 0.001018 | 0.002600 |
| 4000 | 0.986225 | 0.000789 | 0.984750 | 0.987750 | 0.000692 | 0.001650 |

## Plots

- `outputs/<method>/accuracy_vs_samples.png` — per-method accuracy convergence with ±1σ/±2σ bands
- `summary/all_methods_comparison.png` — all methods overlaid
- `summary/error_vs_samples.png` — mean absolute error vs N for all methods
- `summary/error_vs_percentage.png` — mean absolute error (%) vs sample size as % of full dataset

## Interpretation

The accuracy converges quickly because the model is highly accurate (~98.6% for most methods). With only ~100 samples, the estimate is already within ±2-3% of the true accuracy most of the time. By ~500 samples, the error is typically under ±1%.

For the outlier `sparse_nvfp4` method (~76.9% accuracy), convergence is even faster in absolute terms because the variance of a binomial proportion is maximal at p=0.5 and decreases as p approaches 0 or 1.

**Recommendation**: 500-1000 samples provides a good balance between speed and accuracy (error < ±1%). For quick smoke tests, 200 samples gives a reasonable estimate (error < ±2%).
