# Llama3.1 prefill-only uniform-coverage refinement

Target: speedup >= 1.7574x and ARC-Challenge >= 0.5111.

## New measured candidates

| policy | speedup | delta NLL | WikiText PPL | WinoGrande | ARC-Easy | ARC-Challenge | MMLU |
|---|---:|---:|---:|---:|---:|---:|---:|
| point_004 | 1.2523 | — | — | — | — | 0.5461 | — |
| low_q046 | 1.2862 | — | — | — | — | 0.5410 | — |
| low_q050 | 1.3333 | — | — | — | — | 0.5341 | — |
| low_q054 | 1.3573 | — | — | — | — | 0.5265 | — |
| low_q062 | 1.4373 | — | — | — | — | 0.5341 | — |
| bridge_dense_nvfp4_088 | 1.4623 | — | — | — | — | 0.5324 | — |
| point_006 | 1.5042 | — | — | — | — | 0.5324 | — |
| fine_q124 | 1.5911 | — | — | — | — | 0.5179 | — |
| bridge_dense_nvfp4_096 | 1.6590 | — | — | — | — | 0.5162 | — |
| fine_q160 | 1.6662 | — | — | — | — | 0.5179 | — |
| point_008 | 1.7124 | — | — | — | — | 0.5026 | — |
| bridge_dense_nvfp4_120 | 1.7338 | — | — | — | — | 0.5299 | — |
| fine_q190 | 1.7342 | — | — | — | — | 0.5179 | — |
| fine_q196 | 1.7532 | 0.1510 | — | — | — | 0.4991 | — |
| fine_q202 | 1.7741 | 0.1591 | 11.3051 | 0.7143 | 0.7845 | 0.5128 | 0.5934 |
| fine_q208 | 1.7955 | 0.1917 | 11.6346 | 0.7403 | 0.8005 | 0.5205 | 0.5570 |
| fine_q214 | 1.8168 | 0.1986 | — | — | — | 0.5094 | — |

## Uniform coverage

| uniform | covered | example ours |
|---|---:|---|
| dense_bf16 | True | point_000 |
| dense_nvfp4 | True | fine_q208 |
| sparse_bf16 | True | point_011 |
| sparse_nvfp4 | True | point_014 |
| w4a16_ours | True | low_q046 |

## Dense-NVFP4 coverage by metric

| metric | covered | example ours |
|---|---:|---|
| WikiText word perplexity | False | — |
| WinoGrande accuracy | True | fine_q208 |
| ARC-Easy accuracy | True | fine_q208 |
| ARC-Challenge normalized accuracy | True | fine_q208 |
| MMLU accuracy | False | — |

ARC-Challenge is the pre-selected plotting metric for this refinement. The cross-metric audit is reported to expose, rather than hide, metric-specific failures.

All attempted candidates are retained in `all_results.csv`.

## Paper figures

Matched Llama3.1 figures for prefill-only and prefill-decode are under [`paper/`](paper/). The combined figure uses ARC-Challenge for prefill-only and IWSLT SacreBLEU for prefill-decode; CNN/DM ROUGE-L is retained as an appendix alternative.
