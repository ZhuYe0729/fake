# 固定主配置的代表负载迁移

Fixed standard-load P+D maps transferred to three representative loads. No per-load quality guarantee or test-based retuning. Same GPU0/NUMA0,6GiB KV,decode Graph,includes TTFT/TPOT/E2E/restore. Two reverse-order campaigns with BF16 brackets.

以下使用标准负载已冻结的P+D配置；没有为新负载重新优化或提供新负载的质量保证。

| Load | Method | TTFT × | TPOT × | E2E × | E2E campaign range | Restore ms |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| single_b1_s2048_o128 | BF16 | 1.00 | 1.00 | 1.00 | 1.00–1.00 | 0.16 |
| single_b1_s2048_o128 | W4A16 NVFP4 | 1.13 | 1.99 | 1.85 | 1.85–1.85 | 0.16 |
| single_b1_s2048_o128 | W4A16 INT4 Marlin | 1.14 | 2.03 | 1.88 | 1.87–1.88 | 0.14 |
| single_b1_s2048_o128 | SmoothQuant INT8 | 2.29 | 1.40 | 1.46 | 1.46–1.46 | 0.14 |
| single_b1_s2048_o128 | HAWQ-v3 INT4, latency target=.80 BF16 (PTQ) | 1.48 | 1.24 | 1.26 | 1.26–1.26 | 0.15 |
| single_b1_s2048_o128 | HAWQ-v3 FP4, latency target=.80 BF16 (PTQ) | 1.49 | 1.22 | 1.25 | 1.24–1.25 | 11.47 |
| single_b1_s2048_o128 | HAWQ FP4 minimum-cost control, q=.07 | 2.71 | 1.65 | 1.72 | 1.72–1.72 | 50.18 |
| single_b1_s2048_o128 | DASO, 5 candidates, fresh-profile DP | 2.16 | 2.02 | 2.04 | 2.03–2.04 | 105.06 |
| single_b1_s2048_o128 | DASO + 8bit, fresh-profile DP | 1.99 | 1.51 | 1.55 | 1.54–1.56 | 78.55 |
| single_b1_s2048_o128 | DASO + 8bit + exact/validation, epsilon=.01 | 2.10 | 1.34 | 1.39 | 1.39–1.40 | 77.44 |
| single_b1_s2048_o128 | DASO + 8bit + exact/validation, epsilon=.04 | 2.22 | 1.60 | 1.65 | 1.63–1.67 | 101.53 |
| decode_b8_s512_o128 | BF16 | 1.00 | 1.00 | 1.00 | 1.00–1.00 | 0.14 |
| decode_b8_s512_o128 | W4A16 NVFP4 | 1.09 | 2.03 | 1.79 | 1.78–1.79 | 0.11 |
| decode_b8_s512_o128 | W4A16 INT4 Marlin | 1.09 | 2.05 | 1.80 | 1.79–1.81 | 0.15 |
| decode_b8_s512_o128 | SmoothQuant INT8 | 2.33 | 1.45 | 1.54 | 1.52–1.56 | 0.15 |
| decode_b8_s512_o128 | HAWQ-v3 INT4, latency target=.80 BF16 (PTQ) | 1.50 | 1.25 | 1.29 | 1.29–1.29 | 0.17 |
| decode_b8_s512_o128 | HAWQ-v3 FP4, latency target=.80 BF16 (PTQ) | 1.51 | 1.23 | 1.27 | 1.27–1.27 | 9.78 |
| decode_b8_s512_o128 | HAWQ FP4 minimum-cost control, q=.07 | 2.75 | 1.66 | 1.77 | 1.76–1.78 | 48.83 |
| decode_b8_s512_o128 | DASO, 5 candidates, fresh-profile DP | 2.24 | 1.99 | 2.02 | 2.01–2.03 | 109.39 |
| decode_b8_s512_o128 | DASO + 8bit, fresh-profile DP | 1.99 | 1.46 | 1.53 | 1.53–1.53 | 86.43 |
| decode_b8_s512_o128 | DASO + 8bit + exact/validation, epsilon=.01 | 2.06 | 1.44 | 1.52 | 1.50–1.53 | 79.24 |
| decode_b8_s512_o128 | DASO + 8bit + exact/validation, epsilon=.04 | 2.31 | 1.65 | 1.73 | 1.73–1.73 | 109.24 |
| long_b8_s4096_o64 | BF16 | 1.00 | 1.00 | 1.00 | 1.00–1.00 | 0.14 |
| long_b8_s4096_o64 | W4A16 NVFP4 | 0.98 | 1.69 | 1.11 | 1.11–1.11 | 0.17 |
| long_b8_s4096_o64 | W4A16 INT4 Marlin | 0.98 | 1.71 | 1.12 | 1.12–1.12 | 0.14 |
| long_b8_s4096_o64 | SmoothQuant INT8 | 1.79 | 1.35 | 1.64 | 1.63–1.64 | 0.14 |
| long_b8_s4096_o64 | HAWQ-v3 INT4, latency target=.80 BF16 (PTQ) | 1.34 | 1.21 | 1.30 | 1.29–1.30 | 0.14 |
| long_b8_s4096_o64 | HAWQ-v3 FP4, latency target=.80 BF16 (PTQ) | 1.36 | 1.19 | 1.30 | 1.30–1.31 | 5.02 |
| long_b8_s4096_o64 | HAWQ FP4 minimum-cost control, q=.07 | 2.04 | 1.49 | 1.85 | 1.85–1.85 | 36.68 |
| long_b8_s4096_o64 | DASO, 5 candidates, fresh-profile DP | 1.86 | 1.69 | 1.81 | 1.80–1.81 | 103.05 |
| long_b8_s4096_o64 | DASO + 8bit, fresh-profile DP | 1.71 | 1.37 | 1.60 | 1.59–1.60 | 70.89 |
| long_b8_s4096_o64 | DASO + 8bit + exact/validation, epsilon=.01 | 1.62 | 1.27 | 1.50 | 1.47–1.54 | 71.23 |
| long_b8_s4096_o64 | DASO + 8bit + exact/validation, epsilon=.04 | 1.88 | 1.49 | 1.75 | 1.74–1.75 | 100.25 |
