# Llama-3.1-8B-Instruct prefill-only: measured task Pareto

All speed values are 061 warmed phase-vLLM medians (B=8, L=2048, one warmup + five timed requests). `ours` task scores are newly measured with real phase-vLLM canonical checkpoints. Uniform task scores are the existing frozen 058 measurements; p00 is identical to 061 point_000.

| family | policy | speed (ms) | speedup | WikiText PPL ↓ | Winogrande ↑ | ARC-Easy ↑ | ARC-Challenge ↑ | MMLU ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | w4a16_ours | 1114.73 | 0.98x | 9.9351 | 0.7309 | 0.8093 | 0.5367 | 0.6620 |
| uniform | dense_bf16 | 1088.93 | 1.00x | 9.4250 | 0.7348 | 0.8199 | 0.5563 | 0.6840 |
| uniform | sparse_bf16 | 708.17 | 1.54x | 21.4360 | 0.6606 | 0.6818 | 0.3874 | 0.4296 |
| uniform | dense_nvfp4 | 618.84 | 1.76x | 10.3757 | 0.7206 | 0.7929 | 0.5111 | 0.6350 |
| uniform | sparse_nvfp4 | 564.24 | 1.93x | 65.0020 | 0.5328 | 0.4566 | 0.2637 | 0.2388 |
| ours | point_000 | 1087.53 | 1.00x | 9.4250 | 0.7348 | 0.8199 | 0.5563 | 0.6840 |
| ours | point_003 | 965.88 | 1.13x | 9.6055 | 0.7403 | 0.8182 | 0.5503 | 0.6848 |
| ours | point_005 | 754.02 | 1.44x | 9.7967 | 0.7419 | 0.8035 | 0.5256 | 0.6713 |
| ours | point_007 | 685.87 | 1.59x | 10.1009 | 0.7380 | 0.7908 | 0.5145 | 0.6650 |
| ours | point_009 | 569.50 | 1.91x | 12.2887 | 0.7064 | 0.7782 | 0.4923 | 0.5593 |
| ours | point_011 | 514.47 | 2.11x | 15.2804 | 0.6725 | 0.7184 | 0.4061 | 0.4182 |
| ours | point_013 | 482.08 | 2.26x | 26.7803 | 0.6322 | 0.6326 | 0.3413 | 0.3266 |
| ours | point_014 | 481.16 | 2.26x | 31.9439 | 0.6109 | 0.6128 | 0.3302 | 0.3114 |

## Suggested points

- High-quality: `point_005` (1.44x; close to BF16 on all five tasks).
- Balanced / speed-first: `point_009` (1.91x; faster than uniform dense-NVFP4 and vastly higher task quality than uniform sparse-NVFP4).
- Max-speed: `point_014` (2.26x; report as an explicit quality-sacrificing endpoint).
