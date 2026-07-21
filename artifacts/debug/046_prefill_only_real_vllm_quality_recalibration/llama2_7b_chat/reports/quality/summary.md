# llama2 real-vLLM prefill quality calibration

The labels are direct vLLM prompt-logprob NLL over 100 fixed WikiText blocks; local-error features are retained from the prior model.

## Metrics

| split | MAE | RMSE | signed error | Spearman |
|---|---:|---:|---:|---:|
| train | 0.107590 | 0.267225 | -0.000000 | 0.9097 |
| holdout | 0.121419 | 0.149305 | -0.054201 | 0.8204 |

## Per-policy predictions

| policy | split | kind | measured ΔNLL | predicted ΔNLL | residual |
|---|---|---|---:|---:|---:|
| p00 | train | uniform | 0.000000 | -0.012222 | 0.012222 |
| p01 | train | uniform | 0.053822 | 0.117222 | -0.063400 |
| p02 | train | uniform | 0.345707 | 0.811699 | -0.465992 |
| p03 | train | uniform | 1.147809 | 1.752673 | -0.604863 |
| p04 | train | uniform | 0.037336 | 0.120826 | -0.083490 |
| p05 | train | controlled | 0.001727 | -0.004483 | 0.006210 |
| p06 | train | controlled | 0.006031 | 0.058398 | -0.052366 |
| p07 | train | controlled | 0.110048 | 0.181949 | -0.071900 |
| p08 | train | controlled | 0.007755 | 0.003527 | 0.004228 |
| p09 | train | controlled | -0.000725 | -0.008387 | 0.007662 |
| p10 | train | controlled | -0.002704 | 0.039987 | -0.042691 |
| p11 | train | controlled | 0.146264 | 0.069134 | 0.077130 |
| p12 | train | controlled | 0.002031 | -0.003390 | 0.005421 |
| p13 | train | controlled | 0.000201 | -0.007770 | 0.007970 |
| p14 | train | controlled | 0.006523 | 0.035379 | -0.028856 |
| p15 | train | controlled | 0.088622 | 0.079047 | 0.009575 |
| p16 | train | controlled | 0.001754 | -0.003561 | 0.005314 |
| p17 | train | controlled | 0.002791 | -0.006848 | 0.009640 |
| p18 | train | controlled | 0.005196 | 0.030843 | -0.025647 |
| p19 | train | controlled | 0.208154 | 0.086387 | 0.121767 |
| p20 | train | controlled | 0.000008 | -0.004222 | 0.004230 |
| p21 | train | controlled | 0.012874 | 0.037501 | -0.024627 |
| p22 | train | controlled | 0.457091 | 0.294895 | 0.162196 |
| p23 | train | controlled | 2.300020 | 0.625750 | 1.674270 |
| p24 | train | controlled | 0.010621 | 0.038727 | -0.028106 |
| p25 | train | controlled | 0.007264 | 0.014992 | -0.007728 |
| p26 | train | controlled | 0.179241 | 0.158549 | 0.020692 |
| p27 | train | controlled | 0.313941 | 0.350535 | -0.036594 |
| p28 | train | controlled | 0.006224 | 0.015807 | -0.009584 |
| p29 | train | controlled | 0.007948 | 0.014260 | -0.006312 |
| p30 | train | controlled | 0.121353 | 0.159703 | -0.038351 |
| p31 | train | controlled | 0.258953 | 0.360819 | -0.101865 |
| p32 | train | controlled | 0.004519 | 0.015055 | -0.010536 |
| p33 | train | controlled | 0.011591 | 0.013804 | -0.002213 |
| p34 | train | controlled | 0.298447 | 0.161887 | 0.136559 |
| p35 | train | controlled | 0.453967 | 0.378904 | 0.075063 |
| p36 | train | controlled | 0.004453 | 0.014572 | -0.010119 |
| p37 | train | balanced_mixed | 0.060804 | 0.124134 | -0.063331 |
| p38 | train | balanced_mixed | 0.067726 | 0.357576 | -0.289850 |
| p39 | train | balanced_mixed | 0.277422 | 0.278919 | -0.001497 |
| p40 | train | balanced_mixed | 0.389703 | 0.355395 | 0.034308 |
| p41 | train | balanced_mixed | 0.414577 | 0.651811 | -0.237234 |
| p42 | train | balanced_mixed | 0.066899 | 0.135353 | -0.068454 |
| p43 | train | balanced_mixed | 0.060631 | 0.044137 | 0.016494 |
| p44 | train | balanced_mixed | 0.222016 | 0.216150 | 0.005866 |
| p45 | train | balanced_mixed | 0.386928 | 0.104985 | 0.281943 |
| p46 | train | balanced_mixed | 0.503220 | 0.595587 | -0.092367 |
| p47 | train | balanced_mixed | 0.047103 | 0.207187 | -0.160084 |
| p48 | train | balanced_mixed | 0.274598 | 0.287084 | -0.012485 |
| p49 | train | balanced_mixed | 0.177905 | 0.194006 | -0.016101 |
| p50 | train | balanced_mixed | 0.316820 | 0.540606 | -0.223785 |
| p51 | train | balanced_mixed | 0.583935 | 0.372059 | 0.211876 |
| p52 | train | balanced_mixed | 0.030554 | 0.016262 | 0.014292 |
| p53 | train | balanced_mixed | 0.137941 | 0.162441 | -0.024500 |
| p54 | holdout | balanced_mixed | 0.347311 | 0.294302 | 0.053008 |
| p55 | holdout | balanced_mixed | 0.226532 | 0.499568 | -0.273036 |
| p56 | holdout | balanced_mixed | 0.489873 | 0.704969 | -0.215095 |
| p57 | holdout | balanced_mixed | 0.029965 | 0.101351 | -0.071386 |
| p58 | holdout | balanced_mixed | 0.050684 | 0.201657 | -0.150973 |
| p59 | holdout | balanced_mixed | 0.193000 | 0.356147 | -0.163146 |
| p60 | holdout | balanced_mixed | 0.592106 | 0.258914 | 0.333192 |
| p61 | holdout | balanced_mixed | 0.385909 | 0.563665 | -0.177757 |
| p62 | holdout | balanced_mixed | 0.209215 | 0.241517 | -0.032303 |
| p63 | holdout | balanced_mixed | 0.053286 | 0.238561 | -0.185275 |
| p64 | holdout | balanced_mixed | 0.134957 | 0.240882 | -0.105925 |
| p65 | holdout | balanced_mixed | 0.202987 | 0.258244 | -0.055257 |
| p66 | holdout | balanced_mixed | 0.514884 | 0.579015 | -0.064131 |
| p67 | holdout | balanced_mixed | 0.061780 | 0.015784 | 0.045996 |
| p68 | holdout | balanced_mixed | 0.133855 | 0.102093 | 0.031762 |
| p69 | holdout | balanced_mixed | 0.087043 | 0.133734 | -0.046691 |
| p70 | holdout | balanced_mixed | 0.409963 | 0.449570 | -0.039607 |
| p71 | holdout | balanced_mixed | 0.551819 | 0.410821 | 0.140998 |
