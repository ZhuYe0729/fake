# Draft-ready ablation text

**Latency-model ablation.** On strictly unseen (N,K) groups, direct shape fitting, a calibrated Roofline prior, and Roofline plus shape-residual correction obtain MAPE values of 14.80%, 10.78%, and 8.01%, respectively. With the refined kernel predictor fixed, PAVA monotonic E2E calibration reduces policy-level MAPE from 18.17% to 8.30%.

**Quality-model ablation.** Directly summing local perturbations gives a held-out ΔNLL MAE of 0.0571. Structure-aware weighting changes it to 0.0468, and the complete phase-aware nonnegative factorization reaches 0.0413 with a Spearman correlation of 0.963.
