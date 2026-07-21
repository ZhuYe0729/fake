# Sensitivity-coverage calibration result

## Question

Can additional training policies spanning medium/high local-error modules improve the unchanged `047` Q/S/S²/S×Q real-vLLM NLL proxy?

## Protocol

- Added 24 **training-only** policies: eight quant-only, eight sparse-only, and eight mixed policies.
- Each family samples medium and high ranks of the pre-existing local-MSE table at multiple compression levels.
- All labels are direct phase-heterogeneous vLLM prompt-logprob NLL over the fixed 100×2048 WikiText blocks.
- `046` old holdout (18) and `047` mechanism holdout (6) remain frozen and are never included in the fit.
- Formula, optimizer, zero intercept, non-negative ReLU parameterization, and regularization are otherwise unchanged from `047`.

## Result

| frozen validation set | `047` before extra data | `048` after extra data |
|---|---:|---:|
| old holdout MAE / RMSE | 0.0849 / 0.1113 | 0.0879 / 0.1286 |
| old holdout Spearman | 0.8741 | 0.8060 |
| mechanism holdout MAE / RMSE | 0.2145 / 0.2273 | 0.2677 / 0.3080 |

Adding data alone does not help; it worsens both frozen validations.  The added high-rank policies themselves frequently have almost zero measured NLL delta (for example all eight quant-only additions are within about `1.4e-4`).  Thus the local relative-MSE table is not a usable calibrated absolute sensitivity axis for this real-vLLM NLL target.

## Decision

Do not use the `048` fit for Pareto solving.  The next debug must alter the precision feature itself (and validate it on the frozen holdouts), rather than adding more policies derived from the same uncalibrated local-MSE ranking.

Files: [merged labels](../nll.csv), [predictions](predictions.csv), [metrics](metrics.json).
