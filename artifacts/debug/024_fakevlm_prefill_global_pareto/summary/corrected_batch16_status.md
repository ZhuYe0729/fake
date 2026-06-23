# Corrected Batch-16 Status

## Quality Model

- Loss definition: `assistant_answer_token_nll_v2_active_prefix_aligned`.
- Stratified rows: 61/61; dense NLL: `0.6241345918`.
- Actual stratified delta signs: 54 positive, 4 zero, 3 slightly negative; minimum `-0.004286`, maximum `2.418497`.
- Fit Pearson/Spearman on stratified rows: `0.7599` / `0.8048`; RMSE: `0.2183`.
- The all-sparse-NVFP4 point remains underpredicted (`2.4185` actual vs `1.0053` predicted).

## Batch 16 Validation

- Selected policies: P0, P4, P8, P11, P15, P18, P22, P25.
- Speed, corrected NLL, and FakeClue accuracy: 8/8 each with no missing keys.
- Selected-policy NLL prediction Pearson/Spearman: `0.9978` / `0.9286`.
- Single-linear model-prediction MAPE: `6.72%`.
- E2E latency prediction MAPE: `4.17%`.
- Fastest selected point P25: `1.635x` measured E2E speedup, `0.9528` FakeClue accuracy.
- P22: `1.604x` measured E2E speedup, `0.9484` FakeClue accuracy.

## Artifacts

- Report suffix: `_corrected_nll_batch16` under `report/`.
- Prediction comparison: `prediction_vs_actual/corrected_nll_batch16/`.
- Invalid legacy artifacts: `archive_invalid_nll_20260621_142347/`.
- `025_fakevlm_pareto_search_audit` now contains 60 regenerated search policies plus 8 corrected `reference_024` policies.
