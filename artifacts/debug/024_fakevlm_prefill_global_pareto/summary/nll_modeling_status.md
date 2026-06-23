# NLL Modeling Status

> **Invalidated on 2026-06-21:** the v1 NLL rows below used incorrect left-padding/image-token label alignment. They are retained for audit only and must be regenerated with `assistant_answer_token_nll_v2_active_prefix_aligned`. Speed-only results remain valid. See `prediction_vs_actual/QUALITY_MODEL_INVALID_NOTICE.md`.

## Completed

- `quality/stratified_loss.csv` contains 61/61 stratified policies.
- Dense teacher-forcing assistant-answer NLL: `13.8090735314`.
- Raw NLL delta range vs dense: `[-0.6661583378, 0.3271658372]`; 18 policies have positive raw delta.
- `fit_quality_model.py` was rerun from `quality/stratified_loss.csv`; RMSE on clipped nonnegative NLL delta target is `0.0864615469`.
- Cost tables, Pareto frontiers, and selected representative policies were rebuilt for batch sizes `1,2,4,8,16`.
- New selected validation points per batch: `0,5,9,13,18,22,26,30`.

## Validation Complete

- `validation/pareto_speed_validation.csv` contains 40/40 selected policies.
- `quality/validation_quality.csv` contains 40/40 selected policies.
- `validation/pareto_validation_joined.csv`, `summary/analysis.md`, and `report/` were regenerated after validation completed.

## Notes

- Old validation CSVs from the previous accuracy-targeted Pareto run were archived under `validation/archive_20260620_165728/`.
- Current `report/` figures correspond to the NLL/loss-targeted quality model and the regenerated selected policies.
