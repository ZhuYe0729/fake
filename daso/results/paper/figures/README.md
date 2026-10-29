# Llama3.1-8B-Instruct two-scenario paper trade-off figures

- Prefill-only: ARC-Challenge `acc_norm`, sourced from debug 068.
- Prefill-decode main panel: IWSLT SacreBLEU, sourced from debug 060/057.
- CNN/DM ROUGE-L is retained as a standalone appendix alternative.
- Both axes use measured E2E speedup and real downstream-task quality.
- Only exact non-dominated ours points are drawn; dominated measurements remain in the source CSV.
- `point_007` remains in the source table but is excluded from the prefill-decode paper plot because its source speed is explicitly marked anomalous.
- Every uniform baseline is programmatically covered by an ours point in both selected plots; see `coverage_audit.csv`.
- All figures are sized for a 3.35-inch paper column; the two-panel figure is stacked vertically.
- A 3.35-inch side-by-side alternative is provided as `llama31_two_scenario_tradeoff_horizontal`.
- In the side-by-side figure, blue diamonds mark Balanced and orange stars mark Max speed.
- Direct labels use compact names: `2:4` means structured sparse and `W4A16` is the Marlin baseline.

The PNG files are convenient for inspection. Use the corresponding PDF files in the paper.
