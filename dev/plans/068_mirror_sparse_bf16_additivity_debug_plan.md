# 068 MIRROR sparse_bf16 Additivity Debug Plan

## Summary
Debug why MIRROR's additive quality model mispredicts `sparse_bf16`, especially why `uniform_sparse_bf16` is measured near or better than mixed theoretical points such as p153/p158.

## Key Changes
- Create a standalone debug artifact under `artifacts/debug/031_mirror_sparse_bf16_additivity_debug/`.
- Reuse existing MIRROR keyfix artifacts first: stratified GenImage partial quality, full supplemental/theoretical validation, uniform full validation, and keyfix cost table.
- Produce policy-level tables and plots relating sparse ratio, local error sum, predicted quality cost, measured CE/NLL, balanced accuracy, and residuals.
- Generate a concise Markdown diagnosis identifying whether additive modeling, layer/type effects, uniform/backend consistency, or speed-model terms are the main issues.

## Test Plan
- Verify output tables include existing low-error ratio, speed-ratio, type-targeted, theoretical, supplemental, and uniform sparse_bf16 policies.
- Check plots are generated without GPU.
- Validate p153/p158/p162/uniform sparse_bf16 appear in the residual report.

## Assumptions
- Focus on `sparse_bf16` only.
- Treat measured downstream CE/NLL as ground truth quality.
- Do not run new GPU validation until offline residual analysis identifies which controlled tests are worth adding.
