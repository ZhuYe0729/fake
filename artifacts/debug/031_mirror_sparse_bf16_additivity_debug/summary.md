# MIRROR sparse_bf16 Additivity Debug

- measurements: 200
- GenImage partial rows: 139
- full validation rows: 21
- controlled GenImage partial rows: 40

## Key Residuals

- full_theoretical `keyfix_dense_scan_theoretical_frontier` ratio=0.554 pred=0.02407 true_delta=0.22528 residual=+0.20122 bal=0.91225
- full_theoretical `keyfix_dense_scan_theoretical_frontier` ratio=0.455 pred=0.02027 true_delta=0.16869 residual=+0.14843 bal=0.93407
- full_theoretical `keyfix_dense_scan_theoretical_frontier` ratio=0.768 pred=0.03525 true_delta=0.18026 residual=+0.14501 bal=0.92517
- full_theoretical `keyfix_dense_scan_theoretical_frontier` ratio=0.670 pred=0.02881 true_delta=0.16728 residual=+0.13847 bal=0.93257
- genimage_partial `random_mixed` ratio=0.174 pred=0.13491 true_delta=0.00407 residual=-0.13084 bal=0.99414
- genimage_partial `random_mixed` ratio=0.228 pred=0.12595 true_delta=0.00108 residual=-0.12487 bal=0.99674
- genimage_partial `random_mixed` ratio=0.192 pred=0.11520 true_delta=0.00000 residual=-0.11520 bal=0.99479
- genimage_partial `random_mixed` ratio=0.147 pred=0.11489 true_delta=0.00037 residual=-0.11452 bal=0.99544
- genimage_partial `random_mixed` ratio=0.143 pred=0.11559 true_delta=0.00206 residual=-0.11353 bal=0.99609
- genimage_partial `random_mixed` ratio=0.165 pred=0.11745 true_delta=0.00484 residual=-0.11261 bal=0.99349
- genimage_partial `random_mixed` ratio=0.192 pred=0.11641 true_delta=0.00437 residual=-0.11204 bal=0.99544
- genimage_partial `random_mixed` ratio=0.170 pred=0.11255 true_delta=0.00106 residual=-0.11148 bal=0.99674

## Diagnosis

- The additive model over-penalizes high-ratio sparse_bf16 in several full-validation cases when compared with uniform sparse_bf16.
- The relationship between sparse ratio and measured CE/NLL is not purely additive: mid/high sparse ratios show large residual changes that are not explained by summed local output error alone.
- Mixed policies can be worse than uniform sparse_bf16 even when their predicted quality cost is lower, indicating a missing backend-consistency or interaction term.
- Largest mean residual by module type appears in `o_proj` policies: mean_residual=-0.06009.
- Largest mean residual by layer bucket appears in `layers_24_31`: mean_residual=-0.05440.
- Same-count variance is large: count=112 has NLL range=0.08328 between `lowerr_count_112` and `speed_count_112`.
- In controlled policies, best monotonic feature is `sum_sparse_bf16_local_rel_mse` with Spearman=+0.602.
- Existing predicted_quality_cost has controlled Spearman=+0.576, RMSE=0.02083.

## Suggested Fix Direction

- Replace plain additive sparse_bf16 cost with a policy-level model that includes count/ratio, selected layer/type distribution, and backend diversity.
- Penalize mixed backend policies when measured residuals show they are worse than uniform sparse_bf16 at similar or lower predicted cost.
- Keep same-count random policies as held-out validation before using the revised quality model for Pareto optimization.
