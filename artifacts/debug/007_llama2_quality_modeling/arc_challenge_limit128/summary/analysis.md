# Llama2 Quality Modeling Analysis

## Outputs

- `module_quality_features.csv`: per-module, per-method local error and statistics.
- `module_error_summary.csv`: grouped local error summary.
- `policy_quality_eval.csv`: mixed-policy NLL and optional arc_easy results.
- `policy_proxy_scores.csv`: mixed-policy proxy score joined with quality metrics.
- `proxy_correlation.csv`: Pearson/Spearman correlation between proxy score and quality metrics.
- `recommended_proxy_formula.json`: first-pass proxy formula.

## Current Proxy

`quality_cost = local_rel_mse * log1p(numel) * layer_weight * module_family_weight`

The proxy is a first-pass fit target; current correlations use the collected mixed-policy rows.

## Highest Local Error Groups

- sparse_nvfp4 mlp layers_16_23: mean local_rel_mse=0.12602397681403854
- sparse_nvfp4 mlp layers_08_15: mean local_rel_mse=0.11646609060096795
- sparse_nvfp4 mlp layers_24_31: mean local_rel_mse=0.11330391148803198
- sparse_nvfp4 attention layers_24_31: mean local_rel_mse=0.09764759274185063
- sparse_nvfp4 attention layers_16_23: mean local_rel_mse=0.09216606291259255
- sparse_nvfp4 mlp layers_00_07: mean local_rel_mse=0.09213313487563743
- sparse_nvfp4 attention layers_08_15: mean local_rel_mse=0.08322081246705682
- sparse_bf16 mlp layers_16_23: mean local_rel_mse=0.06378985779229297
- sparse_bf16 mlp layers_08_15: mean local_rel_mse=0.0599723943532685
- sparse_bf16 mlp layers_24_31: mean local_rel_mse=0.05643063489302415

## Proxy Correlation

- dense_nvfp4 vs nll_delta_recomputed: pearson=0.7782096806508191, spearman=0.7645256592625013, rows=38
- dense_nvfp4 vs arc_acc_delta_vs_dense: pearson=-0.3269187885556433, spearman=-0.2131867636009144, rows=38
- dense_nvfp4 vs arc_acc_norm: pearson=-0.18795840838798075, spearman=-0.2681605822026102, rows=38
- sparse_bf16 vs nll_delta_recomputed: pearson=0.957422545074862, spearman=0.9113688587372796, rows=38
- sparse_bf16 vs arc_acc_delta_vs_dense: pearson=-0.5346319948034788, spearman=-0.2283624147306115, rows=38
- sparse_bf16 vs arc_acc_norm: pearson=-0.8138529967922743, spearman=-0.6833016472806497, rows=38
- sparse_nvfp4 vs nll_delta_recomputed: pearson=0.9233793974433534, spearman=0.9203413940256043, rows=38
- sparse_nvfp4 vs arc_acc_delta_vs_dense: pearson=-0.861222836961487, spearman=-0.6887931604848702, rows=38
- sparse_nvfp4 vs arc_acc_norm: pearson=-0.8929106582297454, spearman=-0.6341035548187812, rows=38

## Policy Rows

Collected policy rows: 115
