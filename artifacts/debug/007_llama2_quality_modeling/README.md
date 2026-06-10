# 007 Llama2 Quality Modeling

Experimental quality-modeling workspace for Llama2-7B compression policies. The code is intentionally isolated from the main framework and reuses the real compression artifacts from `artifacts/results/main/003_llama2_7b_arc_easy_accuracy`.

## Default workflow

Run a small smoke pass first:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/collect_sensitivity.py --gpu 6 --calib-samples 2 --seq-len 128 --max-modules 4
CUDA_VISIBLE_DEVICES=6 python scripts/run_ablation.py --gpu 6 --calib-samples 2 --seq-len 128 --policies all,family:attention --methods dense_nvfp4 --arc-limit 4
python scripts/summarize_quality.py
```

Run the intended first-pass experiment:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/collect_sensitivity.py --gpu 6 --calib-samples 32 --seq-len 512
CUDA_VISIBLE_DEVICES=6 python scripts/run_ablation.py --gpu 6 --calib-samples 32 --seq-len 512 --arc-limit 128
python scripts/summarize_quality.py
```

Optional full ARC-Easy spot check for the densest policies:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_ablation.py \
  --gpu 7 \
  --calib-samples 32 \
  --seq-len 512 \
  --methods dense_nvfp4,sparse_bf16,sparse_nvfp4 \
  --policies all \
  --full-arc \
  --output-root full_arc_selected \
  --source-root ../../results/main/003_llama2_7b_arc_easy_accuracy
```

Run the ARC-Challenge supplement with the same sensitivity/proxy features:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_ablation.py \
  --gpu 7 \
  --calib-samples 32 \
  --seq-len 512 \
  --task arc_challenge \
  --arc-limit 128 \
  --output-root arc_challenge_limit128 \
  --source-root ../../results/main/003_llama2_7b_arc_easy_accuracy

cp sensitivity/module_features.csv arc_challenge_limit128/sensitivity/module_features.csv
cp sensitivity/module_method_errors.csv arc_challenge_limit128/sensitivity/module_method_errors.csv
python scripts/summarize_quality.py --output-root arc_challenge_limit128
```

## Outputs

- `sensitivity/module_features.csv`: dense module statistics.
- `sensitivity/module_method_errors.csv`: local output error for each module and compression method.
- `ablations/policy_quality_results.csv`: mixed-policy NLL and optional `arc_easy` results.
- `summary/recommended_proxy_formula.json`: first-pass quality proxy.
- `summary/analysis.md`: compact analysis notes.
- `summary/policy_proxy_scores.csv`: policy-level proxy score joined with NLL and `arc_easy`.
- `summary/proxy_correlation.csv`: Pearson/Spearman correlation between proxy score and quality metrics.
- `full_arc_selected/ablations/policy_quality_results.csv`: full ARC-Easy selected-policy check.
- `arc_challenge_limit128/`: ARC-Challenge limit=128 supplement with the same 115 policy rows.

## Completed Run Notes

The completed first-pass run used GPU7, `calib_samples=32`, `seq_len=512`, and `arc_limit=128`.
It produced 224 module feature rows, 672 module-method local error rows, and 115 policy rows.

The optional full ARC-Easy spot check produced 4 rows:

- dense_bf16 none: acc=0.7554713804713805, acc_norm=0.7382154882154882
- dense_nvfp4 all: acc=0.7563131313131313, acc_norm=0.7331649831649831
- sparse_bf16 all: acc=0.648989898989899, acc_norm=0.5925925925925926
- sparse_nvfp4 all: acc=0.37037037037037035, acc_norm=0.3602693602693603

The ARC-Challenge limit=128 supplement produced 115 policy rows. Baseline dense_bf16 was
`acc=0.40625`, `acc_norm=0.4609375`.

All-layer policy results:

- dense_nvfp4 all: acc=0.40625, acc_norm=0.4609375, nll_delta=0.03679563294651489
- sparse_bf16 all: acc=0.34375, acc_norm=0.359375, nll_delta=0.3506198442612143
- sparse_nvfp4 all: acc=0.1875, acc_norm=0.265625, nll_delta=1.0675357848464162

ARC-Challenge proxy correlations:

- dense_nvfp4 vs arc_acc_delta: pearson=-0.3269187885556433, spearman=-0.2131867636009144
- sparse_bf16 vs arc_acc_delta: pearson=-0.5346319948034788, spearman=-0.2283624147306115
- sparse_nvfp4 vs arc_acc_delta: pearson=-0.861222836961487, spearman=-0.6887931604848702
- dense_nvfp4 vs nll_delta: pearson=0.7782096806508191, spearman=0.7645256592625013
- sparse_bf16 vs nll_delta: pearson=0.957422545074862, spearman=0.9113688587372796
- sparse_nvfp4 vs nll_delta: pearson=0.9233793974433534, spearman=0.9203413940256043
