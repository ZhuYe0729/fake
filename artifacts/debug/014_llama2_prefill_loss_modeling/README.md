# Llama2 Prefill-Only Loss Modeling

This experiment collects data for prefill-only precision modeling before building another Pareto optimizer.

## Validity Note

This experiment is **not valid for final prefill NVFP4 quality modeling**.
The scripts replace selected `nn.Linear` weights with prepared dense tensors and
evaluate them through PyTorch `F.linear`. That path does not run the real
`dense_nvfp4` / `sparse_nvfp4` CUTLASS kernels and therefore misses runtime
activation quantization. In real prefill inference, `dense_nvfp4` and
`sparse_nvfp4` quantize activations in the kernel, so their quality impact must
be measured with kernel-installed modules.

The current results are only useful as an offline weight-error diagnostic. They
must not be used as the final Pareto quality model for NVFP4 prefill.

## Goal

Measure whether local linear error proxies can explain WikiText-2 mean CE loss deltas under real compressed weights.

## Methods

- `dense_nvfp4`
- `sparse_bf16`
- `sparse_nvfp4`
- `marlin_nvfp4`

`dense_bf16` is used as the loss baseline. Hybrid prefill+decode is intentionally excluded because this stage has no decode phase.

## Recommended Full Run

Local error collection:

```bash
CUDA_VISIBLE_DEVICES=7 python fake/artifacts/debug/014_llama2_prefill_loss_modeling/scripts/collect_local_errors.py \
  --output-root fake/artifacts/debug/014_llama2_prefill_loss_modeling \
  --gpu 0 \
  --methods dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4 \
  --calib-samples 128 \
  --seq-len 512 \
  --batch-size 1 \
  --module-chunk-size 8
```

Loss ablation can be sharded by method:

```bash
CUDA_VISIBLE_DEVICES=7 python fake/artifacts/debug/014_llama2_prefill_loss_modeling/scripts/run_loss_ablation.py --output-root fake/artifacts/debug/014_llama2_prefill_loss_modeling --gpu 0 --methods dense_nvfp4 --calib-samples 128 --seq-len 512 --batch-size 1
CUDA_VISIBLE_DEVICES=6 python fake/artifacts/debug/014_llama2_prefill_loss_modeling/scripts/run_loss_ablation.py --output-root fake/artifacts/debug/014_llama2_prefill_loss_modeling --gpu 0 --methods sparse_bf16 --calib-samples 128 --seq-len 512 --batch-size 1
CUDA_VISIBLE_DEVICES=5 python fake/artifacts/debug/014_llama2_prefill_loss_modeling/scripts/run_loss_ablation.py --output-root fake/artifacts/debug/014_llama2_prefill_loss_modeling --gpu 0 --methods sparse_nvfp4 --calib-samples 128 --seq-len 512 --batch-size 1
CUDA_VISIBLE_DEVICES=4 python fake/artifacts/debug/014_llama2_prefill_loss_modeling/scripts/run_loss_ablation.py --output-root fake/artifacts/debug/014_llama2_prefill_loss_modeling --gpu 0 --methods marlin_nvfp4 --calib-samples 128 --seq-len 512 --batch-size 1
```

Summarize after all shards finish:

```bash
MPLCONFIGDIR=/tmp/mplconfig python fake/artifacts/debug/014_llama2_prefill_loss_modeling/scripts/summarize_prefill_loss.py \
  --output-root fake/artifacts/debug/014_llama2_prefill_loss_modeling
```

## Expected Full Outputs

- `sensitivity/module_features.csv`
- `sensitivity/module_method_local_errors.csv`
- `ablations/loss_ablation_<method>.csv`
- `summary/prefill_loss_modeling/README.md`
- `summary/prefill_loss_modeling/local_error_loss_correlations.csv`
- `summary/prefill_loss_modeling/layer_loss_summary.csv`
- `summary/prefill_loss_modeling/linear_type_loss_summary.csv`
- `summary/prefill_loss_modeling/*.png`

Expected full row counts:

- local error rows: `224 * 4 = 896`
- loss ablation rows across shards: `4 * (1 dense baseline + 263 policies) = 1056` before dedup; summary deduplicates dense baseline by `(method, policy)`

## Smoke Status

The smoke run under `smoke/` validates:

- local error collection on 2 modules and 1 method
- loss ablation on 2 policies and 1 method
- summary/report generation
