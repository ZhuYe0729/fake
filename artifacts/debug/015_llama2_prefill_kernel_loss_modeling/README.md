# Llama2 Kernel-Aware Prefill Loss Modeling

This experiment fixes the invalid NVFP4 prefill quality path from `014_llama2_prefill_loss_modeling`.

## Goal

Measure `dense_nvfp4` and `sparse_nvfp4` prefill quality with real runtime kernels. The model weights come from the existing calibrated compressed checkpoints under:

`fake/artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared/`

No model compression is rerun here.

## Validity

The scripts install selected real kernel modules before forward:

- `dense_nvfp4`: CUTLASS `NVFP4Linear`
- `sparse_nvfp4`: CUTLASS sparse NVFP4 wrapped by `PaddedSparseNVFP4Linear`

Therefore runtime activation quantization is included. This is the intended prefill-only quality path for NVFP4 methods.

## Smoke

```bash
CUDA_VISIBLE_DEVICES=7 python fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling/scripts/collect_kernel_local_errors.py \
  --output-root fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling/smoke \
  --gpu 0 \
  --methods dense_nvfp4,sparse_nvfp4 \
  --calib-samples 4 \
  --seq-len 128 \
  --batch-size 1 \
  --max-modules 2 \
  --module-chunk-size 1

CUDA_VISIBLE_DEVICES=7 python fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling/scripts/run_kernel_loss_ablation.py \
  --output-root fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling/smoke \
  --gpu 0 \
  --methods dense_nvfp4,sparse_nvfp4 \
  --calib-samples 4 \
  --seq-len 128 \
  --batch-size 1 \
  --max-modules 2 \
  --max-policies 3
```

## Full Run

Local error:

```bash
CUDA_VISIBLE_DEVICES=7 python fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling/scripts/collect_kernel_local_errors.py \
  --output-root fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling \
  --gpu 0 \
  --methods dense_nvfp4,sparse_nvfp4 \
  --calib-samples 128 \
  --seq-len 512 \
  --batch-size 1 \
  --module-chunk-size 4
```

Loss ablation shards:

```bash
CUDA_VISIBLE_DEVICES=6 python fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling/scripts/run_kernel_loss_ablation.py \
  --output-root fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling \
  --gpu 0 \
  --methods dense_nvfp4 \
  --calib-samples 128 \
  --seq-len 512 \
  --batch-size 1

CUDA_VISIBLE_DEVICES=5 python fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling/scripts/run_kernel_loss_ablation.py \
  --output-root fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling \
  --gpu 0 \
  --methods sparse_nvfp4 \
  --calib-samples 128 \
  --seq-len 512 \
  --batch-size 1
```

Summarize:

```bash
MPLCONFIGDIR=/tmp/mplconfig python fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling/scripts/summarize_kernel_prefill_loss.py \
  --output-root fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling
```

## Expected Outputs

- `sensitivity/module_features.csv`: 224 rows
- `sensitivity/module_method_kernel_local_errors.csv`: 448 rows
- `ablations/kernel_loss_ablation_dense_nvfp4.csv`: 264 rows
- `ablations/kernel_loss_ablation_sparse_nvfp4.csv`: 264 rows
- `summary/kernel_prefill_loss_modeling/README.md`
- `summary/kernel_prefill_loss_modeling/*.csv`
- `summary/kernel_prefill_loss_modeling/*.png`

Expected wall-clock on the current machine is roughly 1-2 hours for the full run with three GPUs.
