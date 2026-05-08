# 003 NVFP4 Quantization And Pruning Pipeline

## Summary
实现统一的模型压缩 pipeline，支持 MaxViT 和 DINOv3 ViT-7B 的 NVFP4 量化、非结构化剪枝、半结构化剪枝，以及 joint 方法。第一版产物使用 fake-quant / pruned dequantized checkpoint，保存 mask/scale/配置元数据，可直接复用现有 ImageNet accuracy 和 forward speed 脚本验证。

## Compression Matrix
| Model | Method | Pruning Algo | Quant | Modules | Calibration | Key Config |
|---|---|---:|---:|---|---:|---|
| MaxViT | dense | none | none | none | none | existing baseline |
| MaxViT | nvfp4 | none | NVFP4 | attention/MLP Linear + MBConv pointwise Conv | 128 images | group_size=16, scale=fp16, remap=none |
| MaxViT | unstructured_sparse | OBC-style diag Hessian | none | same selected modules | 128 images | sparsity=0.5 |
| MaxViT | semi_structured_sparse | OBC-style diag Hessian | none | same selected modules | 128 images | dense 2:4, prune 2 of every 4 consecutive weights |
| MaxViT | nvfp4_unstructured_sparse | OBC-style diag Hessian | NVFP4 | same selected modules | 128 images | prune first, sparsity=0.5, group_size=16 |
| MaxViT | nvfp4_semi_structured_sparse | OBC-style diag Hessian | NVFP4 | same selected modules | 128 images | prune first, pair-wise 2:4 over 8 weights, group_size=32 |
| DINOv3 ViT-7B | dense | none | none | none | none | existing baseline |
| DINOv3 ViT-7B | nvfp4 | none | NVFP4 | backbone q/k/v/o + MLP gate/up/down Linear | 16 images | group_size=16, scale=fp16, remap=none |
| DINOv3 ViT-7B | unstructured_sparse | SparseGPT-style diag Hessian | none | same selected modules | 16 images | sparsity=0.5 |
| DINOv3 ViT-7B | semi_structured_sparse | SparseGPT-style diag Hessian | none | same selected modules | 16 images | dense 2:4, prune 2 of every 4 consecutive weights |
| DINOv3 ViT-7B | nvfp4_unstructured_sparse | SparseGPT-style diag Hessian | NVFP4 | same selected modules | 16 images | prune first, sparsity=0.5, group_size=16 |
| DINOv3 ViT-7B | nvfp4_semi_structured_sparse | SparseGPT-style diag Hessian | NVFP4 | same selected modules | 16 images | prune first, pair-wise 2:4 over 8 weights, group_size=32 |

## Implementation Notes
- 压缩生成入口：`scripts/prepare_compressed_model.py`。
- 输出：`artifacts/checkpoints/{model}/{method}/model.pt`、`metadata.json`、`masks.pt`、`scales.pt`。
- 默认 `masks.pt` 和 `scales.pt` 存 metadata-only，避免 DINOv3 7B 产生数 GB 级辅助文件；如需完整张量可用 `--save-full-masks` / `--save-full-scales`。
- joint 方法固定为 prune first, then NVFP4 quantize。
- 不实现 kernel-ready uint4 packing，后续 CUTLASS 接口稳定后再加。

## Module Policy
- MaxViT 默认压缩 attention/MLP `Linear`，以及 MBConv 中 `conv1_1x1` 和 `conv3_1x1` pointwise `Conv2d`。
- MaxViT 默认不压缩 stem、depthwise conv、SE、shortcut、norm、classifier head。
- DINOv3 默认只压缩 backbone transformer block 的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。
- DINOv3 默认不压缩 patch embedding、norm、layer scale、ImageNet linear head。

## Test Plan
- 静态验证：`python -m compileall fake scripts`，Slurm 脚本 `bash -n`。
- 小样本 smoke：每个模型各用 2 个 calibration samples 生成 `nvfp4` checkpoint，验证 checkpoint 可加载并跑 1 个 batch accuracy。
- 正式验证：按 matrix 生成 checkpoint，分别跑 ImageNet accuracy CSV 和 forward-only speed CSV。

