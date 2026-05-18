# 017 CUTLASS Sparse BF16 End-to-End Plan

## Summary
为 DINOv3 ViT-7B/16 和 MaxViT 四个 variant 新增独立的 CUTLASS/cuSPARSELt sparse BF16 推理路径，复用 `fake/kernels/cutlass/cutlass_wrapper` 中已封装的 `SparseBF16Linear`。该路径使用 2:4 BF16 sparse weight，不做 NVFP4 量化；优先从已有 `semi_structured_sparse` fake checkpoint 导出 runtime-packed sparse BF16 checkpoint，再进行 end-to-end speed/accuracy。

## Key Changes
- 新增主仓库 adapter `fake/kernels/cutlass_sparse_bf16.py`：
  - 导入 wrapper 中的 `SparseBF16Linear` / `SparseBF16Weight` / `can_use_cutlass_sparse_bf16`。
  - 提供 `CutlassSparseBF16Config`、`SparseBF16ReplacementReport`、`replace_linear_with_cutlass_sparse_bf16()`、`count_cutlass_sparse_bf16_modules()`。
  - token pad multiple 默认 8；shape guard 遵循 wrapper：`out_features % 8 == 0`、`in_features % 64 == 0`、tokens multiple 8。
- 新增/扩展模型 loader：
  - DINOv3：`fake/models/dinov3_cutlass_sparse_bf16.py`，加载 dense classifier + 可选 semi-structured checkpoint 后替换 280 个 Linear。
  - MaxViT：`fake/models/maxvit_cutlass_sparse_bf16.py`，加载 dense model + 可选 semi-structured checkpoint 后替换支持的 Linear；small/base 跳过 `K=96`。
- 扩展 runtime checkpoint loader/export：
  - DINOv3 `cutlass_runtime_packed_v1` 支持 backend `sparse_bf16`。
  - MaxViT prepare/load 支持 backend `sparse_bf16`，默认输出 `artifacts/checkpoints/maxvit_<variant>/cutlass_sparse_bf16_runtime/model.pt`。
  - DINOv3 默认输出 `artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_bf16_runtime/model.pt`。
- 新增 speed/accuracy 脚本和 Slurm 入口：
  - DINOv3：`bench/eval_dinov3_vit7b16_cutlass_sparse_bf16_*`。
  - MaxViT：`bench/eval_maxvit_cutlass_sparse_bf16_*`。
  - 对应 Slurm 设置 `CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR`，并允许 runtime checkpoint / batch / output 等环境变量覆盖。

## Test Plan
- 静态检查：
  - `python3 -m py_compile` 覆盖新增 adapter、loader、prepare、bench/eval 脚本。
  - `bash -n` 覆盖新增 Slurm 脚本。
- GPU prepare：
  - DINOv3 从 `artifacts/checkpoints/dinov3_vit7b16/semi_structured_sparse/model.pt` 导出 sparse BF16 runtime checkpoint。
  - MaxViT 四个 variant 从对应 `semi_structured_sparse/model.pt` 导出 sparse BF16 runtime checkpoint。
- GPU smoke：
  - speed smoke 使用 `WARMUP=1 ITERS=2`。
  - 确认 DINOv3 replaced=280 skipped=0；MaxViT tiny/large 与 sparse NVFP4 替换计数一致，small/base 跳过 `K=96`。
- 正式验证：
  - 运行 sparse BF16 speed/accuracy，与 dense、NVFP4、sparse NVFP4 横向比较。

## Assumptions
- wrapper 的 `SparseBF16Linear` state dict 包含推理所需 `sparse_weight / metadata / bias`。
- sparse BF16 首版使用 runtime-packed checkpoint；因 cuSPARSELt compressed weight 已是运行时压缩格式，不再另做 storage/runtime 双层。
- sparse BF16 的输入 dtype 固定 BF16；权重来源推荐已有 2:4 `semi_structured_sparse` checkpoint，避免重新剪枝口径漂移。
