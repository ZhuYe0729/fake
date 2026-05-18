## 2026-05-17 - CUTLASS sparse BF16 integration
- 开发目的：在已有 CUTLASS dense NVFP4 / sparse NVFP4 之外，新增 sparse BF16 端到端推理路径。
- 修改内容：新增 `fake/kernels/cutlass_sparse_bf16.py` adapter、DINOv3/MaxViT sparse BF16 loader、DINOv3 runtime checkpoint backend `sparse_bf16`、MaxViT sparse BF16 runtime checkpoint prepare/load、对应 speed/accuracy 脚本和 Slurm 入口。
- 影响文件：`fake/kernels/cutlass_sparse_bf16.py`、`fake/models/*sparse_bf16.py`、`fake/models/dinov3_cutlass_runtime.py`、`fake/models/maxvit_cutlass_checkpoint.py`、prepare/bench/eval 脚本、Slurm 脚本、`scripts/README.md`。
- 验证：`python3 -m py_compile` 覆盖新增/修改 Python 文件；`bash -n` 覆盖新增/修改 Slurm 脚本；`wja-cospaq` 中导入检查通过，MaxViT sparse BF16 支持计数为 tiny 88/22、small 76/34、base 180/60、large 192/48。
- 后续注意：sparse BF16 首版使用 runtime-packed checkpoint；推荐从已有 `semi_structured_sparse` checkpoint 导出并使用 `--no-prune`，避免重新剪枝口径变化。

## 2026-05-17 - Fix sparse BF16 checkpoint specs and blocked shape
- 开发目的：修复 sparse BF16 重测中出现的重复 module_specs、虚假 skipped count，以及 MaxViT small/base 的 cuSPARSELt compressed size mismatch。
- 修改内容：DINOv3/MaxViT runtime metadata 只记录外层 `PaddedSparseBF16Linear` 对应的模块名，不再记录内层 `.sparse_linear`；sparse BF16 adapter 增加 blocked shape `(out_features=96, in_features=384)`，跳过 MaxViT small/base first stage `mlp.fc2` 中会触发 cuSPARSELt compressed buffer size mismatch 的 Linear。
- 影响文件：`fake/models/dinov3_cutlass_runtime.py`、`fake/models/maxvit_cutlass_checkpoint.py`、`fake/kernels/cutlass_sparse_bf16.py`。
- 验证：`python3 -m py_compile` 通过；MaxViT sparse BF16 支持计数更新为 tiny 88/22、small 72/38、base 176/64、large 192/48，blocked shape 不再进入 specs。
- 后续注意：需要重新 prepare sparse BF16 runtime checkpoint；旧 checkpoint 的 metadata 已包含重复 `.sparse_linear` specs，不建议继续引用。

## 2026-05-17 - Refresh sparse BF16 summary plot
- 开发目的：将已完成的 sparse BF16 精度/速度结果纳入 `accuracy_compression_speed_summary.png`，修复半结构 sparse BF16 速度显示为 `NA` 的问题。
- 修改内容：汇总绘图脚本为 MaxViT/DINOv3 增加 `cutlass_sparse_bf16` accuracy/speed CSV 输入；`semi_structured_sparse` 列优先使用 sparse BF16 实测精度和速度，精度缺失时沿用已有半结构稀疏精度但补入 sparse BF16 速度。
- 影响文件：`scripts/plot_accuracy_compression_speed_summary.py`、`artifacts/results/accuracy_compression_speed_summary.csv`、`artifacts/results/accuracy_compression_speed_summary.png`。
- 验证：`python3 -m py_compile scripts/plot_accuracy_compression_speed_summary.py` 通过；`wja-cospaq` 环境下重跑绘图脚本，输出 CSV/PNG 成功，`semi_structured_sparse` speedup 不再为 `NA`。
- 后续注意：当前未找到 `artifacts/results/dinov3_vit7b16_cutlass_sparse_bf16/accuracy.csv`；DINOv3 sparse BF16 汇总图中的精度仍来自已有半结构稀疏结果，速度来自 sparse BF16 speed CSV。
