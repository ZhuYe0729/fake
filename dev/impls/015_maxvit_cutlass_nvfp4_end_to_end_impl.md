## 2026-05-16 - MaxViT CUTLASS NVFP4 end-to-end entrypoints
- 开发目的：打通 MaxViT dense、CUTLASS dense NVFP4、CUTLASS sparse NVFP4 的端到端 speed/accuracy 测试链路。
- 修改内容：新增 MaxViT CUTLASS dense/sparse loader、speed/accuracy 脚本和 Slurm 入口；保留现有 FlashInfer NVFP4 路径。
- 影响文件：`fake/models/maxvit_cutlass_nvfp4.py`、`fake/models/maxvit_cutlass_sparse_nvfp4.py`、`scripts/bench_maxvit_cutlass_*`、`scripts/eval_maxvit_cutlass_*`、对应 Slurm 和 `scripts/README.md`。
- 验证：通过新增 MaxViT CUTLASS loader/脚本的 `python -m py_compile`，通过新增/更新 Slurm 的 `bash -n`，登录节点导入检查通过。
- 后续注意：首版 CUTLASS 只替换 MaxViT 中的 Linear，compressible Conv2d 仍保持 dense；CSV 的 skipped count 会包含这些 Conv2d 跳过项。

## 2026-05-16 - MaxViT compressed checkpoint prepare/load
- 开发目的：让 MaxViT CUTLASS dense NVFP4 和 sparse NVFP4 也具备磁盘真实压缩 checkpoint 口径，避免只依赖 on-the-fly conversion。
- 修改内容：新增 `fake/models/maxvit_cutlass_checkpoint.py` 和 `scripts/prepare_maxvit_cutlass_checkpoints.py`；dense NVFP4 保存 runtime-packed checkpoint，sparse NVFP4 保存 storage checkpoint；speed/accuracy 脚本支持 `RUNTIME_CHECKPOINT` / `STORAGE_CHECKPOINT`。
- 影响文件：MaxViT CUTLASS checkpoint 模块、prepare 脚本、dense/sparse CUTLASS speed/accuracy 脚本和 Slurm。
- 验证：通过新增 checkpoint 模块/脚本的 `python -m py_compile`，通过 `prepare_maxvit_cutlass_checkpoints.sh` 等 Slurm 的 `bash -n`。
## 2026-05-16 - Fix MaxViT small/base dense NVFP4 shape guard
- 开发目的：修复 MaxViT small/base dense CUTLASS NVFP4 精度接近随机的问题。
- 修改内容：MaxViT dense NVFP4 替换新增 `in_features % 64 == 0` 约束，跳过 small/base 第一 stage 的 12 个 `K=96` Linear；checkpoint 元数据选择同步使用该约束，并避免登录节点 shape 检查触发 extension build。
- 影响文件：`fake/models/maxvit_cutlass_nvfp4.py`、`fake/models/maxvit_cutlass_checkpoint.py`。
- 验证：`python3 -m py_compile` 通过；登录节点 shape 检查显示 tiny/large 替换计数不变，small/base 分别新增跳过 12 个 unsupported dense NVFP4 shape。

## 2026-05-16 - Refresh MaxViT small/base result summary
- 开发目的：同步 small/base dense NVFP4 重测后的有效结果，移除 summary 中旧的 abnormal 结论。
- 修改内容：更新 `artifacts/results/summary.md` 中 MaxViT small/base dense NVFP4 的 accuracy、speed、checkpoint size、replaced/skipped count 和结论说明。
- 影响文件：`artifacts/results/summary.md`。
- 后续注意：summary 当前保留历史 CSV 中旧 abnormal 行，但表格和结论均引用最新一行重测结果。

## 2026-05-16 - Plot accuracy/compression/speed summary
- 开发目的：在现有 accuracy summary 基础上增加 compression ratio 和 speedup 的联合展示。
- 修改内容：新增 `scripts/plot_accuracy_compression_speed_summary.py`，生成 `artifacts/results/accuracy_compression_speed_summary.png` 和同源 CSV；DINOv3 speedup 使用 batch size 8，未实现真实 kernel 的 unstructured/2:4 fake 路径 speedup 标为 `NA`。
- 影响文件：绘图脚本、`artifacts/results/accuracy_compression_speed_summary.png`、`artifacts/results/accuracy_compression_speed_summary.csv`。
- 验证：`python3 -m py_compile` 通过；在 `wja-cospaq` 环境成功生成 PNG/CSV。
