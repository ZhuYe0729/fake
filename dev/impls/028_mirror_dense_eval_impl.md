## 2026-05-25 - MIRROR dense 评测入口
- 开发目的：跑通 MIRROR 原始 dense 鉴伪模型在 Chameleon 与 GenImage validation 上的评测流程。
- 修改内容：新增 MIRROR 评测脚本，支持 Chameleon 解压目录与 GenImage validation zip；新增 Slurm 提交脚本；补充命令文档。
- 影响文件：`scripts/eval_mirror_dense_accuracy.py`、`scripts/slurm/eval_mirror_dense_accuracy.sh`、`scripts/slurm/all_model_test_commands.md`。
- 验证：`python -m py_compile scripts/eval_mirror_dense_accuracy.py` 通过；`bash -n scripts/slurm/eval_mirror_dense_accuracy.sh` 通过；`--discover-only` 确认 Chameleon 为 26033 张、GenImage zip 为 8 个子集且 real/fake 数量匹配。
- 后续注意：默认优先读取完整 GenImage zip，避免半解压目录导致漏评；默认 `BATCH_SIZE=16` 且 `USE_AMP=1`，如 RTX 5090 显存不足可继续降低 batch size。

## 2026-05-25 - 解压目录发现检查
- 开发目的：确认 Chameleon 与 GenImage 完整解压后可以直接用于评测。
- 修改内容：`--discover-only` 路径延迟导入 PyTorch/torchvision，避免登录节点默认 Python 环境缺少 torch 动态库时无法做数据发现。
- 影响文件：`scripts/eval_mirror_dense_accuracy.py`、`scripts/slurm/eval_mirror_dense_accuracy.sh`、`scripts/slurm/all_model_test_commands.md`。
- 验证：`--prefer-extracted-genimage --discover-only` 确认 Chameleon 为 26033 张，GenImage 解压目录为 8 个子集且 real/fake 数量匹配；`python -m py_compile` 与 `bash -n` 通过。

## 2026-05-26 - MIRROR smoke test 跑通
- 开发目的：验证安装 `peft` 后 MIRROR 能在 RTX 5090 计算节点加载权重并完成前向。
- 修改内容：修复 MIRROR memory bank 在 AMP/fp16 下 `masked_fill(-1e9)` 溢出，改用当前 dtype 可表示最小值；Slurm 脚本改为 `python -u` 便于实时日志。
- 影响文件：`third_party/MIRROR/models/mirror.py`、`scripts/slurm/eval_mirror_dense_accuracy.sh`。
- 验证：`LIMIT_PER_CLASS=8` smoke job `476603` 完成，Chameleon 与 GenImage 8 个子集均写入 `artifacts/results/mirror_dense/smoke_accuracy.csv`。

## 2026-05-26 - 完整评测启动与坏图容错
- 开发目的：开始 Chameleon 与 GenImage 完整评测，并避免单张坏图中断整个 GenImage 流程。
- 修改内容：DataLoader 增加 invalid image skip collate；指标中的 `num_samples`、`real_samples`、`fake_samples` 改为实际成功读取样本数。
- 影响文件：`scripts/eval_mirror_dense_accuracy.py`。
- 验证：Chameleon 完整作业 `476614` 完成并写入 `artifacts/results/mirror_dense/chameleon_accuracy.csv`，Bal_Acc `0.941241`、AUC `0.983735`；GenImage 完整作业 `476625` 完成并写入 `artifacts/results/mirror_dense/genimage_accuracy.csv`，跳过 1 张坏图，MEAN Bal_Acc `0.995859`、AUC `0.999943`。

## 2026-05-26 - MIRROR 架构文本导出
- 开发目的：为 MIRROR dense detector 补充与现有 MaxViT/DINOv3 一致的模型架构 txt 产物。
- 修改内容：`scripts/temp_dump_models.py` 新增 `--model mirror` 分支，复用本地 MIRROR backbone 与 memory bank 路径导出架构；`scripts/slurm/dump_arch.sh` 增加 mirror dump 命令；生成 `artifacts/model_details/mirror_arch.txt`。
- 影响文件：`scripts/temp_dump_models.py`、`scripts/slurm/dump_arch.sh`、`artifacts/model_details/mirror_arch.txt`。
- 验证：conda 环境 `python -m py_compile scripts/temp_dump_models.py` 通过；`bash -n scripts/slurm/dump_arch.sh` 通过；`python scripts/temp_dump_models.py --model mirror` 成功生成 109 行架构文本。
- 后续注意：生成过程需要读取本地 DINOv3-Huge backbone 与 MIRROR memory bank，登录节点 CPU 加载会有短暂等待。
