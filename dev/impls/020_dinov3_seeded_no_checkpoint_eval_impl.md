## 2026-05-18 - DINOv3 Seeded No-Checkpoint Eval
- 开发目的：支持 DINOv3 多 seed 精度筛选，同时避免每个 seed 保存约 26GB fake checkpoint。
- 修改内容：新增 one-shot accuracy 脚本，在内存中完成 seeded calibration compression 和 ImageNet eval；4/6 方法支持 activation off/on；新增 Slurm sweep 入口。
- 影响文件：`scripts/eval_dinov3_vit7b16_seeded_compression_accuracy.py`、`scripts/slurm/eval_dinov3_vit7b16_seeded_compression_accuracy.sh`。
- 验证：`python3 -m py_compile scripts/eval_dinov3_vit7b16_seeded_compression_accuracy.py` 通过；Slurm 脚本 `bash -n` 通过。
- 后续注意：该路径不保存 `model.pt`；选出 best seed 后，如需持久 checkpoint，再用已有 prepare 脚本单独保存最终候选。
