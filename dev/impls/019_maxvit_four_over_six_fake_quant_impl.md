## 2026-05-18 - MaxViT Four Over Six Entrypoints
- 开发目的：让 MaxViT tiny/small/base/large 可以测试 NVFP4 4/6 unstructured 与 semi-structured sparse fake quant。
- 修改内容：放开 `prepare_compressed_model.py` 中 DINO-only 的 4/6 限制；新增 MaxViT 4/6 prepare、accuracy、speed Slurm 脚本。
- 影响文件：`scripts/prepare_compressed_model.py`、`scripts/slurm/prepare_maxvit_four_over_six_checkpoints.sh`、`scripts/slurm/eval_maxvit_four_over_six_accuracy.sh`、`scripts/slurm/bench_maxvit_four_over_six_speed.sh`。
- 验证：`python3 -m py_compile` 通过；新增 Slurm 脚本 `bash -n` 通过。
- 后续注意：当前 MaxViT 4/6 只覆盖权重 fake quant，不包含 activation fake quant。

## 2026-05-18 - Add MaxViT Activation Fake Quant Switch
- 开发目的：补齐 MaxViT 4/6 的 activation fake quant 开关，支持和 DINOv3 一样默认开启、用环境变量关闭。
- 修改内容：扩展 activation fake quant wrapper，支持 MaxViT 的 Linear 与 1x1 Conv2d；MaxViT accuracy/speed 脚本新增 `--activation-quant` 相关参数；MaxViT 4/6 Slurm 脚本默认传入 activation fake quant，可用 `NO_ACTIVATION_QUANT=1` 关闭。
- 影响文件：`fake/compression/activation.py`、`scripts/eval_maxvit_dense_accuracy.py`、`scripts/bench_maxvit_dense_speed.py`、`scripts/slurm/eval_maxvit_four_over_six_accuracy.sh`、`scripts/slurm/bench_maxvit_four_over_six_speed.sh`。
- 后续注意：activation fake quant 仍是 PyTorch fake path，速度只适合观察 fake-quant 开销，不代表真实 kernel 性能。
