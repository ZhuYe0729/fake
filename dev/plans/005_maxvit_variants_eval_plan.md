# MaxViT 多尺寸评测扩展计划

## Summary
将现有只面向 `timm/maxvit_tiny_tf_224.in1k` 的 MaxViT dense/压缩评测链路参数化，补充 `small`、`base`、`large` 三个模型，并保留 `tiny` 默认行为。

## Key Changes
- 在 `fake/models/maxvit.py` 增加 `tiny`、`small`、`base`、`large` variant registry，记录真实 timm model id、本地路径和结果 key。
- MaxViT accuracy/speed CLI 增加 `--variant`，CSV 写入真实 `model` 和 `model_variant`；speed 未指定 `--input-size` 时从模型 config 推断，large 使用 `3x512x512`。
- 压缩 checkpoint 生成支持 `--maxvit-variant`，MaxViT 默认输出到 `artifacts/checkpoints/maxvit_<variant>/<method>/`，metadata 写入 `model_id` 和 `model_variant`。
- Slurm 脚本通过 `MAXVIT_VARIANT` 单变体提交，MaxViT 新结果写入 `artifacts/results/maxvit_<variant>_dense/` 和 `artifacts/results/maxvit_<variant>_compressed/`。
- `maxvit_dense/` 和 `maxvit_compressed/` 作为历史目录保留，不再作为 MaxViT 新实验默认输出目标。

## Test Plan
- 登录节点检查四个本地 config 可读，并验证 registry 能返回正确 model id、路径和输入尺寸。
- 运行 MaxViT accuracy/speed/compression CLI 的 `--help`，确认新参数可用。
- 计算节点正式验证 small/base/large 的 dense accuracy/speed，以及五种压缩方法的 checkpoint、accuracy、speed。

## Assumptions
- 四个 MaxViT 权重目录均已在 `/data/home/scxj523/run/wja/data/models/timm/` 下准备好。
- tiny/small/base 默认 batch size 为 `128`，large 默认 batch size 为 `16`；large 压缩 calib batch 默认 `4`。
- 若 large 显存不足，通过 `BATCH_SIZE` 或 `CALIB_BATCH_SIZE` 环境变量下调。
