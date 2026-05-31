# 029 MIRROR 压缩精度与速度评测计划

## Summary
为 MIRROR 增加全量现有压缩方法评测流程：先准备压缩 checkpoint，再在 Chameleon 与 GenImage 上测精度；只有具备真实端到端 runtime/kernel 的方法额外测速度。压缩范围限定为 MIRROR 的 DINOv3 backbone transformer Linear，不压缩 memory bank、detector head 或数据预处理逻辑。

## Key Changes
- 新增 MIRROR 压缩模型支持：
  - 在压缩模块选择中增加 `mirror`，选择 MIRROR backbone 内 transformer 的 `q/k/v/o` projection 与 MLP Linear。
  - 复用现有 `CompressionConfig` 和方法集合：`nvfp4`、`int4`、`unstructured_sparse`、`semi_structured_sparse`、`nvfp4_*_sparse`、`int4_*_sparse`、`nvfp4_4over6_*_sparse`。
  - 新增 MIRROR loader helper，统一加载 dense checkpoint、memory bank、DINOv3 backbone，并支持加载压缩 checkpoint。
- 新增评测入口：
  - `prepare_mirror_compressed_model.py`：按方法生成 `artifacts/checkpoints/mirror/{method}/model.pt` 与 metadata。
  - `eval_mirror_compressed_accuracy.py`：复用 MIRROR 数据发现、坏图跳过、Chameleon/GenImage 指标输出，结果写入 `artifacts/results/mirror_compressed/accuracy.csv`。
  - `bench_mirror_compressed_speed.py`：只对真实端到端 runtime 路径测速，结果写入 `artifacts/results/mirror_compressed/speed.csv`。
- 速度测试口径：
  - 测端到端速度：`dense`、`nvfp4` via CUTLASS dense NVFP4、`semi_structured_sparse` via CUTLASS sparse BF16、`nvfp4_semi_structured_sparse` via CUTLASS sparse NVFP4。
  - 只测精度：`int4`、`unstructured_sparse`、`nvfp4_unstructured_sparse`、`int4_unstructured_sparse`、`int4_semi_structured_sparse`、`nvfp4_4over6_unstructured_sparse`、`nvfp4_4over6_semi_structured_sparse`。
- 新增 Slurm 批量脚本和命令文档：
  - 准备 checkpoint、跑全部精度、跑可测速方法速度。
  - 默认读取已解压 GenImage，保留坏图跳过逻辑。

## Test Plan
- 静态检查：
  - `python3 -m py_compile` 覆盖新增脚本和改动模块。
  - `bash -n` 覆盖新增 Slurm 脚本。
- smoke tests：
  - `LIMIT_PER_CLASS=8` 跑 dense、一个 fake checkpoint 方法、一个 CUTLASS runtime 方法。
  - 确认 checkpoint metadata、accuracy CSV、speed CSV 字段完整。
- full tests：
  - 全量压缩 checkpoint 准备。
  - 全量方法跑 Chameleon + GenImage 精度。
  - 仅真实 runtime 方法跑端到端速度。
  - 记录失败方法、跳过模块数、坏图数量和实际样本数。

## Assumptions
- 方法范围采用“全量现有方法”。
- 压缩校准默认使用 MIRROR 评测数据发现逻辑抽样，默认 `calib_samples=64`、`calib_batch_size=1`，避免 DINOv3-Huge 在 5090 上内存压力过大。
- 端到端速度指完整 MIRROR forward，包括 preprocessing 后的模型 forward、backbone、memory bank、detector head；不把 fake quant 的普通 PyTorch Linear 速度当作真实压缩 runtime。
- 当前 GenImage 解压目录中已知坏图继续按现有策略跳过并记录实际样本数。
