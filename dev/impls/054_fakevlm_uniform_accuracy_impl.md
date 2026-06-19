## 2026-06-18 - FakeVLM uniform accuracy workflow
- 开发目的：为 FakeVLM 的六种 uniform 压缩/推理方法建立 debug-only 精度评测流程。
- 修改内容：新增 FakeVLM/llava language linear selector；新增 020 debug 目录中的评测脚本、并行 launcher、smoke launcher、汇总脚本和说明；稀疏方法使用校准 Hessian 先剪枝再安装真实 runtime；NVFP4 W4A4 路径记录在线 activation global scale 口径。
- 影响文件：`fake/compression/modules.py`、`artifacts/debug/020_fakevlm_uniform_accuracy/*`、`dev/plans/054_fakevlm_uniform_accuracy_plan.md`。
- 后续注意：速度测试仍是 TODO；正式精度需要在能访问 GPU 2-7 的非沙箱环境中运行。

## 2026-06-18 - Full FakeVLM accuracy run
- 开发目的：按计划完成 FakeVLM 六种 uniform 方法的 5000 样本精度测试。
- 修改内容：使用 `BATCH_SIZE=8 WORKERS=2 OVERWRITE=1` 并行运行 6 个独立进程，物理 GPU 分配为 `dense_bf16:7`、`sparse_bf16:6`、`dense_nvfp4:5`、`sparse_nvfp4:4`、`marlin_weight_only:3`、`dense_nvfp4_prefill_marlin_decode:2`；未使用 GPU 0/1。
- 影响文件：`artifacts/debug/020_fakevlm_uniform_accuracy/outputs/*`、`artifacts/debug/020_fakevlm_uniform_accuracy/status/*`、`artifacts/debug/020_fakevlm_uniform_accuracy/summary/accuracy_summary.csv`、`artifacts/debug/020_fakevlm_uniform_accuracy/README.md`。
- 结果摘要：dense BF16 0.9864，sparse BF16 0.9852，dense NVFP4 0.9870，sparse NVFP4 0.7686，Marlin weight-only 0.9876，dense NVFP4 prefill + Marlin decode 0.9868。
- 后续注意：`sparse_nvfp4` 精度明显低于其他方法，需要后续单独分析剪枝/量化口径或 kernel 数值误差；速度仍未测试。
