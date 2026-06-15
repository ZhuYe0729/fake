## 2026-06-13 - Sparse BF16 precision proxy experiment scaffold
- 开发目的：为 Llama2 sparse BF16 建立新的 sampled-policy loss 建模实验目录。
- 修改内容：新增 `016_llama2_sparse_bf16_precision_proxy`，包含 policy 采样、多 GPU loss 评估、乘性系数拟合和 holdout 趋势图脚本。
- 影响文件：`dev/plans/045_sparse_bf16_precision_proxy_plan.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/`。
- 后续注意：完整 loss 评估需要本机 CUDA/GPU 和可用的 Llama2 权重；先用 `--max-policies` 做 smoke，再跑完整 120 policy。

## 2026-06-13 - GPU smoke validation
- 开发目的：确认沙箱外 GPU 环境可以运行 sparse BF16 sampled loss 评估。
- 修改内容：用 `/tmp/sparse_bf16_gpu_smoke` 跑通 1-policy GPU smoke，验证模型加载、4 个 linear 替换、loss 计算和 worker 输出合并。
- 影响文件：仅追加本实现记录；正式 `016` 结果目录未写入 smoke loss。
- 后续注意：完整实验可直接使用 `CUDA_VISIBLE_DEVICES=1,2,3,4` 启动多 worker。

## 2026-06-13 - Full sampled loss run started
- 开发目的：启动完整 120-policy sparse BF16 loss 采样。
- 修改内容：用 `setsid` 后台启动 `run_sparse_bf16_loss_samples.py --skip-existing`，使用 `CUDA_VISIBLE_DEVICES=1,2,3,4` 和 4 个 worker。
- 影响文件：`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/loss/`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/logs/`。
- 后续注意：监督到 worker 总结果数达到 26/120，主进程 PID 记录在 `logs/run_sparse_bf16_loss_samples.pid`；完成后需运行拟合脚本生成 holdout 图。

## 2026-06-13 - Fit proxy and generate holdout plot
- 开发目的：用完整 120 条 sampled sparse BF16 loss 数据拟合乘性精度代理。
- 修改内容：将拟合脚本改为向量化设计矩阵实现，完成模型拟合、prediction CSV、metrics CSV、holdout 趋势图和 summary 生成。
- 影响文件：`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/model/`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/plots/`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/summary/README.md`，`scripts/fit_sparse_bf16_proxy.py`。
- 后续注意：holdout 36 条，Pearson=0.9870，Spearman=0.9822；当前代理对 sampled 配置 loss 趋势对应很好。
