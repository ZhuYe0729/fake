## 2026-06-13 - Kernel NVFP4 precision proxy scaffold
- 开发目的：将 `dense_nvfp4` 和 `sparse_nvfp4` 纳入 `016` sampled-policy 精度代理建模，并使用 `015` 的 kernel-aware local error。
- 修改内容：新增 kernel-aware sampled loss runner 和 NVFP4 proxy fitter；新增 `046` plan。
- 影响文件：`dev/plans/046_kernel_nvfp4_precision_proxy_plan.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/`。
- 后续注意：完整 NVFP4 loss 评估需要 GPU，建议先 `--max-policies 2` smoke，再跑 120-policy full run。

## 2026-06-13 - Kernel smoke and full run start
- 开发目的：验证真实 kernel path 后启动完整 NVFP4 sampled loss 采样。
- 修改内容：在 `/tmp/kernel_nvfp4_proxy_smoke` 跑通 `dense_nvfp4` 和 `sparse_nvfp4` 各 2 条 policy；随后用 `CUDA_VISIBLE_DEVICES=1,2,3,4` 启动完整后台任务。
- 影响文件：`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/logs/run_kernel_nvfp4_loss_samples.*`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/loss/kernel_workers/`。
- 后续注意：已监督到 4 个 worker 正常处理 `dense_nvfp4`，worker CSV 总计 19 条；后台 PID 在 `logs/run_kernel_nvfp4_loss_samples.pid`。

## 2026-06-13 - Kernel NVFP4 proxy fit complete
- 开发目的：完成 `dense_nvfp4` 和 `sparse_nvfp4` 的 sampled-policy 精度代理拟合。
- 修改内容：确认两个方法各 120 条 loss 样本完整且 `runtime_kernel` 路径正常；运行 `fit_kernel_nvfp4_proxy.py` 生成模型、预测、指标、holdout 图和汇总。
- 影响文件：`loss/loss_samples_dense_nvfp4.csv`，`loss/loss_samples_sparse_nvfp4.csv`，`model/fitted_*_nvfp4_proxy.json`，`model/proxy_metrics_*_nvfp4.csv`，`plots/holdout_*_nvfp4_proxy_vs_loss_delta.png`，`summary/kernel_nvfp4_proxy_summary.md`。
- 后续注意：holdout 指标为 `dense_nvfp4` Pearson=0.9689/Spearman=0.9363，`sparse_nvfp4` Pearson=0.9801/Spearman=0.9778；无残留 run/fit 进程。

## 2026-06-13 - Dense NVFP4 nonlinear calibration
- 开发目的：修正 `dense_nvfp4` 一阶乘性代理对大压缩数量的系统性高估。
- 修改内容：仅对 `dense_nvfp4` 增加校准层 `c0 + c1*base_pred + c2*base_pred^2 + c3*log1p(selected_modules)`，保留原始乘性代理作为 `base_pred_loss_delta`。
- 影响文件：`scripts/fit_kernel_nvfp4_proxy.py`，`model/fitted_dense_nvfp4_proxy.json`，`model/predictions_dense_nvfp4.csv`，`model/proxy_metrics_dense_nvfp4.csv`，`plots/holdout_dense_nvfp4_proxy_vs_loss_delta.png`，`summary/dense_nvfp4_README.md`，`summary/kernel_nvfp4_proxy_summary.md`。
- 后续注意：dense holdout MAE 从 0.029674 降到 0.004917，RMSE 从 0.034916 降到 0.007741；Pearson/Spearman 基本不变，说明校准主要修正绝对尺度和饱和曲线。
