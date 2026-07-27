# Llama2-7B-Chat RTX Pro 6000 prefill-only 实现记录

## 2026-07-21 - 建立独立实验 bundle 并完成主闭环

- 开发目的：在双 RTX Pro 6000 上复现 Llama2-7B-Chat、B=8/S=2048/O=1 的 prefill-only 论文实验，且不修改历史 debug 目录。
- 修改内容：
  - 新建 `artifacts/debug/064_llama2_pro6000_prefill_only/`，复制必要输入后将 exporter、校准、拟合、profile、solver、closure、任务和验证脚本全部隔离到该目录。
  - 建立 sparse BF16 与 sparse NVFP4 canonical state；后者固定为 prequant-only，checkpoint exporter 禁止 direct prune，并记录/校验 canonical provenance。
  - 增加 vLLM 0.11 V1 的局部 chunked-prefill 兼容层及运行时断言，不修改共享 `vllm-cospaq` checkout。
  - 修复多独立 block 的 phase 状态：每次 generate 前重置到 prefill；NLL 结果强制审计 `apply_prefill=blocks*128`、`apply_decode=0`。
  - 修复 mixed-policy quality feature 的逐层 method/bucket/type 归因；solver 排除模型加载前的一次性 CUTLASS/Marlin layout 转换。
  - 增加 checkpoint、GPU UUID、协议、phase trace、任务调用和最终结果的机器可读验证。
- 已完成结果：
  - 72/72 policy 的 100-block NLL，全部 phase 审计通过；质量模型 holdout Spearman 0.9381、MAE 0.0830。
  - Pro 6000 精确 kernel profile，覆盖 4 个 Llama2 shape 与 5 种 kernel；predictor provenance 指向 064。
  - 求解 25 个 Pareto point；完成 5 个 uniform + 25 个 point 的闭环，共 30 份 100-block NLL、30 次 warmup、150 次 measured，同一 GPU UUID，closure validator 通过。
  - 下游任务选出 9 个点：5 个 uniform，以及 `point_010`、`point_014`、`point_019`、`point_024`。
- 影响文件：`artifacts/debug/064_llama2_pro6000_prefill_only/**`、`dev/plans/126_llama2_pro6000_prefill_only_plan.md`、本文件。
- 后续注意：WinoGrande、ARC-Easy/Challenge、MMLU 尚未写入 064 cache；联网授权通道中断后未继续重试。数据下载、9 点完整任务、最终 consolidation/六张图和 `validate all` 待联网权限恢复后完成。

## 2026-07-21 - 完成共享任务缓存、完整任务评测与最终汇总

- 开发目的：解除下游数据阻塞，完成计划剩余的完整 lm-eval、结果汇总和全量验证。
- 修改内容：
  - 将任务数据 payload 从 064 迁移至 `/root/.cache/huggingface` 共享缓存；064 仅保留数据 manifest，并在当前机器配置中显式引用共享缓存。
  - 修正全局 `HF_ENDPOINT=hf-mirror.com` 与新版客户端的元数据不兼容问题；代理下载固定官方 endpoint、HTTP/HTTPS 代理变量并禁用会停滞的 Xet 通道。
  - 按当前 lm-eval YAML 将 WikiText 数据源修正为 `EleutherAI/wikitext_document_level`，并为 MMLU 构建 57 个学科 config，而非仅构建不能满足 group task 的 `all` config。
  - 将 Matplotlib 配置缓存固定到 064 的可写 cache，避免只读 `/root/.config` 警告。
- 验证结果：
  - 五任务 `limit_2` smoke 通过；累计 phase guard 满足每次模型调用之间均重置到 prefill。
  - 9/9 选定策略完成 WikiText、WinoGrande、ARC-Easy、ARC-Challenge 和 MMLU，共 45/45 个 full result；无失败、无缺失 checkpoint audit，全部 `chunked_prefill=false` 且 `prefill_resets=model_generate_calls-1`。
  - 生成 `results/complete_results.csv` 与六张 measured Pareto 图；`validate all` 通过。
- 影响文件：`artifacts/debug/064_llama2_pro6000_prefill_only/{config.current.env,README.md,scripts,cache,runs,results,validation}`、本文件。
- 后续注意：共享 Hugging Face cache 是本机外部依赖；实验冻结 manifest、策略、测量、图表和验证报告仍全部保留在 064。
