## 2026-07-15 - 基础评测脚手架
- 开发目的：为原始 Llama-2-7B-Chat 建立与 037 对齐的 PPL 与下游精度评测流程，并记录每项总耗时。
- 修改内容：新建 040 独立实验目录、HFLM 单任务 runner、四卡队列 launcher 和结果汇总脚本；固定 WikiText/C4 PPL 与 Winogrande/ARC-Easy/MMLU 0-shot 评测。
- 影响文件：`artifacts/debug/040_llama2_7b_dense_prefill_eval/`、`dev/plans/103_llama2_dense_prefill_eval_plan.md`。
- 后续注意：需在具备 GPU 驱动且可下载缺失数据集的机器上运行 smoke 后再启动 full profile。

## 2026-07-15 - 数据下载代理开关
- 开发目的：支持本地数据集缓存缺失时的网络恢复。
- 修改内容：四卡 launcher 增加 `--proxy`；启用后向每个 worker 注入用户指定的 `http_proxy`、`https_proxy`、`all_proxy`，并在命令记录中标明代理状态。
- 影响文件：`artifacts/debug/040_llama2_7b_dense_prefill_eval/{README.md,scripts/run_all.py}`。

## 2026-07-15 - 四卡 smoke 验证
- 开发目的：验证 037 对齐的 HFLM 路径、四卡隔离调度、任务耗时写入和首次数据下载恢复。
- 修改内容：在 GPU 0--3 执行 `--limit 1` smoke；ARC-Easy 完成后 GPU 3 自动补位为 MMLU。直连缺失数据集无进展后，使用 `--proxy` 重试；WikiText、Winogrande、ARC-Easy 均写出结果，WikiText PPL 指标为 `7.6505`，证明 HFLM PPL 与准确率路径可用。
- 验证：所有脚本通过 `py_compile`、`--help` 和 `git diff --check`。C4 与 MMLU 的首次全量数据准备明显较长，已停止未完成 smoke 以释放 GPU；正式运行使用 `--proxy`。
- 影响文件：`artifacts/debug/040_llama2_7b_dense_prefill_eval/tasks/*/limit_1/`。

## 2026-07-15 - Full PPL batch-size correction
- 开发目的：解决正式 WikiText rolling-PPL 在 `batch_size=4` 下因长文本 attention score 申请 8 GiB 导致的 OOM。
- 修改内容：launcher 对 WikiText/C4 固定默认 `--ppl-batch-size 1`，Winogrande/ARC-Easy/MMLU 仍使用 `--batch-size 4`；停止未完成的 C4 worker，待仅重启无完整结果任务。
- 影响文件：`artifacts/debug/040_llama2_7b_dense_prefill_eval/{README.md,scripts/run_all.py}`。

## 2026-07-15 - C4 lm-eval download-path correction
- 开发目的：保持 lm-eval 标准 `c4` rolling-PPL 任务不变，修复 C4 validation shard 的下载停滞。
- 修改内容：诊断确认停在 Hugging Face Xet 的 `transfer.xethub.hf.co` TLS handshake EOF，且 `.incomplete` 文件保持 0 B；`--proxy` 同时设置 `HF_HUB_DISABLE_XET=1`，使 lm-eval/datasets 改走普通 Hugging Face HTTP 下载。
- 影响文件：`artifacts/debug/040_llama2_7b_dense_prefill_eval/{README.md,scripts/run_all.py}`。

## 2026-07-15 - C4 validation-only lm-eval task
- 开发目的：避免 lm-eval 原始 C4 配置为未使用的训练分片建立 Arrow 缓存，同时保留 rolling-PPL 的文本处理与聚合指标。
- 修改内容：添加仅声明 validation 分片的本地 `c4_validation_only` 任务；runner 将逻辑任务 `c4` 映射到该任务，并在结果中记录实际 lm-eval task 名称。
- 影响文件：`artifacts/debug/040_llama2_7b_dense_prefill_eval/{lm_eval_tasks/c4_validation_only/,scripts/evaluate_task.py}`。

## 2026-07-15 - C4 persistent worker launch
- 开发目的：避免长时 C4 worker 依附交互执行会话而被回收，保留可审计的独立运行记录。
- 修改内容：新增独立 session 的后台 worker launcher；记录 PID、GPU、启动时间、命令和专属日志，结果仍写入既有 `tasks/c4/full/result.json`。
- 影响文件：`artifacts/debug/040_llama2_7b_dense_prefill_eval/scripts/start_background_task.py`。

## 2026-07-15 - Dense full evaluation completed
- 开发目的：完成原始 dense Llama-2-7B-Chat 的五项质量评测并交付每项总耗时。
- 修改内容：C4 validation-only rolling-PPL 已完成（61.44 分钟）；更新汇总脚本，在任务分次启动时明确 wall-clock 未统一追踪，避免显示为缺失错误。
- 验证：`summary/results_full.{md,csv}` 含五项 `ok` 结果；C4 结果文件记录 HFLM/lm-eval 指标与起止时间。
- 影响文件：`artifacts/debug/040_llama2_7b_dense_prefill_eval/{tasks/*/full/result.json,summary/results_full.*,scripts/summarize.py}`。
