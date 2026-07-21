# 103 Llama2 Dense Prefill 基础质量评测计划

## Summary

- 在 `artifacts/debug/040_llama2_7b_dense_prefill_eval/` 中完成原始未压缩 Llama-2-7B-Chat 的基础质量评测。
- 使用 `cospaq` 环境和 lm-eval `HFLM` / Transformers 后端，对齐最近 `037_llama2_prefill_only_pareto` 的 ARC-Challenge 精度口径。
- 最多使用四张 GPU 并发；每卡同一时刻只运行一个任务。

## Fixed protocol

- 模型：`/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf`，BF16，`local_files_only=True`。
- PPL：lm-eval `wikitext` 与 `c4`，报告 `word_perplexity`、`byte_perplexity`、`bits_per_byte`。
- 精度：lm-eval `winogrande`、`arc_easy`、`mmlu`，均为 0-shot，记录 `acc` 和可用的 `acc_norm`。
- 不重复运行 ARC-Challenge；不评测任何压缩模型。

## Implementation

- 新增单任务 HFLM runner，保存原始 lm-eval 输出、模型/数据环境信息和包含模型加载与数据读取的总墙钟时间。
- 新增四卡队列 launcher：首批启动 WikiText、C4、Winogrande、ARC-Easy；任一卡完成后在该卡启动 MMLU。任务以 `CUDA_VISIBLE_DEVICES` 隔离。
- 新增汇总脚本，输出每项指标、样本量、单项耗时、并行总 wall-clock、日志和原始结果路径。

## Verification

- 静态编译所有 Python 脚本并检查 launcher 帮助文本。
- 以 `--limit` 对全部五项做四卡 smoke，确认 GPU 绑定、结果写入和补位调度。
- 再运行无 `--limit` 的完整五项评测；汇总必须能够追溯到每项原始 JSON。

## TODO

- 在相同任务和时间口径下补充 uniform/heterogeneous 压缩模型。
