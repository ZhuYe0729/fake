## 2026-10-26 - 三模型与五位人工reviewer实验总清单
- 根据用户和老师建议创建总览Markdown，包含18张表：两Llama各17行完整主表、Qwen19行主表（17已有+2运行中）、同候选/阶段、质量预算、反向约束、负载、mixed调度、延迟模型、准备/内存/接口成本、剪枝、30B、跨GPU、搜索消融及文档复核。
- 原Llama 14行154个数字来自规范表，全部明确旧协议参考、速度需统一重测；Qwen17行187个数字直接从run015主表CSV复制。P0/P1/P2、待测/待汇总/条件项区分，不要求两Llama复制全部历史探索，也不把Qwen替代Llama强基线。
- 核查reference/wu.pdf主表确为Table2，原规范目录README的表号不能替代提交PDF。核对dsum loader与1500条冻结reference全部对应实际DialogSum，明确旧run010文字XSum误标，不改封存报告。
- 影响文件：EXPERIMENT_CHECKLIST.md、experiment_checklist_audit.json、responses/README.md及plan150。校验全部Markdown表格列数、本地链接和源文件SHA，未修改实验配置或抢占GPU；run016继续进入confirmation。
