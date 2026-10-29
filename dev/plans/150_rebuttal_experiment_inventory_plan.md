# 150 — 人工reviewer补充实验总清单与完整表格

## 目标
- 根据用户及老师建议，将原Llama-2-7B-Chat、Llama-3.1-8B-Instruct缺少的强基线与Qwen补充统一盘点。
- 交付一个Markdown：逐表对应人工reviewer，展开主表和各专项实验的行列、优先级、已有来源、缺项与验收口径。
- 只整理和核查证据，不中断run016，不在本次文档任务中自动启动新的Llama/30B GPU实验。

## 执行与验收
1. 对照reference/wu.pdf、五位人工reviewer、原论文规范表及004–016证据 → 核对五位concern均有对应条目，排除AI Reviewer。
2. 创建artifacts/rebuttal/EXPERIMENT_CHECKLIST.md → 三模型完整主表；已有数值注明协议与来源，缺项留待测，必补与条件扩展分开。
3. 检查Markdown表格列数、本地链接、原主表数值与Qwen CSV一致 → 添加responses入口及impl记录，不修改封存实验。

## 约定
- Qwen解决跨家族验证，不替代原Llama上的HAWQ/SmoothQuant/FP8基线。
- 原Llama旧速度不与新增Graph结果直接混排。新八候选的局部误差、scales、训练与Hessian均按目标模型重新准备，不能复制Qwen系数。
- 同候选消融与完整DASO动作空间分开；所有预算、失败、退化端点与不利结果保留。
