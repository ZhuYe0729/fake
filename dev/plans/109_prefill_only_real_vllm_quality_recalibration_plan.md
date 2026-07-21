# Prefill-only real-vLLM quality recalibration plan

在新的独立 debug 目录中，对 Llama2-7B-Chat 和 Llama3.1-8B-Instruct
以真实 vLLM prompt-logprob NLL 重新校准现有精度模型。保留原来的
local-error 特征、positive local+global/method/bucket/type 拟合结构和
速度模型；不改写历史实验或 exports。

1. 用统一的 72 个 prefill-only 策略（54 train / 18 holdout）和固定
   100 个 WikiText 2048-token block 产生校准数据。
2. uniform 端点走其独立 vLLM checkpoint；混合策略临时导出为
   phase-heterogeneous checkpoint，核验 policy 后以真实 vLLM 评分并删除。
3. 从既有 Llama2 / Llama3 local-error 表提取同一特征，拟合新 NLL 标签，
   输出 holdout 指标、分组残差和图表。
4. 仅在新质量模型验证后再讨论是否将其接入既有 prefill-only solver；
   本计划不重测速度、不加入手工 sparse 修正或 uniform 锚点。
