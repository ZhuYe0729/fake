# Prefill-decode intermediate speed-point plan

目标：为 Llama2-7B-chat 的 prefill-decode 实测帕累托补充约 1.25×、1.45×、1.55×、1.65× 的策略，减少 point 8（1.179×）到 point 11（1.714×）之间的空档。

1. 复用 034/035 的速度预测、WikiText NLL 模型和 phase-hetero 策略物化流程，针对四个速度目标生成候选；验证：每个目标都有可加载的 policy/checkpoint 或明确无可行解原因。
2. 在单独 GPU 上按 `.85` formal prefill-decode 协议测量候选 E2E 速度，并进行 WikiText NLL 实测；验证：候选确实落在缺口内且速度稳定。
3. 仅保留非支配且填补缺口的点，合并到现有 real-task Pareto 汇总；生成任务全量测试和最终重绘列为后续步骤，避免先污染主结果。
