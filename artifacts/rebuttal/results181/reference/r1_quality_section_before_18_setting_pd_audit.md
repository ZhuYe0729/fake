<a id="quality-budget"></a>
## 2. P+D质量预算核验（R1）

**整理状态：当前三列表仍是早期版本的内部诊断，暂不选为最终回复主表。** 旧P+D测试均值满足预算，四点绝对误差MAE为0.00764、最大误差0.01997，并未超出早期Qwen整体留出误差的量级；较宽预算有保守预测，小预算仍需结合不确定性单独讨论。已确认沿用002/003冻结质量模型，005全部2016个动作成本与该模型重算完全一致；最终回复需明确版本与实验范围。

本轮先用[文字回复草稿](responses/reviewer1/quality_budget_text_reply/README.md)说明P+D原DP的0/4实测均值超限及其有限范围；旧三列表与validation增强版分别保留在下方折叠区。

<details>
<summary>历史P+D直接求解审计：预算、预测与实测（原数据完整保留）</summary>

Qwen3-8B，B=8、输入2048、输出64。每个配置在同一组64个独立于拟合的测试文本窗口上评测，每窗64个teacher-forced目标token；表中均为相对BF16的平均ΔNLL，单位nats/token。

以下是Run005当时冻结的七候选、原DP直接输出，未经额外validation筛选；与最新主表代表点分开，不作Prefill或其他模型的满足率声明。

| 预算 ε | 预测平均 ΔNLL | 实测平均 ΔNLL |
| ---: | ---: | ---: |
| 0.005 | -0.00298 | -0.00762 |
| 0.010 | -0.00041 | -0.00593 |
| 0.020 | 0.00470 | 0.00426 |
| 0.040 | 0.02176 | 0.00180 |

四个预测值均不超过预算，四个测试均值也均未超限，观察到0/4平均超限。负值表示这批窗口上的NLL略低于BF16。较宽预算下预测偏保守，不能概括为预测值与实测值始终接近。

最紧预算0.005的实测均值未超限，但其描述性95%配对bootstrap区间[-0.02225, 0.00744]跨过预算，因此不作严格质量保证。仅审计本表P+D设置，不代表所有场景。

[完整协议与精确求解版附表](responses/reviewer1/pd_quality_budget/README.md) · [逐点CSV/SE/区间](responses/reviewer1/pd_quality_budget/points.csv) · [512对窗口重算与来源SHA](responses/reviewer1/pd_quality_budget/audit.json)

</details>

<details>
<summary>补充：加入validation后的三模型确认结果（与上方直接求解审计分开）</summary>


**结论：** 三模型、四个预算和两种验证样本数，共24个预算设定，在确认样本上实测的**平均NLL损失**均未超限。它们复用了14个模型/配置选择，不是24次独立试验。

这里的损失用ΔNLL表示，越小越好；小于允许上限即满足本表的平均损失要求。以下展示64个验证样本时的代表结果。**这些是按NLL预算单独选择的配置，不是上方Balanced/Max-speed的精度声明。**

| 模型 | 允许损失上限 | 实测平均损失 | 平均损失满足？ |
| --- | ---: | ---: | --- |
| Qwen3-8B | 0.02 | -0.0264 | 是 |
| Qwen3-8B | 0.04 | -0.0155 | 是 |
| Llama-3.1-8B | 0.02 | 0.0086 | 是 |
| Llama-3.1-8B | 0.04 | 0.0187 | 是 |
| Llama-2-7B | 0.02 | 0.0043 | 是 |
| Llama-2-7B | 0.04 | 0.0235 | 是 |

<details>
<summary>测量说明与来源</summary>

先用一组样本验证并冻结选择，再用另一组不重叠的64个样本检查；检查结果不参与重新选点。使用公开WikiText中的新窗口，排除旧校准/验证/测试分词流中共享的连续16-token片段，不是外部新语料。

选择规则是验证均值＋2SE＋0.005；上表六个确认均值及均值＋2SE均低于预算。完整24设定中，平均损失全部满足；另加余量的确认保护值并非全部满足（21/24），所以不称为硬质量保证。

这些预算选点可能与模型主表的Balanced/Max-speed不同。NLL是新的确认结果，速度继承同一配置已有正式测量，不是同步新测；NLL预算也不保证每项下游生成分数。

[六个预算与同配置速度](response174/budget_speed.csv) · [完整计数](response174/budget_counts.csv) · [Qwen来源](responses/reviewer1/fresh_quality_run044/README.md) · [Llama-3.1来源](responses/reviewer1/fresh_quality_run045/README.md) · [Llama-2来源](responses/reviewer1/fresh_quality_run046/README.md)。

</details>

</details>

