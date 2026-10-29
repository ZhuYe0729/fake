<a id="quality-budget"></a>
## 2. 质量预算：要求的损失是否真的满足？（R1）

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

