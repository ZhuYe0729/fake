## R1 / AK6a：预算满足率与双表示开销

> We added a validation-and-confirmation procedure that freezes the selected policy before opening new, disjoint confirmation windows. Across three models, four budgets (ε=0.01/0.02/0.04/0.07), and two validation sizes (32/64 windows), all 24 budget labels have measured confirmation means within their budgets. These labels reuse 14 distinct model/policy selections and are not independent trials. This result supports the validation-enhanced procedure empirically; it is not a guarantee from the original additive surrogate alone.
>
> With 64 validation windows at ε=0.04, the selected Qwen3, Llama-3.1, and Llama-2 policies have new confirmation ΔNLL of −0.01548, 0.01866, and 0.02353, respectively. Their corresponding, previously measured policies achieve 2.061×, 1.891×, and 1.770× BF16 E2E speed. We explicitly distinguish inherited speed measurements from the new NLL confirmation.
>
> We also quantify representation storage. In the dual-resident mixed-runtime experiment, a representative Qwen policy has 17.104 GiB peak Torch-allocated GPU memory versus 21.384 GiB for BF16, together with an additional 8.236 GiB host adapter-weight cache. Thus, we report both compressed GPU residency and the host-side cost instead of treating phase switching as free.

对应[质量预算](REBUTTAL_SELECTED_EVIDENCE.md#quality-budget)与[运行开销](REBUTTAL_SELECTED_EVIDENCE.md#runtime-cost)。完整24标签来自[044](responses/reviewer1/fresh_quality_run044/README.md)、[045](responses/reviewer1/fresh_quality_run045/README.md)、[046](responses/reviewer1/fresh_quality_run046/README.md)；[速度关联与SHA](response174/budget_speed.csv)。

措辞要点：0/24指实测**均值**超限，不能改写为所有保守保护值均通过或0/24独立失败概率；这些是公开WikiText语料中的新窗口，不是新的外部语料。ε=.04的Qwen map是`w1_e0.04`，不是R2的两个生成质量代表点。

