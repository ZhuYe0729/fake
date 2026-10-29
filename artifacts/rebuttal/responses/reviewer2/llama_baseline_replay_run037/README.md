# Llama8bit基线同输入重放（R2/R5）

两模型plain INT8、SQ-INT8、FP8；沿034/035使用相同数值代码和精确export缓存，仅隔离输出目录。NLL重放原64窗口，生成重放每任务最前128条，保持原batch8/4与分组、BOS/PMPD和256token/EOS。这里比较相同128子集，不用子集分数替代完整benchmark。

共2304条生成重放，逐token完全一致2304条；全部NLL落在原cache gate内：True。完全一致率和数值容差只是复用可靠性证据，不是新泛化或新性能结论。

| 模型 | 方法 | NLL均值差 | 最大窗口差 | 原容差通过 |
| --- | --- | ---: | ---: | --- |
| llama31_8b_instruct | uniform_int8 | 0.00000000 | 0.00000000 | True |
| llama31_8b_instruct | uniform_sq_int8 | 0.00000000 | 0.00000000 | True |
| llama31_8b_instruct | uniform_fp8 | 0.00000000 | 0.00000000 | True |
| llama2_7b_chat | uniform_int8 | 0.00000000 | 0.00000000 | True |
| llama2_7b_chat | uniform_sq_int8 | 0.00000000 | 0.00000000 | True |
| llama2_7b_chat | uniform_fp8 | 0.00000000 | 0.00000000 | True |

| 模型 | 方法 | 任务 | token完全一致/128 | 原128子集分数 | 新128子集分数 | 差值 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| llama31_8b_instruct | uniform_int8 | cnn_dm_1000 | 128 | 17.25176 | 17.25176 | 0.00000 |
| llama31_8b_instruct | uniform_int8 | dsum | 128 | 12.24831 | 12.24831 | 0.00000 |
| llama31_8b_instruct | uniform_int8 | IWSLT | 128 | 10.17783 | 10.17783 | 0.00000 |
| llama31_8b_instruct | uniform_sq_int8 | cnn_dm_1000 | 128 | 17.95056 | 17.95056 | 0.00000 |
| llama31_8b_instruct | uniform_sq_int8 | dsum | 128 | 11.04818 | 11.04818 | 0.00000 |
| llama31_8b_instruct | uniform_sq_int8 | IWSLT | 128 | 10.07780 | 10.07780 | 0.00000 |
| llama31_8b_instruct | uniform_fp8 | cnn_dm_1000 | 128 | 16.72023 | 16.72023 | 0.00000 |
| llama31_8b_instruct | uniform_fp8 | dsum | 128 | 12.89730 | 12.89730 | 0.00000 |
| llama31_8b_instruct | uniform_fp8 | IWSLT | 128 | 10.04957 | 10.04957 | 0.00000 |
| llama2_7b_chat | uniform_int8 | cnn_dm_1000 | 128 | 21.72117 | 21.72117 | 0.00000 |
| llama2_7b_chat | uniform_int8 | dsum | 128 | 19.78255 | 19.78255 | 0.00000 |
| llama2_7b_chat | uniform_int8 | IWSLT | 128 | 15.62625 | 15.62625 | 0.00000 |
| llama2_7b_chat | uniform_sq_int8 | cnn_dm_1000 | 128 | 20.86579 | 20.86579 | 0.00000 |
| llama2_7b_chat | uniform_sq_int8 | dsum | 128 | 18.75589 | 18.75589 | 0.00000 |
| llama2_7b_chat | uniform_sq_int8 | IWSLT | 128 | 17.87514 | 17.87514 | 0.00000 |
| llama2_7b_chat | uniform_fp8 | cnn_dm_1000 | 128 | 21.60925 | 21.60925 | 0.00000 |
| llama2_7b_chat | uniform_fp8 | dsum | 128 | 19.52519 | 19.52519 | 0.00000 |
| llama2_7b_chat | uniform_fp8 | IWSLT | 128 | 18.74516 | 18.74516 | 0.00000 |

任何差异均保留，不能据此选择更弱的基线。必要的完整基线复测需在定位输入/权重/运行时差异后独立开展。原017/018结果与各追加主表不改。

[NLL CSV](nll_comparison.csv) · [生成CSV](generation_comparison.csv) · [逐token差异](token_differences.json) · [原始代码/输出](../../../runs/037/README.md)
