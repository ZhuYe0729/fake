# `main_table.tex` 结果来源与配置一致性审计

审计日期：2026-07-28  
审计对象：`artifacts/debug/071_my_emergency/main_table.tex`

## 1. 结论

### 1.1 Llama-3.1 8B-Instruct 的 Ours (Balanced)

这组更新后的结果是正确的，且不是旧的异常结果：

- Prefill-only 使用 `point_007`：
  - ARC-C 51.4505
  - ARC-E 79.0825
  - WinoGrande 73.7964
  - MMLU 66.5005
  - TTFT speedup 1.5857，表中四舍五入为 1.59
- Prefill--Decode 使用 `point_005`：
  - CNN/DM 18.8374
  - DialogSum 11.4805
  - IWSLT'17 10.5502
  - TTFT / TPOT / E2E speedup 分别为 1.40 / 1.13 / 1.26

Prefill-only 的 `point_007` 在质量和速度两边使用同一个 policy 文件：

`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/pareto/policies/point_007.json`

其 policy SHA-256 为：

`e422940febbd6f19f8c74d594c209c350d660b5b945ae41138a3aeec2b1ea274`

原始 MMLU 文件内记录了同一个 policy 路径和 SHA-256，因此不是把别的配置的质量结果贴到了这条速度曲线上。该 policy 的 128 个线性模块由 68 个 Dense BF16 和 60 个 Dense NVFP4 组成，不含 sparse 方法。

旧的异常候选是 `point_009`，其 MMLU 为 55.9322，而不是 66.5005。它的配置包含 31 个 Dense BF16、43 个 Dense NVFP4 和 54 个 Sparse BF16 模块。当前表格没有继续使用该结果。

后续的 `068_llama31_prefill_uniform_coverage_refinement` 仍把该 `point_007` 以相同的 685.8683 ms、MMLU 0.665005 纳入其来源表，并把新增实验用于补充其他速度区间，而没有生成一个同 policy、同完整下游评测协议的新结果来替代它。因此就当前仓库而言，`061/point_007` 仍是这个 exact policy 的最新完整四任务质量证据，并非已经被后续结果作废的旧值。这里的主要证据仍是 `061` 的原始 JSON；`068/.../report/all_results.csv` 只用于检查后续实验是否将其标记为被替代。

### 1.2 整张表

整张表的数值计算和两位小数舍入均可复现，但不能把所有单元格都描述成“原始证据完整且配置标签、评测协议完全准确”。有三个需要明确说明或补实验的问题：

1. **Llama-2 Prefill--Decode 的 BF16、Dense NVFP4、Sparse BF16 三组生成质量缺少原始逐样本结果。**
   表中的九个质量值只在聚合 CSV 中出现，未找到对应的 `metrics.json` 和逐样本 `jsonl`。因此这些值很可能是上一轮实验继承来的实测结果，但目前无法从仓库保留的原始输出独立复算。
2. **Llama-2 Prefill--Decode 的 “Sparse NVFP4” 不是两个阶段都使用 Sparse NVFP4。**
   `p03.json` 明确记录的是 prefill 使用 Sparse NVFP4、decode 使用 Dense NVFP4。原因是该轮实验在 `M=8` 的 decode 阶段不支持 Sparse NVFP4。表格应改名或加脚注说明这是 legal projection。
3. **Llama-3.1 Prefill--Decode 的生成质量使用 legacy/common PMPD prompt，不是模型原生 chat template。**
   所有 Llama-3.1 方法使用同一个 legacy prompt，因此模型内部的 BF16、uniform、Ours 相对比较仍然公平；但后续配对诊断表明 native template 会显著改变绝对生成分数，所以这些绝对值不能与采用原生模板的实验直接横向比较。

除上述三项外，其余表格结果都有 policy、原始质量输出和逐次速度测量可以追溯。

## 2. 表格由哪几轮实验组成

该表不是单个实验目录直接导出的，而是四条 canonical 轨道拼接而成：

| 模型与场景 | 实验根目录 | 主要状态 |
|---|---|---|
| Llama-2 / Prefill-only | `artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat` | 原始质量和速度文件完整 |
| Llama-2 / Prefill--Decode | `artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat` | 速度完整；三组 uniform 生成质量缺原始文件 |
| Llama-3.1 / Prefill-only uniform quality | `artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/llama31_8b_instruct` | 原始质量文件完整 |
| Llama-3.1 / Prefill-only warmed speed 与 Ours quality | `artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct` | warmed speed 和原始质量文件完整 |
| Llama-3.1 / Prefill--Decode | `artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct` | 原始质量和速度文件完整 |

Llama-3.1 Prefill-only 的 uniform 质量来自 `058`，速度来自后续 warmed-speed 复测 `061`。这属于不同运行批次，但不是不同 policy：`061/.../speed/calibration/design.csv` 直接指向 `058/.../policies/prefill_only/p00.json` 至 `p04.json`。因此这里是同一配置的分批质量/速度测量，不是配置混用。

## 3. 评测口径与复算方法

### 3.1 Prefill-only

- 速度 workload：batch 8、input sequence 2048、output sequence 1。
- 质量评测：完整数据集，`limit=null`，0-shot。
- ARC-C 使用 `acc_norm,none`。
- ARC-E、WinoGrande、MMLU 使用 `acc,none`。
- 表中质量数值为原始准确率乘以 100。
- TTFT speedup 为：

  `BF16 median latency / 当前方法 median latency`

Llama-2 的每个方法保留五个独立的 `measured_0.json` 至 `measured_4.json`，取五次 `elapsed_ms` 的中位数。Llama-3.1 的 warmed protocol 在同一个 engine 中先 warmup 一次，再记录五次 `timed_ms` 并取中位数。

### 3.2 Prefill--Decode

- 速度 workload：batch 8、input sequence 2048、output sequence 64。
- 每个速度点包含一次 warmup 和五次 timed iteration。
- 原始逐次速度记录在 `iterations.csv`。
- 生成质量不是用速度 workload 直接评测：原始 shard `run_config.json` 记录 batch 4、`max_input_tokens=3840`、`max_new_tokens=256`、greedy vLLM generation。
- CNN/DM 和 DialogSum 使用 `rougeL_percent`。
- IWSLT'17 使用 `sacre_bleu`。
- 原始生成样本数：
  - CNN/DM：1000
  - DialogSum：1500
  - IWSLT：333
- 三个速度比均按 latency 越低越好的方式计算：

  `speedup = 同模型 BF16 latency / 当前方法 latency`

## 4. Llama-2 7B-Chat / Prefill-only

### 4.1 Policy 与表格值

| 表格方法 | Policy / 运行 ID | ARC-C | ARC-E | Wino. | MMLU | median ms | TTFT speedup | 证据 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | quality `uniform_dense_bf16`; speed `point_000` | 43.3447 | 72.8114 | 68.2715 | 46.5746 | 1135.0202 | 1.0000 | 完整 |
| Dense NVFP4 | `uniform_dense_nvfp4` | 43.1741 | 71.1279 | 67.7980 | 44.9437 | 664.4216 | 1.7083 | 完整 |
| Sparse BF16 | `uniform_sparse_bf16` | 35.0683 | 63.8468 | 63.8516 | 28.2438 | 740.3935 | 1.5330 | 完整 |
| Sparse NVFP4 | `uniform_sparse_nvfp4` | 23.3788 | 36.6162 | 50.4341 | 23.0879 | 611.5206 | 1.8561 | 完整 |
| W4A16 Marlin | `uniform_marlin_nvfp4` | 43.0887 | 71.8855 | 68.8240 | 45.6559 | 1115.7993 | 1.0172 | 完整 |
| Ours (Max speed) | `point_024` | 27.7304 | 53.1566 | 56.7482 | 23.5080 | 545.2521 | 2.0816 | 完整 |
| Ours (Balanced) | `point_017` | 43.6860 | 70.5808 | 67.3244 | 44.3527 | 652.0115 | 1.7408 | 完整 |

### 4.2 原始文件

Uniform policy 文件：

`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/policies/prefill_only/{p00,p01,p02,p03,p04}.json`

其中 `p00` 至 `p04` 分别对应 BF16、Dense NVFP4、Sparse BF16、Sparse NVFP4、W4A16 Marlin。

Ours policy 文件：

- Balanced：`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/pareto/policies/point_017.json`
- Max speed：`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/pareto/policies/point_024.json`

每个质量值的原始 lm-eval JSON：

`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/pareto/validation/tasks/<运行ID>/{arc_challenge,arc_easy,winogrande,mmlu}/full/result.json`

这里 `<运行ID>` 为上表中的 quality ID。文件内包含最终 metric、0-shot/limit 配置、lm-eval 配置、运行时间、runtime、policy 路径及 policy SHA-256，不依赖 summary 文档。

每个速度值的五次原始 JSON：

`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/pareto/validation/speed/<运行ID>/runs/measured_{0,1,2,3,4}.json`

warmup 原始文件：

`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/pareto/validation/speed/<运行ID>/runs/warmup.json`

BF16 速度使用 `point_000`；其 policy 的 128 个模块全是 Dense BF16，因此与质量侧的 `uniform_dense_bf16` 配置等价。

## 5. Llama-2 7B-Chat / Prefill--Decode

### 5.1 Policy、质量与速度

| 表格方法 | Policy / 运行 ID | CNN/DM | DialogSum | IWSLT | TTFT ms | TPOT ms | E2E ms | 质量证据 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | `p00` | 23.6709 | 21.6884 | 19.2958 | 1096.9024 | 18.9673 | 2291.8432 | **缺原始文件** |
| Dense NVFP4 | `p01` | 24.2732 | 20.5924 | 16.8350 | 668.0662 | 34.6973 | 2853.9937 | **缺原始文件** |
| Sparse BF16 | `p02` | 15.3499 | 13.5395 | 3.8998 | 701.8478 | 20.9353 | 2020.7710 | **缺原始文件** |
| Sparse NVFP4 | `p03` | 15.9628 | 13.5679 | 1.5693 | 679.0296 | 36.7058 | 2991.4962 | 完整；配置标签有条件 |
| W4A16 Marlin | `p04` | 23.7277 | 21.7697 | 18.8523 | 1128.2218 | 12.6605 | 1925.8347 | 完整 |
| Ours (Max speed) | `b8o64009` | 18.5626 | 13.5738 | 2.6026 | 731.9760 | 11.8267 | 1477.0602 | 完整 |
| Ours (Balanced) | `b8o64004` | 23.7983 | 21.4006 | 21.0832 | 1063.1898 | 15.0040 | 2008.4384 | 完整 |

用 BF16 latency 复算后，表中 TTFT / TPOT / E2E speedup 分别为：

| 方法 | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| BF16 | 1.00 | 1.00 | 1.00 |
| Dense NVFP4 | 1.64 | 0.55 | 0.80 |
| Sparse BF16 | 1.56 | 0.91 | 1.13 |
| Sparse NVFP4 | 1.62 | 0.52 | 0.77 |
| W4A16 Marlin | 0.97 | 1.50 | 1.19 |
| Ours (Max speed) | 1.50 | 1.60 | 1.55 |
| Ours (Balanced) | 1.03 | 1.26 | 1.14 |

### 5.2 原始文件

Uniform policy：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/policies/prefill_decode/{p00,p01,p02,p03,p04}.json`

Ours policy：

- Balanced：`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/pareto/candidates/policies/b8o64004.json`
- Max speed：`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/pareto/candidates/policies/b8o64009.json`

逐次速度原始 CSV：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/speed/runs/<运行ID>/iterations.csv`

速度运行配置：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/speed/runs/<运行ID>/metadata.json`

中位数结果 CSV：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/speed/runs/<运行ID>/summary.csv`

`summary.csv` 这里只是对 `iterations.csv` 的复算检查；本审计以 `iterations.csv` 为原始证据。

`p03`、`p04`、`b8o64004`、`b8o64009` 的逐样本生成结果：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/task_quality/results/<运行ID>/{cnn_dm_1000,dsum,IWSLT}/*.jsonl`

对应 metric 文件：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/task_quality/results/<运行ID>/{cnn_dm_1000,dsum,IWSLT}/metrics.json`

### 5.3 证据缺口

BF16、Dense NVFP4、Sparse BF16 的表格质量值仅能定位到：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/task_quality/report/task_pareto_points.csv`

同样的值也出现在上一轮：

`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat/task_quality/report/task_pareto_points.csv`

两处都标记为 `baseline_measured`，但仓库中没有这三组结果对应的逐样本 `jsonl` 或 `metrics.json`。因此它们是“有聚合实测记录、无原始生成输出”，不能满足完全 provenance 的标准。

本轮 `056` 实际保留了一组全 BF16 的新原始结果 `b8o64000`：

- CNN/DM 23.6103
- DialogSum 21.5982
- IWSLT 19.4118

其 policy 的 128 个模块均为 Dense BF16，原始文件位于：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/task_quality/results/b8o64000/{cnn_dm_1000,dsum,IWSLT}/`

这组结果与当前表中的 BF16 23.6709 / 21.6884 / 19.2958 很接近，但并不相同。这进一步说明当前 BF16 质量确实沿用了旧的 aggregate，而不是使用 `056` 中保留下来的新原始结果。

### 5.4 Sparse NVFP4 配置说明

`p03.json` 的全部 128 个模块均配置为：

- prefill：`sparse_nvfp4`
- decode：`dense_nvfp4`

原始 policy：

`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat/policies/prefill_decode/p03.json`

因此当前表格的 “Sparse NVFP4” 应理解为 “Sparse NVFP4 prefill + Dense NVFP4 decode”，而不是全阶段 Sparse NVFP4。

## 6. Llama-3.1 8B-Instruct / Prefill-only

### 6.1 Policy 与表格值

| 表格方法 | Quality ID | Speed ID | ARC-C | ARC-E | Wino. | MMLU | median ms | TTFT speedup | 证据 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | `point_000` in `061` | `point_000` | 55.6314 | 81.9865 | 73.4807 | 68.4019 | 1087.5269 | 1.0000 | 完整 |
| Dense NVFP4 | `p01` in `058` | `p01` in `061` | 51.1092 | 79.2929 | 72.0599 | 63.5024 | 618.8403 | 1.7573 | 完整；分批测量 |
| Sparse BF16 | `p02` in `058` | `p02` in `061` | 38.7372 | 68.1818 | 66.0616 | 42.9640 | 708.1749 | 1.5357 | 完整；分批测量 |
| Sparse NVFP4 | `p03` in `058` | `p03` in `061` | 26.3652 | 45.6650 | 53.2755 | 23.8784 | 564.2376 | 1.9274 | 完整；分批测量 |
| W4A16 Marlin | `p04` in `058` | `p04` in `061` | 53.6689 | 80.9343 | 73.0859 | 66.2014 | 1114.7320 | 0.9756 | 完整；分批测量 |
| Ours (Max speed) | `point_014` | `point_014` | 33.0205 | 61.2795 | 61.0892 | 31.1423 | 481.1632 | 2.2602 | 完整 |
| Ours (Balanced) | `point_007` | `point_007` | 51.4505 | 79.0825 | 73.7964 | 66.5005 | 685.8683 | 1.5857 | 完整 |

`061` 还保留了一组 uniform BF16 calibration median 1088.9263 ms。主表使用 closure 中的 BF16 `point_000` 1087.5269 ms 作为统一分母。两者相差约 0.13%，都是 warmed protocol；这不会导致 Balanced 的异常，也没有混用旧的非 warmed 速度。

### 6.2 原始文件

Uniform policy：

`artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/llama31_8b_instruct/policies/prefill_only/{p00,p01,p02,p03,p04}.json`

`061` 的 calibration 设计文件直接绑定上述 policy：

`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/speed/calibration/design.csv`

Uniform 质量原始 lm-eval JSON：

`artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/llama31_8b_instruct/task_quality/results/<p00-p04>/{arc_challenge,arc_easy,winogrande,mmlu}/full/result.json`

BF16、Balanced 和 Max speed 的原始 lm-eval JSON：

`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/task_quality/results/<point_000|point_007|point_014>/{arc_challenge,arc_easy,winogrande,mmlu}/full/result.json`

Uniform warmed-speed 原始 JSON：

`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/speed/calibration/runs/<p01-p04>.json`

BF16、Balanced、Max speed 的 warmed-speed 原始 JSON：

`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/pareto/closure/speed/<point_000|point_007|point_014>.json`

Ours policy：

- Balanced：`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/pareto/policies/point_007.json`
- Max speed：`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/pareto/policies/point_014.json`

### 6.3 Balanced MMLU 的直接证据

原始文件：

`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/task_quality/results/point_007/mmlu/full/result.json`

关键字段：

- `metrics["acc,none"] = 0.6650049850448654`
- `num_fewshot = 0`
- `limit = null`
- MMLU effective samples 总数为 14,042
- `runtime.policy_json` 指向 `point_007.json`
- `runtime.policy_sha256` 为 `e422940...ea274`
- backend 为 `lm_eval.VLLM/vllm-runtime`
- vLLM 为 `0.11.1.dev0+gb8b302cde.d20260624`
- `max_model_len = 2048`
- 环境信息记录 GPU 为 NVIDIA GeForce RTX 5090
- 启动时间为 `2026-07-21T01:07:26.231279+00:00`

因此 66.50 是 full MMLU、0-shot、当前 `point_007` policy 的结果，不是小样本结果，也不是旧 `point_009` 的结果。

## 7. Llama-3.1 8B-Instruct / Prefill--Decode

### 7.1 Policy、质量与速度

| 表格方法 | Policy / 运行 ID | CNN/DM | DialogSum | IWSLT | TTFT ms | TPOT ms | E2E ms | 证据 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | `p00` | 19.3894 | 13.4815 | 10.7771 | 1150.7013 | 16.5504 | 2193.3788 | 完整 |
| Dense NVFP4 | `p01` | 17.1082 | 9.5353 | 10.5135 | 674.7187 | 31.2199 | 2641.5738 | 完整 |
| Sparse BF16 | `p02` | 6.7571 | 5.1084 | 3.0384 | 753.1926 | 18.7998 | 1937.5784 | 完整 |
| Sparse NVFP4 | `p03` | 4.2491 | 0.4073 | 0.0542 | 635.2193 | 33.8910 | 2770.3551 | 完整 |
| W4A16 Marlin | `p04` | 17.4999 | 9.7512 | 10.5635 | 1171.9508 | 13.0322 | 1992.9818 | 完整 |
| Ours (Max speed) | `point_011` | 14.5948 | 4.2136 | 1.2885 | 661.0322 | 14.0780 | 1547.9456 | 完整 |
| Ours (Balanced) | `point_005` | 18.8374 | 11.4805 | 10.5502 | 821.6359 | 14.6080 | 1741.9374 | 完整 |

用 BF16 latency 复算后，表中 TTFT / TPOT / E2E speedup 分别为：

| 方法 | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| BF16 | 1.00 | 1.00 | 1.00 |
| Dense NVFP4 | 1.71 | 0.53 | 0.83 |
| Sparse BF16 | 1.53 | 0.88 | 1.13 |
| Sparse NVFP4 | 1.81 | 0.49 | 0.79 |
| W4A16 Marlin | 0.98 | 1.27 | 1.10 |
| Ours (Max speed) | 1.74 | 1.18 | 1.42 |
| Ours (Balanced) | 1.40 | 1.13 | 1.26 |

### 7.2 原始文件

Uniform policy：

`artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/policies/prefill_decode/{p00,p01,p02,p03,p04}.json`

Ours policy：

- Balanced：`artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/pareto/policies/point_005.json`
- Max speed：`artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/pareto/policies/point_011.json`

逐次速度原始 CSV：

`artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/speed/runs/<运行ID>/iterations.csv`

速度运行配置：

`artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/speed/runs/<运行ID>/metadata.json`

逐样本生成 JSONL：

`artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/task_quality/results/<运行ID>/{cnn_dm_1000,dsum,IWSLT}/*.jsonl`

对应 metric JSON：

`artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/task_quality/results/<运行ID>/{cnn_dm_1000,dsum,IWSLT}/metrics.json`

这里的 Llama-3.1 `p03` 与 Llama-2 不同：其 policy 确实在 prefill 和 decode 两个阶段都将全部 128 个模块设为 `sparse_nvfp4`，因此 Llama-3.1 的 “Sparse NVFP4” 标签没有 legal-projection 问题。

## 8. 配置一致性检查

### 8.1 模型没有串用

- `054` 与 `056` 只用于 Llama-2 7B-Chat。
- `057`、`058`、`061` 只用于 Llama-3.1 8B-Instruct。
- 表中没有把 Llama-2 的质量和 Llama-3.1 的速度拼在同一半行。

### 8.2 Workload 没有串用

- Prefill-only 的所有速度结果均为 B=8、S=2048、O=1。
- Prefill--Decode 的所有速度结果均为 B=8、S=2048、O=64。
- 每个 speedup 只相对于同模型、同 workload 的 BF16 计算，没有跨模型或跨 workload 归一化。

### 8.3 Llama-3.1 Prefill-only 没有使用旧冷启动速度

表中 Llama-3.1 Prefill-only 速度来自 `061` 的 warmed protocol：

`same_engine_warmup_then_timed_explicit_prefill_phase`

Balanced 的原始速度文件记录一次 warmup 和五次 timed measurement：

`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/pareto/closure/speed/point_007.json`

其中 median 为 685.8683 ms。表中的 1.59 使用同一轮 warmed BF16 1087.5269 ms 复算得到。

### 8.4 质量协议

Prefill-only 原始 `result.json` 记录：

- lm-eval 0.4.12
- 0-shot
- full evaluation (`limit=null`)
- vLLM phase-heterogeneous runtime
- policy JSON 与 SHA-256

Prefill--Decode 的逐样本 JSONL 同时保存 prediction、reference、token 数和 precision log，可用于重新计算 ROUGE-L 或 SacreBLEU，不依赖报告性 summary。每个 shard 的实际生成配置还保存在相邻的 `run_config.json` 中。

Llama-3.1 的后续 prompt-template 配对诊断使用相同的 100 个样本、相同 reference、BF16、greedy decoding、`max_new_tokens=256` 和 `max_input_tokens=3840`，只改变 prompt construction。其原始文件为：

- Legacy manifest：`artifacts/debug/062_llama31_prompt_template_diagnosis/outputs/legacy/run_manifest.json`
- Native manifest：`artifacts/debug/062_llama31_prompt_template_diagnosis/outputs/native/run_manifest.json`
- Legacy 逐样本：`artifacts/debug/062_llama31_prompt_template_diagnosis/outputs/legacy/{cnn_dm_1000,dsum,IWSLT}/answers.jsonl`
- Native 逐样本：`artifacts/debug/062_llama31_prompt_template_diagnosis/outputs/native/{cnn_dm_1000,dsum,IWSLT}/answers.jsonl`
- 从上述逐样本结果计算的指标：`artifacts/debug/062_llama31_prompt_template_diagnosis/report/metrics.json`

100 样本配对结果为：

| 任务 | Legacy | Native |
|---|---:|---:|
| CNN/DM ROUGE-L | 15.6573 | 20.5600 |
| DialogSum ROUGE-L | 12.0200 | 17.7957 |
| IWSLT SacreBLEU | 9.8819 | 27.4388 |

因此主表 Llama-3.1 Prefill--Decode 的绝对质量值应标明是 legacy/common PMPD protocol。它不是“把某一个方法放在不同模板下”造成的不公平，因为该模型的七个方法全部使用同一 protocol；但它确实是跨模型或跨论文结果比较时必须披露的配置差异。

### 8.5 GPU 证据

Prefill-only 的 lm-eval 原始结果内 `pretty_env_info` 明确列出 NVIDIA GeForce RTX 5090。Prefill--Decode 的逐次速度 `metadata.json` 保留 workload 和 runtime protocol，但没有单独写入 GPU 型号；其运行日志按同一 5090 机器的 GPU 编号分片。若需要达到审稿级、完全自包含的硬件 provenance，后续复测应把 GPU name、driver、CUDA、vLLM commit 和 policy SHA-256 一并写入每个 speed run 的 `metadata.json`。

### 8.6 Runtime、KV cache 与 chunked prefill

两条 Prefill--Decode 轨道的 baseline 记录均注明 `phase_hetero_mytest`、vLLM V1、BF16 KV cache、chunked prefill disabled，来源为各自的：

`speed/uniform_baselines.csv`

但单个 `speed/runs/<ID>/metadata.json` 中 `kv_cache_dtype` 为 `null`，没有把 BF16 KV 自包含地重复记录到每个 run；生成质量 shard 的 `run_config.json` 则写的是 `kv_cache_dtype="auto"`。这不是已发现的结果混用，但属于元数据不够自包含。后续复测应在每个原始 run metadata 中记录最终解析后的 KV dtype 与 chunked-prefill 状态。

## 9. 建议的论文处理

在提交前建议做以下最小修正：

1. 保留当前 Llama-3.1 Ours (Balanced) 的 `point_007` 数值；这组结果证据完整且配置一致。
2. 将 Llama-2 Prefill--Decode 的 “Sparse NVFP4” 改成更准确的名称，例如 “Sparse NVFP4 / Dense NVFP4 decode”，或在表下注明 legal projection。
3. 对 Llama-2 Prefill--Decode 的 BF16、Dense NVFP4、Sparse BF16 重新跑生成质量，保留逐样本 JSONL 和 `metrics.json`；在此之前，应把这九个格子标记为 provenance incomplete。
4. 在表注或实验设置中明确 Llama-3.1 Prefill--Decode 使用 legacy/common PMPD prompt；如果论文要强调跨模型绝对质量，则应以 native Llama-3.1 chat template 重跑全部七个方法，而不能只替换 BF16。
5. 如果暂时不重跑 Llama-2 BF16，可考虑使用本轮有完整原始证据的 `b8o64000` 数值 23.61 / 21.60 / 19.41，但 Dense NVFP4 和 Sparse BF16 仍需补测，且替换前应统一决定整组 baseline 的评测批次。

最终判断是：**更新后的 Llama-3.1 Balanced 结果可以使用；当前整表的主要风险不在其 MMLU，而在 Llama-2 三组 inherited generation-quality baseline 的原始证据缺失、一个 Sparse NVFP4 标签没有准确描述 decode 配置，以及 Llama-3.1 generation quality 的 legacy prompt 需要在论文中明确披露。**
