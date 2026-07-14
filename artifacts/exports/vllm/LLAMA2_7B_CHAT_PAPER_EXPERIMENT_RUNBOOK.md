# Llama2-7B-Chat：vLLM 论文实验端到端运行手册

本文档面向把本仓库迁移到另一台机器、由另一位实验执行者完成一个模型全套论文实验的情形。它以 Llama2-7B-Chat 为具体示例；新模型应复制流程和检查点，而不是复制 Llama2 的数值、策略 JSON 或 kernel profile。

目标产物不是某个“预测最优策略”，而是两类场景下由**实测速度和实测质量**组成的论文表、Pareto 图和可追溯的原始文件：

| 场景 | workload | 主任务质量 |
|---|---|---|
| prefill-only | `B=8, input=2048, output=1` | ARC-Challenge `acc_norm`，1172 条 |
| prefill-decode | `B=16, input=2048, output=80` | CNN/DM-1000 ROUGE-L、DialogSum-1500 ROUGE-L、IWSLT-333 SacreBLEU |

最终 Llama2 示例结果位于
[`ours/llama2-7b-chat/pareto_summary/`](ours/llama2-7b-chat/pareto_summary/)。该目录应包含
`all_measured_results.csv`、`summary.md` 和五张图；它是论文选择点时唯一应直接引用的汇总入口。

## 0. 总览：先理解依赖关系

```text
模型 + vLLM extension + kernel profile
       │
       ├── uniform baseline：压缩 → 导出专用 checkpoint → vLLM 测速/任务
       │
       ├── ours 建模：WikiText local error + policy NLL → quality proxy
       │             kernel roofline/residual profile → latency surrogate
       │
       ├── 约束求解：quality budget → 离散 phase/layer policy
       │
       ├── 闭环：导出 phase-hetero checkpoint → vLLM E2E / NLL
       │
       └── 少量前沿点：真实生成任务 → 合并、非支配筛选、论文图表
```

不要将速度预测、NLL 预测或 solver 输出直接画成论文最终曲线。它们只用于搜索和筛选；最终横轴必须来自 vLLM 实测，最终质量轴必须来自实测 WikiText NLL 或真实下游任务。

## 1. 迁移前准备

### 1.1 必须迁移的代码与数据

- 本仓库的 `fake/kernels/cutlass/cutlass_wrapper/`：kernel profile 和 CUTLASS wrapper。
- 修改后的 vLLM 仓库：当前示例为 `/home/agent/wja/project/my/cospaq/test/vllm/`。它包含 phase-heterogeneous quantization、`prepare_next_prefill()` 和 PMPD evaluator。
- 原始模型：`/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf`。
- 离线数据/缓存：WikiText-2 calibration blocks，CNN/DM、DialogSum、IWSLT、ARC-Challenge，以及 BERTScore 模型缓存。
- 两个 Python 环境：压缩/建模环境（当前 `cospaq`）和带 CUDA extension 的 vLLM 环境（当前 `vllm`）。两者的 PyTorch/CUDA ABI 必须和编译 extension 一致。

建议用环境变量消除机器路径差异：

```bash
export REPO=/path/to/cospaq/fake
export VLLM_ROOT=/path/to/patched-vllm
export MODEL=/path/to/Llama-2-7b-chat-hf
export CUTLASS_ROOT=$REPO/fake/kernels/cutlass/cutlass_wrapper
export VLLM_PYTHON=/path/to/miniconda/envs/vllm/bin/python
export COSPAQ_PYTHON=/path/to/miniconda/envs/cospaq/bin/python
export HF_DATASETS_OFFLINE=1      # 若数据已预下载；否则先关闭并完成下载
```

### 1.2 上机 smoke test（先做，失败时不要开始批量实验）

1. 在 vLLM 环境中导入 `vllm`，并验证 CUDA capability 与 wrapper extension 可见。
2. 使用 dense BF16 模型执行一次 `B=8,S=2048,O=1` 的 `LLM.generate()`。
3. 导出一个已知 uniform NVFP4 checkpoint，加载后执行同样请求。
4. 导出一个小的 ours phase-hetero policy checkpoint，确认 `prefill → decode → prepare_next_prefill` 连续执行不会重载权重、不会 OOM。

可复用的最小基线 benchmark 在
[`baselines/llama2-7b-chat/scripts/benchmark_vllm_scenarios.py`](baselines/llama2-7b-chat/scripts/benchmark_vllm_scenarios.py)；ours 的 phase-hetero benchmark 在
[`ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py`](ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py)。

记录 GPU 型号、driver、CUDA、PyTorch、vLLM commit、CUTLASS wrapper commit、`nvidia-smi`。这些不是附注：kernel profile 和速度模型在换 GPU/driver 后必须重建。

## 2. 统一方法 baseline：先完成并冻结

### 2.1 方法与 vLLM 接口

要比较的 uniform 方法：`dense_bf16`、`dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4`、`marlin_nvfp4`。

uniform checkpoint 使用各自的专用 vLLM quantization interface，不使用 `phase_hetero_mytest`：

| 方法 | `config.json` 的 `quant_method` |
|---|---|
| dense BF16 | 无 `quantization_config` |
| dense NVFP4 | `nvfp4_mytest` |
| sparse BF16 | `sparse_bf16_mytest` |
| sparse NVFP4 | `sparse_nvfp4_mytest` |
| Marlin NVFP4 | `marlin_nvfp4_mytest` |

对应实现在 [`baselines/llama2-7b-chat/scripts/export_uniform_vllm.py`](baselines/llama2-7b-chat/scripts/export_uniform_vllm.py)。这点很重要：不要把 baseline checkpoint 的 `quant_method` 改成 phase-hetero；后者只用于 ours 的 per-layer/per-phase dispatch。若要以 phase-hetero lifecycle 对 uniform 做闭环重测，可以构造所有层与两个 phase 都选同一 method 的 policy；该结果应标注为 *phase-hetero-wrapped uniform closure*，不得与原生 uniform 数字混为同一口径。

### 2.2 压缩并导出 uniform checkpoint

在压缩环境执行（实际参数可按新模型显存调整，但校准数据、随机种子、方法定义要落盘）：

```bash
cd $REPO
CUDA_VISIBLE_DEVICES=0 $COSPAQ_PYTHON \
  artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/prepare_uniform_compressed.py \
  --methods sparse_bf16,dense_nvfp4,sparse_nvfp4,marlin_nvfp4 --gpu 0

CUDA_VISIBLE_DEVICES=0 $COSPAQ_PYTHON \
  artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/export_uniform_vllm.py \
  --model-path $MODEL --methods dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4
```

检查：每个 `artifacts/exports/vllm/baselines/llama2-7b-chat/checkpoints/uniform_*` 必须含 `config.json`、`model.safetensors` 和对应 manifest。直接检查 config 的 `quant_method`，不要只看目录名。

### 2.3 Baseline 速度

原生 baseline speed runner 对每个 method/scenario 建一个 vLLM `LLM`，固定随机 token prompt，`temperature=0`、`ignore_eos=True`、CUDA synchronize 包围 `generate()`；默认 1 warmup + 5 measurements：

```bash
CUDA_VISIBLE_DEVICES=0 $VLLM_PYTHON \
  artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/benchmark_vllm_scenarios.py \
  --methods dense_bf16,dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4 \
  --scenarios prefill_only,prefill_decode --warmup-iters 1 --iters 5 \
  --gpu-memory-utilization 0.85
```

固定并写入结果的 workload 参数是：

- prefill-only：`max_model_len=2049`、`max_num_seqs=8`；
- prefill-decode：`max_model_len=2128`、`max_num_seqs=16`；
- tensor parallel = 1，BF16 activation，`enforce_eager=True`，关闭 prefix caching。

`gpu_memory_utilization` 不是可随意混用的性能参数。Llama2 历史实验先出现 `.9/.8/.85` 多种口径；新一轮论文图应在 feasibility probe 后选定一个最终值（推荐先 `.85`），再以同一值重测**所有**要进入同一张图的 uniform 与 ours 点。

输出：`baselines/.../results/speed/{iterations.csv,summary.csv,metadata.json}`。保留每次时间样本，报告 median；不要只保留平均值。

### 2.4 Baseline 下游任务精度

使用相同 vLLM checkpoint 和确定性 generation，跑三任务：

```bash
bash artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/run_all_quality.sh
$VLLM_PYTHON artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/summarize_results.py
```

相关 evaluator 是 [`baselines/llama2-7b-chat/scripts/pmpd_vllm_eval.py`](baselines/llama2-7b-chat/scripts/pmpd_vllm_eval.py)。CNN/DM 只使用冻结的 1000-example subset，不能与 full test 的结果混写。每种方法、数据集应保存 generation JSONL、run config 和 `metrics.json`。

**Gate A：** baseline 五方法在两个 workload 均可运行；速度 summary 和三任务 metrics 完整；checkpoint 的实际 `quant_method` 与预期一致。此时冻结 baseline 版本及其软件栈，不要在 ours 实验中悄悄更新 kernel 或 vLLM。

## 3. 建立质量与速度代理（搜索用，不是最终结果）

### 3.1 先定义可路由 module/action space

Llama2 的运行时单位是每层四种 fused module：`qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj`，共 `32 × 4 = 128` 个 group。对 prefill-decode，每组 action 是 `(prefill_method, decode_method)`；合法性完全由 backend support 决定。

当前动作语义和设计细节见 [`dev/094_llama2_prefill_decode_pareto_design.md`](../../../dev/094_llama2_prefill_decode_pareto_design.md)。至少应重新生成：

- 当前 GPU/kernel stack 的合法 kernel shapes；
- dense NVFP4 ↔ Marlin 的合法 phase pair；
- decode small-`M` 不支持的 sparse action；
- materialization/conversion 开销；
- OOM/显存 feasibility probe 结果。

不要从 Llama2 policy JSON 复制给另一模型；它编码的是具体层数、fused shape、local sensitivity 和 runtime capability。

### 3.2 质量代理：WikiText 校准而非下游分数拟合

质量模型以固定 WikiText teacher-forced NLL 为监督：

1. 在 100 个固定、2048-token blocks 上，收集 dense 与每个 kernel action 的 local module error / residual；
2. 设计覆盖 method、layer bucket、fused type、prefill/decode 分配的 calibration policies；
3. 对每个 policy 计算真实 pooled target：
   - prefill-only：`ΔNLL_prefill`；
   - prefill-decode：`ΔNLL_prefill + 80 × ΔNLL_decode`；
4. 用 local error 的 method × layer-bucket × fused-type 汇总特征拟合正性 global calibrated proxy；
5. 保留独立 holdout policy，检查 MAE/RMSE 和 Spearman 排序相关。

Llama2 的历史建模入口和产物在
[`ours/llama2-7b-chat/pareto/nll_modeling_v1/`](ours/llama2-7b-chat/pareto/nll_modeling_v1/)，最终选型与失败尝试的解释在上面的 design 文档。换模型时必须重新采样 calibration policies、重新拟合，不能沿用系数。

质量模型的合格标准不是“训练误差低”，而是 holdout 对策略排序足够稳定；若 holdout 排序不佳，优先增加覆盖性校准策略、减少 feature 自由度或调整 layer bucket，不要直接改用昂贵下游任务分数作训练标签。

### 3.3 速度代理：kernel profile-calibrated roofline + policy E2E calibration

底层模型使用 `fake/kernels/cutlass/cutlass_wrapper/modeling` 的 kernel-specific predictor。它为每个 kernel 建立：

```text
calibrated roofline base (effective FLOPs / effective bandwidth / launch floor)
  + local shape residual correction
  + hard shape-support filtering
  → per-module phase latency
  → policy raw latency sum + conversion/materialization cost
  → monotone policy-level E2E calibrator
```

新机器必须在目标 GPU 上重新 profile kernel shape matrix；新模型还要加入其 fused shapes 和两个 phase 的 `M`。然后抽取覆盖慢/中/快区间的若干合法策略，用正式 vLLM runner 测 E2E，拟合单调 `raw latency → E2E latency` 校正。Llama2 的相关历史实现/参考为：

- `artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/scripts/fit_monotone_e2e_calibrator.py`；
- `artifacts/debug/037_llama2_prefill_only_pareto/scripts/fit_e2e_calibrator.py`；
- `artifacts/debug/039_llama31_8b_instruct_prefill_decode_pareto/scripts/fit_monotone_e2e_calibrator.py`（更完整的 recent template）。

**Gate B：** 保存 kernel profile、quality holdout metrics、E2E calibration residual 和 OOM probe。若任一项明显异常，先修 backend/support 或增补 calibration，不进入 solver。

## 4. 约束求解并物化候选

对每个质量预算 `ε` 求解：

```text
min predicted calibrated E2E latency(policy)
subject to predicted ΔNLL(policy) ≤ ε
          action is kernel-supported and memory-feasible
```

因每个 fused module 必须选择一个 action，使用 multiple-choice knapsack dynamic programming；扫多个 `ε`，去除重复和预测支配点。参考 solver：

```bash
$COSPAQ_PYTHON artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/scripts/solve_predicted_pareto.py --help
```

输出应包括每个 policy 的 JSON/CSV、预算、预测 NLL、raw speed、calibrated speed、每层 method assignment 和合法性诊断。Llama2 历史输出在：

- prefill-only：`artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/prefill_only/pareto/`；
- prefill-decode：`artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/prefill_decode/pareto/`。

选择闭环点时至少覆盖：identity、近无损、高质量、中段、dense-NVFP4 附近、最高速端点；不要只测 solver 给出的单一 max-speed policy。

## 5. 闭环实测：checkpoint、速度、NLL

### 5.1 导出 ours phase-hetero checkpoint

ours checkpoint 使用 `phase_hetero_mytest`，其 manifest 保存每个 fused module 的 prefill/decode method：

```bash
CUDA_VISIBLE_DEVICES=0 $COSPAQ_PYTHON \
  artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py \
  --policy-json /path/to/policy.json --model-path $MODEL \
  --output-dir /path/to/closure/checkpoints/point_XXX --force --prune
```

导出后检查 `config.json` 的 `quant_method=phase_hetero_mytest`、`phase_hetero_policy.json`、`phase_hetero_manifest.json` 和 `model.safetensors`。这一步将 solver 的逻辑策略变为实际可运行的 fused vLLM 权重；未导出/未加载 checkpoint 的 predictor 验证不算闭环。

### 5.2 速度测试

prefill-only 可复用 Llama2 的 1 warmup + 5 个 fresh-process 测量方式：

```bash
bash artifacts/debug/037_llama2_prefill_only_pareto/scripts/run_calibration_point.sh POINT_ID GPU_ID
```

参考 runner 是 [`benchmark_phase_baseline_one.py`](ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py)。所有显示点都要保存 warmup 和五个 measured JSON；聚合时取 median。

prefill-decode 必须验证 phase 切换。推荐使用一个进程内复用同一 `LLM` 的 continuous protocol（避免把加载权重时间混进每次请求）：6 warmup + 5 measured E2E，测量前后调用 vLLM phase state 的 `prepare_next_prefill()`。可参考：

- [`benchmark_phase_hetero.py`](ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py)；
- [`039/.../run_pareto_closure_speed.sh`](../../debug/039_llama31_8b_instruct_prefill_decode_pareto/scripts/run_pareto_closure_speed.sh)。

若继续使用 Llama2 历史 fresh-process runner，也可以，但 uniform 和 ours 必须统一口径；在最终表中明确标为 `fresh-process`，且不要与 continuous 数字做细微差异比较。

每个候选同时运行 WikiText NLL。Llama2 参考：

```bash
bash artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/scripts/run_actual_nll_point.sh POINT_ID GPU_ID
```

这里的 pooled NLL 是选择/验证质量 proxy 的统一尺度；它不是替代下游任务。

### 5.3 OOM 与异常速度处理

- OOM：记录 checkpoint、workload、`gpu_memory_utilization`、日志和方法分配；先使用同一 workload 下较小 KV-cache headroom 重测。不能只删除 OOM 点而不说明。
- 异常速度：保留每次重复样本；若跨度极大，标记 stalled/outlier，重测而不是只选择最快的一次。
- 中间补点：若要展示更平滑的曲线，可在相邻质量预算之间再求解/测试；表中标注其 timing protocol 与 formal closure 是否相同。

**Gate C：** 对每个候选，存在 checkpoint manifest、原始速度样本、实测 ΔNLL。以这些实测数重新做 Pareto/non-dominated filter；预测 frontier 只作候选来源。

## 6. 真实生成任务：只测少量闭环候选，但测完整

对最终前沿中若干代表点（通常 high-quality / balanced / fast / max-speed）以及五个 uniform 方法，使用 vLLM 跑三个任务。大数据集可按 shard 并行，但单 shard 内要复用一个 vLLM `LLM`，不能每个 batch 重载 checkpoint。

Llama2 的脚本入口：

```bash
# 启动某个 ours 点的 task shards；按机器上的可用 GPU 分配 shard。
$VLLM_PYTHON artifacts/debug/037_llama2_prefill_only_pareto/scripts/run_task_quality.py --help

# 单 shard runner 的真实 vLLM phase-switch 配置：
bash artifacts/debug/037_llama2_prefill_only_pareto/scripts/run_task_quality_shard.sh ...

# shard 完成后合并、去重、计算 metrics：
$VLLM_PYTHON artifacts/exports/vllm/ours/llama2-7b-chat/scripts/merge_pmpd_shards.py --help
```

完整性检查：CNN/DM=1000、DialogSum=1500、IWSLT=333 条；每项 `empty_predictions=0`；合并后的 JSONL、`run_config.json`、`metrics.json` 都存在。若输出 token 上限/stop token/temperature 与 baseline 不同，必须全部重跑，不能只补 ours。

## 7. 形成论文产物

将数据源收敛为一个新目录，避免论文图直接依赖许多 debug root。Llama2 的当前聚合器：

```bash
MPLCONFIGDIR=/tmp/mplconfig $COSPAQ_PYTHON \
  artifacts/debug/037_llama2_prefill_only_pareto/scripts/make_paper_result_bundle.py
```

其输出为：

```text
artifacts/exports/vllm/ours/llama2-7b-chat/pareto_summary/
├── all_measured_results.csv
├── summary.md
├── pareto_prefill_only_arc_challenge.png
├── pareto_prefill_decode_cnn_dm.png
├── pareto_prefill_decode_dsum.png
├── pareto_prefill_decode_iwslt.png
└── pareto_prefill_decode_wikitext_nll.png
```

聚合表必须列出：scenario、family（uniform/ours）、policy、推荐标签、绝对 E2E、相对 dense speedup、速度来源、NLL、各任务分数、任务状态。保留所有实测点；`recommended` 只能是后续作者挑论文展示点的建议，不能通过隐藏不利点制造曲线。

图的规则：uniform 与 ours 使用不同 marker/color；legacy 或 screened-stall 数字必须显式区分；图的速度和质量只使用实测字段；若某点没有下游任务分数，不将其画进该任务的曲线。

## 8. 最终审计清单

在交付论文数字前逐项确认：

- [ ] 每个图中的 uniform 方法实际使用专用 `quant_method`，不是无意中套用了 ours dispatcher。
- [ ] 每个 ours 点的 policy、manifest、checkpoint、speed raw samples、NLL 文件和 task metrics 可回溯。
- [ ] 同一张速度图中的 `B/S/O`、GPU、driver、vLLM commit、runner lifecycle、`gpu_memory_utilization` 一致；不一致则分组或重测。
- [ ] 所有 speedup 以该场景 dense BF16 的同口径 median E2E 为分母。
- [ ] `prefill-only` 不拿 decode 任务分数充当其质量轴；`prefill-decode` 的 NLL 权重与 output length 一致。
- [ ] 论文中的最终曲线是实测 non-dominated union frontier，而非 solver 的 predicted frontier。
- [ ] 保存运行命令、环境版本、随机种子、日志和原始 JSONL；结果包可从原始文件重新生成。

## 9. 常见问题与处理顺序

| 现象 | 优先检查 |
|---|---|
| uniform checkpoint 加载后没有预期加速 | `config.json` 的专用 `quant_method`、manifest、extension import、真实 GPU capability |
| ours checkpoint 速度很慢或 phase 不切换 | `phase_hetero_policy.json`、`prepare_next_prefill()` 调用、phase runtime 日志、是否误用了原生 vLLM |
| 特定策略 OOM | KV-cache headroom、NVFP4 workspace、`max_model_len/max_num_batched_tokens`，然后重新做统一口径测量 |
| NLL 代理 holdout 很差 | calibration policy 覆盖、layer bucket、local error 的 method 映射；不要先拟合下游任务 |
| 任务测试特别慢 | shard 数据集、每 shard 一个常驻 LLM、batch size、禁用反复 checkpoint load；保持和 baseline 相同 generation 配置 |
| 图看起来 ours 没有覆盖 uniform | 检查是否遗漏 dense-NVFP4 邻域/高质量点；补求解与实测点，不要删除 uniform 点 |

## 10. 参考索引

- 设计与公式：[`dev/094_llama2_prefill_decode_pareto_design.md`](../../../dev/094_llama2_prefill_decode_pareto_design.md)
- Llama2 solver/初始闭环：[`debug/034_llama2_7b_chat_wikitext_pareto_solver/`](../../debug/034_llama2_7b_chat_wikitext_pareto_solver/)
- Llama2 prefill-only 收口：[ `debug/037_llama2_prefill_only_pareto/`](../../debug/037_llama2_prefill_only_pareto/)
- Llama2 final results：[ `ours/llama2-7b-chat/pareto_summary/`](ours/llama2-7b-chat/pareto_summary/)
- Llama3.1 的更近期 phase-continuous 运行范例：[ `debug/039_llama31_8b_instruct_prefill_decode_pareto/scripts/`](../../debug/039_llama31_8b_instruct_prefill_decode_pareto/scripts/)

迁移到新模型时，优先复制第 0--8 节的过程和审计规则；第 2 节的具体 checkpoint 与第 3--5 节的 calibration/profile/policy 必须从零开始生成。
