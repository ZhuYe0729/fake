# Llama2-7B-Chat 两场景论文实验：可执行复现流程

本文档是当前最终口径的复现入口。它不是某台机器的运维手册，而是让后续 Agent 在新机器上从模型与代码出发，逐阶段复现 Llama2-7B-Chat 的 baseline、建模、约束求解、真实闭环和论文产物。每一节均给出实际命令、完成标志和常见失败的判断方式。

> 旧的 `artifacts/exports/vllm/LLAMA2_7B_CHAT_PAPER_EXPERIMENT_RUNBOOK.md` 记录了早期 B=16/O=80 和 native-uniform 口径，不再用于复现最终结果。

## 1. 冻结的实验定义

| 项目 | prefill-only | prefill-decode |
|---|---:|---:|
| batch | 8 | 8 |
| input tokens | 2048 | 2048 |
| output tokens | 1 | 64 |
| vLLM / KV cache | V1 / BF16 (`auto`) | V1 / BF16 (`auto`) |
| chunked prefill / prefix cache | off / off | off / off |
| `gpu_memory_utilization` | 同图固定 | 0.80 |
| 正式测速 | 1 warmup + 5 measured | 1 warmup + 5 measured |

两类方法使用同一 `phase_hetero_mytest` runtime。uniform 是 method map 在所有层、所有 phase 都相同的特殊策略；它不是另一套 native runner。这样运行时、phase lifecycle、在线激活量化及测量代码均一致。

| ID | prefill | decode | 说明 |
|---|---|---|---|
| p00 | dense BF16 | dense BF16 | 速度和 NLL reference |
| p01 | dense NVFP4 | dense NVFP4 | W4A4，含在线激活量化 |
| p02 | canonical sparse BF16 | canonical sparse BF16 | SparseGPT/Hessian 校准 |
| p03 | canonical sparse NVFP4 | dense NVFP4 | decode M=8 不支持 sparse NVFP4 的合法投影 |
| p04 | Marlin W4A16 | Marlin W4A16 | 仅权重量化 |

prefill-only 质量：WikiText PPL、WinoGrande `acc`、ARC-Easy `acc`、ARC-Challenge `acc_norm`、MMLU `acc`。prefill-decode 使用 CNN/DM 1000、DialogSum 1500、IWSLT 333 并保留所有指标。PMPD generation 保留 legacy/common prompt；这是与 PMPD 参考实现一致的模型内公平比较，不用 native chat template 替换主结果。

## 2. 一次性配置与预检

```bash
cd /path/to/cospaq/fake
cp artifacts/debug/063_llama2_two_scenario_reproduction_workflow/config.example.env \
   artifacts/debug/063_llama2_two_scenario_reproduction_workflow/config.env
vim artifacts/debug/063_llama2_two_scenario_reproduction_workflow/config.env
source artifacts/debug/063_llama2_two_scenario_reproduction_workflow/config.env

$COSPAQ_PYTHON artifacts/debug/063_llama2_two_scenario_reproduction_workflow/scripts/preflight.py \
  --output "$COSPAQ_RUN_ROOT/validation/preflight.json"
```

`$COSPAQ_PYTHON` 必须来自 `cospaq` 环境，负责 SparseGPT、local error、拟合和 solver；`$VLLM_PYTHON` 必须来自 `vllm` 环境，负责 phase checkpoint、真实 NLL、速度和生成任务。当前机器示例在 `config.current.env`，不能原样复制到另一台机器。

登录节点无 GPU 时先加 `--no-gpu`，计算节点再执行不带该参数的预检。`ok` 必须为 `true`，尤其检查两个 Python 的 torch/CUDA、32 层模型结构、patched vLLM custom quant method、磁盘与 compute capability。

必须同步代码、patched vLLM、原始模型、离线数据集与 tokenizer/BERTScore 缓存。不要同步旧临时 checkpoint 作为复现输入。目标 GPU/CUDA/PyTorch ABI 变化时，重新编译 CUTLASS wrapper并重做速度 profile。安装说明见 `fake/kernels/cutlass/cutlass_wrapper/INSTALL.md`。

## 3. 创建干净实验树

bootstrap 只复制 72 个固定校准 policy、manifest 和固定 WikiText sample；不会复制 canonical checkpoint、NLL、速度或拟合系数：

```bash
$COSPAQ_PYTHON artifacts/debug/063_llama2_two_scenario_reproduction_workflow/scripts/bootstrap_repro.py
$COSPAQ_PYTHON artifacts/debug/063_llama2_two_scenario_reproduction_workflow/scripts/validate_stage.py bootstrap
```

中断后可用 `bootstrap_repro.py --resume`，它会逐文件 hash 校验而非静默覆盖。不要把 `COSPAQ_RUN_ROOT` 指向 054、056 或 060。

```bash
export PREFILL_EXP="$COSPAQ_RUN_ROOT/prefill_only"
export DECODE_EXP="$COSPAQ_RUN_ROOT/prefill_decode"
export COSPAQ_CUTLASS_WRAPPER="$COSPAQ_REPO_ROOT/fake/kernels/cutlass/cutlass_wrapper"
```

## 4. Canonical sparse 权重

稀疏权重不是运行时直接 `--prune` 得到的：`sparse_bf16` 是 SparseGPT/Hessian 校准后的 structured sparse BF16；`sparse_nvfp4` 保存校准后的 BF16 sparse state，NVFP4 只在 phase exporter 中进行一次最终量化和 pack。

```bash
export COSPAQ_EXPERIMENT_DIR="$PREFILL_EXP"
CUDA_VISIBLE_DEVICES=0 $COSPAQ_PYTHON \
  artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/prepare_canonical_sparse.py \
  --gpu 0

export COSPAQ_CANONICAL_DIR="$PREFILL_EXP/canonical/prepared"
$COSPAQ_PYTHON artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/verify_canonical_sparse.py
$COSPAQ_PYTHON artifacts/debug/063_llama2_two_scenario_reproduction_workflow/scripts/validate_stage.py canonical
```

禁止给 exporter 传 `--prune`，也禁止把已量化 sparse NVFP4 state 再次 NVFP4 量化。每个 checkpoint 均运行：

```bash
$VLLM_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/verify_canonical_checkpoint.py \
  --policy /path/to/policy.json --checkpoint /path/to/checkpoint
```

它会拒绝 policy 不一致、`prune != false` 或 canonical provenance 缺失。

## 5. 串行预热 extension

多个进程同时首次 JIT 编译会造成锁等待、初始化显存竞争，甚至表现为 GPU 利用率为零。批量实验前独占一张卡串行预热：

```bash
export COSPAQ_EXPERIMENT_DIR="$PREFILL_EXP"
export TORCH_EXTENSIONS_DIR="$COSPAQ_EXT_CACHE_ROOT/prewarm"
CUDA_VISIBLE_DEVICES=0 $VLLM_PYTHON \
  artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/prewarm_phase_extensions.py
```

确认 dense NVFP4、Marlin、sparse BF16、sparse NVFP4 module 均可加载。换 PyTorch/CUDA、vLLM wheel/source 或清 cache 后必须重做。

## 6. Prefill-only 完整流程

### 6.1 Local error

```bash
export COSPAQ_EXPERIMENT_DIR="$PREFILL_EXP"
CUDA_VISIBLE_DEVICES=0 $COSPAQ_PYTHON \
  artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/collect_canonical_sparse_local_errors.py \
  --method sparse_bf16 --gpu 0 --blocks 16 > "$PREFILL_EXP/local_sparse_bf16.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 $COSPAQ_PYTHON \
  artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/collect_canonical_sparse_local_errors.py \
  --method sparse_nvfp4 --gpu 1 --blocks 16 > "$PREFILL_EXP/local_sparse_nvfp4.log" 2>&1 &
pid1=$!
wait $pid0
wait $pid1
$COSPAQ_PYTHON artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/assemble_local_error_table.py
```

完成标志为 `local_errors/module_method_errors.csv` 的 layer/module/method 全覆盖；sparse 特征来自 canonical wrapper，不是 direct prune。

### 6.2 72-policy 真实 vLLM NLL 与拟合

```bash
export COSPAQ_EXPERIMENT_DIR="$PREFILL_EXP"
export COSPAQ_PREFILL_EXPERIMENT_ROOT="$COSPAQ_RUN_ROOT"
$VLLM_PYTHON artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/run_all.py \
  --gpus 0,1,2,3 --blocks 100
$VLLM_PYTHON artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/merge_nll.py
$COSPAQ_PYTHON artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/fit_and_report.py
```

完成标志为 `nll/prefill_only.csv` 恰好 72 行，p00 是同 runtime reference；查看 train/holdout 的 `reports/quality/metrics.json` 和预测散点。若拟合失效，先排查 runtime、sample hash、canonical state、重复 pack，再增加 calibration policy；不直接用下游分数拟合。

### 6.3 速度代理、solver、真实闭环

CUTLASS `KernelLatencyPredictor` 是 calibrated roofline + shape residual 的 per-module proxy；目标 GPU 变化时按 Llama2 shape 在 `fake/kernels/cutlass/cutlass_wrapper/modeling` 重做 profile。论文速度必须实测。

```bash
export COSPAQ_EXPERIMENT_DIR="$PREFILL_EXP"
$COSPAQ_PYTHON artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/solve_canonical_pareto.py

CUDA_VISIBLE_DEVICES=0 $VLLM_PYTHON \
  artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/validate_canonical_pareto_point.py \
  --point 7 --runs 5 --blocks 100 --tmp-root "$PREFILL_EXP/temporary_checkpoints"
```

solver 输出 `pareto/pareto_points.csv` 与 `pareto/policies/point_*.json`。闭环点应覆盖 dense BF16、近无损、dense NVFP4 附近、中段和 max-speed，而非只测一个点。

prefill-only 的 uniform p00-p04 也用同一闭环入口逐点独占 GPU 测速：

```bash
for id in p00 p01 p02 p03 p04; do
  CUDA_VISIBLE_DEVICES=0 $VLLM_PYTHON \
    artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/validate_canonical_pareto_point.py \
    --policy-json "$PREFILL_EXP/policies/prefill_only/$id.json" \
    --label "uniform_$id" --runs 5 --blocks 100 \
    --tmp-root "$PREFILL_EXP/temporary_checkpoints"
done
```

下游任务示例：

```bash
CUDA_VISIBLE_DEVICES=0 $VLLM_PYTHON \
  artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_pareto_tasks.py \
  --policy-json "$PREFILL_EXP/pareto/policies/point_007.json" --label point_007 \
  --experiment-root "$PREFILL_EXP" \
  --canonical-sparse-bf16-state "$COSPAQ_CANONICAL_DIR/sparse_bf16/model.pt" \
  --canonical-sparse-nvfp4-state "$COSPAQ_CANONICAL_DIR/sparse_nvfp4/model.pt" \
  --tasks wikitext,winogrande,arc_easy,arc_challenge,mmlu \
  --temporary-root "$PREFILL_EXP/task_temporary"
```

对 p00-p04 运行同一入口，保证 uniform 与 ours runtime 完全一致。正式任务不设置 `--limit`；smoke 才用小 limit。

## 7. Prefill-decode 完整流程

056 脚本通过 `scenario.py` 读取统一配置：

```bash
export COSPAQ_EXPERIMENT_DIR="$DECODE_EXP"
export COSPAQ_CANONICAL_DIR="$PREFILL_EXP/canonical/prepared"
export COSPAQ_MODEL_PATH
export COSPAQ_VLLM_ROOT
```

### 7.1 逐 phase local error 与 action support

```bash
$COSPAQ_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/run_local_errors.py \
  --gpus 0,1,2,3
$COSPAQ_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/audit_speed_actions.py
```

检查 `local_errors/prefill_*.csv`、`local_errors/decode_*.csv` 和 `speed/action_support.csv`。decode M=8 的 sparse NVFP4 必须标为 unsupported，不能等 solver 生成后再因 runtime 失败而删点。

### 7.2 真实 prefill + decode NLL

```bash
$VLLM_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/run_canonical_nll.py \
  --gpus 0,1,2,3 --blocks 100 --gpu-memory-utilization 0.80 \
  --startup-stagger-seconds 12
$VLLM_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/merge_canonical_nll.py
```

teacher forcing capacity 必须容纳 `8*(2048+64)=16896` tokens。完成标志是 72 个 raw JSON、export provenance、phase trace 均存在；trace 包含 `enter_decode` 和 `apply_decode`；`nll/prefill_decode.csv` 恰好 72 行。

### 7.3 Phase-aware 拟合、速度公式与约束求解

```bash
$COSPAQ_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/build_coverage_holdout.py
$COSPAQ_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/fit_phase_quality.py \
  --split-json "$DECODE_EXP/policies/prefill_decode/coverage_holdout.json" \
  --report-name quality_coverage_holdout
$COSPAQ_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/solve_pareto.py
```

质量 proxy 同时使用 prefill/decode local feature。速度 raw 公式是：

```text
sum(prefill module latency at M=8*2048)
  + 63 * sum(decode module latency at M=8)
```

这不是另建一套速度模型，而是相同 `KernelLatencyPredictor` 在不同 phase M 和重复次数下的组合。若换 GPU，两场景都重新 profile 并用少量 E2E 点复核。

### 7.4 uniform 与 ours 的统一测速

uniform p00-p04 同样经 phase exporter 与 `benchmark_phase_hetero.py`：

```bash
for id in p00 p01 p02 p03 p04; do
  bash artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/run_speed_policy.sh \
    "$id" "$DECODE_EXP/policies/prefill_decode/$id.json" 0
done
```

上面故意串行：正式速度不要多卡并发，以免 CPU→GPU 带宽、CPU 调度或外部任务污染。对 solver 点运行：

```bash
$VLLM_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/run_pareto_speed.py \
  --gpus 0 --policies b8o64000,b8o64001,b8o64002,b8o64003,b8o64004
$COSPAQ_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/summarize_pareto_speed.py
```

每个点保留 `iterations.csv`、`summary.csv`、`metadata.json`。同图只接受 B=8/S=2048/O=64、BF16 KV、0.80 memory util、1+5 repeats。异常样本先检查外部 GPU 进程并整点重测，不能只挑最快一次。

### 7.5 三个生成任务

选中的 closure 点可以多卡分 shard；每个 shard 一个 fresh vLLM 进程，但同一 shard 内复用已加载 LLM，不会每条样本重载权重：

```bash
$VLLM_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/run_pareto_task_quality.py \
  --gpus 0,1,2,3 \
  --policies b8o64000,b8o64001,b8o64002,b8o64003,b8o64004 \
  --shard-size 360 --iwslt-shard-size 100 --batch-size 4 \
  --gpu-memory-utilization 0.75

$VLLM_PYTHON artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/merge_pareto_task_quality.py \
  --policies b8o64000,b8o64001,b8o64002,b8o64003,b8o64004
```

uniform p00-p04 也放在 `speed/runs/<id>/checkpoint` 并用相同 PMPD runner。合并前严格检查 CNN/DM=1000、DialogSum=1500、IWSLT=333；残缺 shard 不得成为完整结果。

## 8. 最终表与 Pareto 图

060 是最终 schema 参考，不是新实验标签输入。把新 run 的 measured speed、NLL、task metrics 与 policy provenance 汇成一行一个 point 的 CSV。参考脚本：

```bash
$COSPAQ_PYTHON artifacts/debug/060_two_model_two_scenario_result_consolidation/scripts/rebuild_source_summaries.py
$COSPAQ_PYTHON artifacts/debug/060_two_model_two_scenario_result_consolidation/scripts/rebuild_prefill_arc_easy_acc.py
$COSPAQ_PYTHON artifacts/debug/060_two_model_two_scenario_result_consolidation/scripts/rebuild_decode_tables.py
```

这些脚本默认写 060；新 run 应复制脚本并显式改输入/输出，不能覆盖 060。最终产物必须满足：

- uniform p00-p04 在表格前五行，ours 每点一行；
- 每行含 policy、预测 NLL/速度、实测 NLL/速度和所有任务指标；
- 每个指标一张 uniform + ours Pareto 图；
- 标注 max-speed 与 balanced 推荐点，但保留全部点供最终选择；
- 保存环境、commit、GPU、protocol、source path 和异常点排除理由。

论文图横轴只能是 measured E2E speedup，纵轴只能是 measured task metric/NLL。预测曲线只能作为 model validation 单独展示。

## 9. 四卡快速闭环（非论文数据）

p00/p01/p02/p71 覆盖 dense、dense NVFP4、canonical sparse BF16 和综合 phase/method 混合。完整 canonical 代价大时，当前机器验证可以只在 smoke 会话复用已通过 hash/provenance 验证的 canonical state：

```bash
export COSPAQ_CANONICAL_DIR=/path/to/verified/canonical/prepared
bash artifacts/debug/063_llama2_two_scenario_reproduction_workflow/scripts/run_smoke_matrix.sh
$COSPAQ_PYTHON artifacts/debug/063_llama2_two_scenario_reproduction_workflow/scripts/validate_stage.py smoke
```

smoke 每点只用 2 个 NLL blocks、1 warmup+2 repeats，证明导出、canonical provenance、prefill、phase switch 与落盘可运行；不得引用它作为论文数字。

## 10. 故障排查顺序

1. 初始化 OOM：先查外部/残留进程和 worker 的 `CUDA_VISIBLE_DEVICES`，不要立即改 workload。
2. GPU 利用率 0：看 extension build log/lock；停止并串行预热，避免共享正在写的 cache。
   若 `fake/kernels/cutlass/cutlass_wrapper/artifacts/torch_extensions/lock` 超过 30 分钟，先确认没有 `ninja/nvcc/c++` 进程；仅在确认是中断遗留后删除该 lock，再串行预热。不要在编译仍运行时删锁。
3. dense NVFP4 与 Marlin 精度相同：检查是否误走 Transformers evaluator或遗漏在线激活量化。
4. sparse 质量异常：检查 SparseGPT canonical provenance 和 `prune=false`；direct-prune 旧 checkpoint 作废。
5. 速度跳变：核对 B/S/O、chunked prefill、KV dtype、memory util、runner、外部进程和每次原始样本。
6. 多个 prefill wave：最终 B=8/O=64 用于规避 B=16/O=80 的显存/调度问题；不启用 chunked prefill，也不使用 FP8 KV 主结果。
7. 任务无 GPU：看 shard JSONL 是否增长、tokenizer/BERTScore 是否加载、磁盘是否满；完整 shard 不重跑。
8. 磁盘不足：只删除有 policy + canonical provenance 可重建的临时 checkpoint；保留 policy、manifest、hash、raw measurement、generation 与 summary。

## 11. 完整参考与审计

- 054：prefill-only 完整过程；
- 056：prefill-decode B=8/O=64 完整过程；
- 060：两模型两场景论文汇总。

```bash
$COSPAQ_PYTHON artifacts/debug/063_llama2_two_scenario_reproduction_workflow/scripts/validate_stage.py retained
```

retained audit 只证明历史完整产物结构和行数闭合，不把历史结果伪装成 fresh run。实际验证范围见 `VALIDATION_REPORT.md`。
