# Llama2-7B-Chat / RTX Pro 6000 / prefill-only

这是一个独立、可恢复的论文实验目录。历史 debug bundle 只用于确定实验设计；运行时的
policy、sample、canonical state、proxy、checkpoint provenance、测量和图表均位于本目录。

## 冻结口径

- 模型：Llama2-7B-Chat，32 层，BF16。
- workload：B=8、input=2048、output=1。
- runtime：patched vLLM V1 `phase_hetero_mytest`，BF16 KV，关闭 prefix cache 与 chunked prefill。
- uniform p00--p04 和 ours 使用同一 exporter/runtime。
- 正式速度：同一张独占 GPU，1 warmup + 5 measured，主指标取五次 median。
- NLL calibration：72 个固定 policy、100 个固定 WikiText block；p00--p53 拟合，p54--p71 holdout。

## 目录

- `inputs/`：冻结的 policy 副本和非 sparse local-error 输入。
- `scripts/`：本实验全部专用代码，不调用历史 debug 脚本。
- `runs/experiment/`：canonical、NLL、profile、Pareto、closure、任务结果和临时状态。
- `results/`：最终表、图和 source summary。
- `validation/`：机器可读预检和阶段审计。
- `cache/`：任务数据 manifest；数据 payload 可由 `COSPAQ_TASK_CACHE` 指向本机共享缓存。

模型、CUTLASS wrapper 和 patched vLLM 是基础依赖，不复制到实验目录。

## 配置和预检

```bash
cd /root/workspaces/cospaq/fake
source artifacts/debug/064_llama2_pro6000_prefill_only/config.current.env

$COSPAQ_COSPAQ_PYTHON \
  artifacts/debug/064_llama2_pro6000_prefill_only/scripts/run_stage.py preflight --no-gpu

# 在 GPU 可见的会话中再次执行：
$COSPAQ_COSPAQ_PYTHON \
  artifacts/debug/064_llama2_pro6000_prefill_only/scripts/run_stage.py preflight
```

预检记录 fake/vLLM commit、dirty diff、解释器版本、patched runtime、模型、数据、磁盘和 GPU。
dirty vLLM 不会被静默清理，但其 diff 会进入 provenance。

当前 vLLM 0.11 的 V1 会在参数解析后无条件重开 chunked prefill。本目录的
`scripts/vllm_compat.py` 仅在实验进程内恢复冻结口径，不修改共享 vLLM checkout；NLL、测速和
任务脚本会在引擎构造后再次断言该值为 `False`，并把证据写入每个结果 JSON。

## 执行顺序

以下命令均可在中断后重跑；已有产物只有通过完整性检查才会跳过。

```bash
RUNNER=artifacts/debug/064_llama2_pro6000_prefill_only/scripts/run_stage.py

$COSPAQ_COSPAQ_PYTHON $RUNNER bootstrap
$COSPAQ_COSPAQ_PYTHON $RUNNER canonical
$COSPAQ_COSPAQ_PYTHON $RUNNER prewarm --speed-gpu 0
$COSPAQ_COSPAQ_PYTHON $RUNNER local-errors --gpus 0,1
$COSPAQ_COSPAQ_PYTHON $RUNNER smoke --gpus 0,1
$COSPAQ_COSPAQ_PYTHON $RUNNER nll --gpus 0,1
$COSPAQ_COSPAQ_PYTHON $RUNNER fit
$COSPAQ_COSPAQ_PYTHON $RUNNER profile --speed-gpu 0
$COSPAQ_COSPAQ_PYTHON $RUNNER solve

# 必须在 GPU 0 独占窗口串行运行：
$COSPAQ_COSPAQ_PYTHON $RUNNER closure --speed-gpu 0

$COSPAQ_COSPAQ_PYTHON $RUNNER select-tasks
```

数据准备阶段允许联网；当前机器配置将数据 payload 写入共享的
`/root/.cache/huggingface`，并把 manifest 保留在本目录 `cache/`。若直连失败，使用用户提供的本地代理：

```bash
$COSPAQ_COSPAQ_PYTHON $RUNNER task-data --use-local-proxy
```

数据准备结束后恢复配置中的离线模式，再运行：

```bash
$COSPAQ_COSPAQ_PYTHON $RUNNER tasks --gpus 0,1
$COSPAQ_COSPAQ_PYTHON $RUNNER consolidate
$COSPAQ_COSPAQ_PYTHON $RUNNER validate
```

## 阶段完成标志

- bootstrap：72 个 policy，sample shape `(100, 2049)`，SHA-256 为
  `4c859a5b657834d501ba08b1e212c92dc0a7aec638e9ec67437caf11fc0f52dc`。
- canonical：两个约 13.5 GB state；sparse BF16 为 SparseGPT 2:4，sparse NVFP4 为
  SparseGPT pairwise 4:8 且 `sparse_nvfp4_prequant_only=true`。
- local error：两种 sparse method 各 224 行，然后与本目录冻结的 non-sparse 特征合并。
- NLL：`runs/experiment/calibration/nll/prefill_only.csv` 恰好 72 行。
- profile：solver provenance 中的 predictor root 必须指向本目录。
- closure：uniform p00--p04 和所有 solved point 都有 100-block NLL 与 1+5 speed raw files。
- tasks：selection 中每个点均有五个完整 task result。
- final：`results/complete_results.csv` 和六张 measured Pareto 图。

## 严重错误防护

1. 不向 exporter 传 `--prune`；所有 sparse checkpoint 必须显式使用本目录 canonical state。
2. sparse NVFP4 canonical state 保留 BF16 sparse 权重，只在 exporter 中进行一次最终量化。
3. 不读取 CUTLASS 默认 modeling；solver 必须显式读取本目录 Pro 6000 predictor。
4. extension 首次编译必须串行 prewarm。发现旧 lock 时先确认没有 `ninja/nvcc/c++` 进程。
5. 正式测速不多卡并发、不挑最快样本；外部任务污染时整点重测并保留原因。
6. smoke/limit 结果不得进入最终表；完整任务不设置 `--limit`。
7. mixed policy 的 quality feature 必须逐层归入 method/bucket/type，不能把一个 bucket 的聚合误记到首层方法。
8. canonical NVFP4 到 CUTLASS/Marlin layout 是模型加载前的一次性权重转换，不计入 module-forward predictor 或正式 generate timing。
