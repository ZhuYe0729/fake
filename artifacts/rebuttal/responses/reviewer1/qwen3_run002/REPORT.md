# Qwen3-8B：真实 8bit baseline 与扩展 DASO

本轮新增 SmoothQuant INT8、原生 FP8 和 SmoothQuant-FP8 对照；DASO 在原五候选上加入 SmoothQuant INT8 与原生 FP8。
所有新策略均先冻结再测独立测试集。新增候选组与五候选控制组使用完全相同的质量拟合和模块实测延迟表。

按实际 prefill ΔNLL≤0.02 筛选已测点：uniform 最佳为 1.911× (uniform_sq_int8)；扩展 DASO 为 1.225× (daso8_prefill_only_00)。
按实际 decode ΔNLL≤0.01 筛选：匹配五候选为 1.433× (daso5_measured_prefill_decode_04)；扩展七候选为 1.366× (daso8_prefill_decode_01)。当前数据不支持加入 8bit 后全面提升的结论。

## 完整 baseline

| 方法 | Prefill ΔNLL | TTFT ms | Prefill 加速 | Decode ΔNLL | E2E ms | E2E 加速 | GPU allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_dense_bf16 | 0.000000 | 1110.5 | 1.000× | 0.000000 | 1991.4 | 1.000× | 20.04 |
| uniform_dense_nvfp4 | 0.037419 | 665.0 | 1.670× | 0.032543 | 2882.6 | 0.691× | 12.89 |
| uniform_sparse_bf16 | 0.279001 | 731.3 | 1.518× | 0.271667 | 1856.3 | 1.073× | 14.38 |
| uniform_sparse_nvfp4 | 0.785806 | 611.1 | 1.817× | 0.791491 | 2710.3 | 0.735× | 13.09 |
| uniform_w4a16_ours | 0.023560 | 1132.9 | 0.980× | 0.019833 | 1904.8 | 1.045× | 10.77 |
| uniform_int8 | -0.007590 | 1400.9 | 0.793× | -0.002363 | 3543.9 | 0.562× | 16.87 |
| uniform_marlin_int4 | 0.050921 | 1122.6 | 0.989× | 0.057993 | 2037.3 | 0.977× | 10.46 |
| uniform_sq_int8 | -0.012790 | 581.1 | 1.911× | -0.002323 | 2082.7 | 0.956× | 14.63 |
| uniform_fp8 | 0.006421 | 660.0 | 1.683× | 0.009928 | 1924.5 | 1.035× | 13.76 |
| uniform_sq_fp8 | 0.007024 | 660.0 | 1.682× | 0.012822 | 1955.1 | 1.019× | 13.76 |

旧七种 uniform baseline 的速度在本轮同卡重新测量；旧基线任务与质量证据复用 001 冻结结果，BF16 test NLL 重新测量校验。三种新增 8bit baseline 均完成完整任务。

## 按实际质量比较候选扩展

下表在实际测试 ΔNLL 不超过给定上限的已测点中选择最高加速。该表用于结果分析，不参与策略选择。

| 场景 | 实际 ΔNLL 上限 | DASO + 8bit | 五候选，同质量拟合/延迟表 | 简单 phase-uniform |
| --- | ---: | --- | --- | --- |
| prefill_only | 0.005 | — | — | — |
| prefill_only | 0.01 | 1.225× (daso8_prefill_only_00) | — | — |
| prefill_only | 0.02 | 1.225× (daso8_prefill_only_00) | 1.163× (daso5_measured_prefill_only_00) | — |
| prefill_only | 0.04 | 1.972× (daso8_prefill_only_02) | 1.163× (daso5_measured_prefill_only_00) | — |
| prefill_only | 0.07 | 1.972× (daso8_prefill_only_02) | 1.651× (daso5_measured_prefill_only_02) | — |
| prefill_only | 0.1 | 1.972× (daso8_prefill_only_02) | 1.791× (daso5_measured_prefill_only_03) | — |
| prefill_only | 0.15 | 2.229× (daso8_prefill_only_04) | 1.993× (daso5_measured_prefill_only_05) | — |
| prefill_only | 0.22 | 2.261× (daso8_prefill_only_05) | 1.993× (daso5_measured_prefill_only_05) | — |
| prefill_only | 0.45 | 2.299× (daso8_prefill_only_07) | 2.079× (daso5_measured_prefill_only_07) | — |
| prefill_decode | 0.005 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 0.903× (phase_fp8_sq_int8) |
| prefill_decode | 0.01 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 1.229× (phase_dense_nvfp4_w4a16_ours) |
| prefill_decode | 0.02 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 1.229× (phase_dense_nvfp4_w4a16_ours) |
| prefill_decode | 0.04 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 1.287× (phase_sq_int8_w4a16_ours) |
| prefill_decode | 0.07 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 1.287× (phase_sq_int8_w4a16_ours) |
| prefill_decode | 0.1 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 1.287× (phase_sq_int8_w4a16_ours) |
| prefill_decode | 0.15 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 1.287× (phase_sq_int8_w4a16_ours) |
| prefill_decode | 0.22 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 1.287× (phase_sq_int8_w4a16_ours) |
| prefill_decode | 0.45 | 1.366× (daso8_prefill_decode_01) | 1.433× (daso5_measured_prefill_decode_04) | 1.287× (phase_sq_int8_w4a16_ours) |

![实测质量与延迟](results/figures/quality_latency_zoom.png)

001 的 HAWQ 与原 DASO 在图中只作为历史参考：本轮 INT8 激活量化与输出缩放使用融合实现，候选集合与成本输入也已变化，不能据此把跨轮差异归因于分配算法。

## 预算满足情况

| 候选组 | 场景 | 测试点 | 超预算点 | 最大超预算 nats/token |
| --- | --- | ---: | ---: | ---: |
| daso8 | prefill_only | 8 | 8 | 0.112124 |
| daso8 | prefill_decode | 6 | 0 | 0.000000 |
| daso5_measured | prefill_only | 9 | 9 | 0.198705 |
| daso5_measured | prefill_decode | 8 | 0 | 0.000000 |

随机 policy holdout 误差不能作为搜索后策略的预算保证；不使用测试结果重新拟合、回选或校正本轮策略。

## 求解离散化诊断

原 2000-bin 全局网格在小误差 8bit 动作上可能过粗。Uniform SmoothQuant-INT8 的 prefill 原始预测 ΔNLL 为 0.001799，离散后为 0.028430；它的原始预测 Linear 成本 443.16 ms，低于预算 0.01 时 DP 选中策略的 497.92 ms，但离散可行域排除了该 uniform 配置。
这与质量代理的预测误差是两个问题。本轮仍使用原求解器和原网格；此处只审计已有训练标签、拟合与离散成本，不修改策略或超参数。详见 [uniform_grid_audit.json](results/uniform_grid_audit.json)。

## 下游任务

| 方法/策略 | WikiText PPL | WinoGrande % | ARC-Easy % | ARC-Challenge % | MMLU % | CNN/DM ROUGE-L | DialogSum ROUGE-L | IWSLT BLEU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_dense_bf16 / prefill_only | 13.49 | 68.03 | 83.46 | 56.91 | 72.87 | — | — | — |
| uniform_dense_bf16 / prefill_decode | — | — | — | — | — | 22.81 | 26.96 | 40.13 |
| uniform_dense_nvfp4 / prefill_only | 14.16 | 67.96 | 82.37 | 54.61 | 70.97 | — | — | — |
| uniform_dense_nvfp4 / prefill_decode | — | — | — | — | — | 22.35 | 27.08 | 40.20 |
| uniform_sparse_bf16 / prefill_only | 20.13 | 64.25 | 73.91 | 44.45 | 54.10 | — | — | — |
| uniform_sparse_bf16 / prefill_decode | — | — | — | — | — | 21.45 | 21.29 | 32.30 |
| uniform_sparse_nvfp4 / prefill_only | 38.15 | 55.41 | 53.87 | 28.16 | 30.80 | — | — | — |
| uniform_sparse_nvfp4 / prefill_decode | — | — | — | — | — | 13.53 | 5.32 | 0.91 |
| uniform_w4a16_ours / prefill_only | 13.82 | 69.22 | 83.54 | 56.48 | 71.92 | — | — | — |
| uniform_w4a16_ours / prefill_decode | — | — | — | — | — | 22.55 | 27.44 | 40.12 |
| uniform_int8 / prefill_only | 13.45 | 67.72 | 82.28 | 57.08 | 72.55 | — | — | — |
| uniform_int8 / prefill_decode | — | — | — | — | — | 22.79 | 27.27 | 39.79 |
| uniform_marlin_int4 / prefill_only | 14.27 | 65.35 | 80.09 | 52.65 | 71.16 | — | — | — |
| uniform_marlin_int4 / prefill_decode | — | — | — | — | — | 21.96 | 26.35 | 40.02 |
| uniform_sq_int8 / prefill_only | 13.40 | 68.98 | 83.54 | 57.17 | 72.61 | — | — | — |
| uniform_sq_int8 / prefill_decode | — | — | — | — | — | 22.70 | 27.18 | 39.32 |
| uniform_fp8 / prefill_only | 13.57 | 67.96 | 83.16 | 55.72 | 72.85 | — | — | — |
| uniform_fp8 / prefill_decode | — | — | — | — | — | 22.79 | 26.97 | 40.07 |
| uniform_sq_fp8 / prefill_only | 13.61 | 68.43 | 83.71 | 57.00 | 72.83 | — | — | — |
| uniform_sq_fp8 / prefill_decode | — | — | — | — | — | 22.83 | 27.03 | 40.01 |
| daso8_prefill_only_02 / prefill_only | 14.03 | 67.64 | 81.99 | 56.23 | 71.51 | — | — | — |
| daso8_prefill_only_04 / prefill_only | 16.45 | 65.90 | 79.08 | 49.83 | 66.03 | — | — | — |
| daso8_prefill_only_07 / prefill_only | 21.18 | 65.43 | 72.52 | 44.45 | 58.72 | — | — | — |
| daso8_prefill_decode_01 / prefill_decode | — | — | — | — | — | 22.52 | 26.80 | 39.49 |
| daso8_prefill_decode_02 / prefill_decode | — | — | — | — | — | 22.56 | 26.63 | 39.36 |
| daso8_prefill_decode_04 / prefill_decode | — | — | — | — | — | 22.03 | 25.81 | 39.20 |
| daso8_prefill_decode_05 / prefill_decode | — | — | — | — | — | 22.07 | 25.61 | 36.53 |
| daso5_measured_prefill_only_02 / prefill_only | 14.68 | 66.69 | 81.40 | 54.44 | 68.96 | — | — | — |
| daso5_measured_prefill_only_04 / prefill_only | 16.35 | 65.67 | 77.78 | 50.51 | 61.41 | — | — | — |
| daso5_measured_prefill_only_08 / prefill_only | 27.26 | 61.09 | 65.82 | 36.43 | 46.11 | — | — | — |
| daso5_measured_prefill_decode_02 / prefill_decode | — | — | — | — | — | 22.37 | 26.43 | 39.56 |
| daso5_measured_prefill_decode_04 / prefill_decode | — | — | — | — | — | 21.67 | 25.44 | 38.58 |
| daso5_measured_prefill_decode_07 / prefill_decode | — | — | — | — | — | 21.22 | 19.52 | 10.66 |
| phase_fp8_w4a16_ours / prefill_decode | — | — | — | — | — | 22.72 | 27.14 | 40.29 |
| phase_sq_int8_w4a16_ours / prefill_decode | — | — | — | — | — | 22.73 | 27.06 | 39.99 |

除原先冻结的预算代表点外，追加了每组速度 sweep 中实测最快且尚未覆盖的点的完整任务。选择仅使用 TTFT/E2E，不使用 NLL 或下游分数；本轮唯一追加点为 daso8_prefill_decode_01。原预算代表点全部保留。

## 实现与协议

- INT8：真实 `torch._int_mm`，INT8×INT8→INT32，融合 Triton BF16 epilogue；M<32 补到 32 后丢弃多余输出。
- FP8：真实 vLLM SM120 CUTLASS E4M3 W8A8 GEMM，BF16 输出。动态 per-token 激活、per-output-channel 权重；注意力和 KV 保持 BF16。
- 旧 sparse BF16 使用 2:4；sparse NVFP4 使用后端要求的 paired 4:8。W4A16 NVFP4 与 native INT4 group128 Marlin 分别标记，不混用。
- SmoothQuant：采用[官方实现](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py)的 alpha=0.5 公式与 Llama-like 范围，Qwen3 QKV 和 gate/up 平滑，o/down 不平滑。训练集 128×2048 tokens 收集通道最大值。
- 为支持不同模块/阶段独立候选，x/s 融入候选内部激活量化，而非改动共享 RMSNorm；所有实际开销计入计时。平滑 FP8 明确标记为扩展对照。
- 初始 128 项 shape/kernel 数值与有限值检查通过；最终搜索延迟来自 GPU3 的 64 项完整 forward、多轮同步 wall-time 测量。NVFP4 包装层不支持 CUDA Graph，失败尝试与原始单次计时保留。
- 两组搜索共用 90 training / 30 policy holdout 标签，其中每场景 72 个原标签复用、48 个新增标签；质量模型、正则、4 个深度桶与 2000-bin 保守 DP 沿用原实现。
- 模块延迟包含平滑、激活量化、GEMM、输出缩放与 Python dispatch；完整模型速度额外包含阶段切换、权重传输、注意力等全部开销。
- 正式速度同一物理 GPU3、B8/S2048/O1 或 O64、2 warmups + 5 timed iterations、固定 3GiB BF16 KV；生成固定 6GiB KV、原冻结输入、greedy 256 tokens。
- 完整 likelihood 任务使用 001 已归档 Arrow 样本，直接本地加载；61 个任务数据配置已核验。原 Hub 离线失败日志保留。
- CPU 双表示缓存、GPU allocator peak、reserved peak 和主机 RSS 均保留在原始 metadata / CSV。导出体积与运行时内存分别报告。

完整策略及候选比例：[candidate_usage.json](results/candidate_usage.json)；完整数表：[summary_final.csv](results/summary_final.csv)；完整性审计：[final_audit.json](results/final_audit.json)。

## 重现

依赖与模型哈希见 configs；原模型、patched vLLM 和 CUTLASS wrapper 路径见 code/common.py，源 lineage 见 reference/sources.json。数据链接依赖同项目下冻结的 runs/001。
先在临时目录准备两种 canonical SparseGPT state（可复用 001，否则用 prepare_sparse.py --method ... --output-root ... 重建），再执行 prepare_8bit.py、profile_stable.py、daso_pipeline.py prepare。
并行启动 experiment_queue.py 的 cal0/cal1/cal2/primary 四个 role，分别绑定物理 GPU0/4/7/3；统一 OMP_NUM_THREADS=4、OPENBLAS_NUM_THREADS=1、HF_HUB_OFFLINE=1、HF_DATASETS_OFFLINE=1。队列已完成步骤自动跳过；失败日志不覆盖。
质量阶段使用 quality_queue.py 的共享 policy claim，先补齐 NLL，再做完整任务；可在 GPU3 另启动 quality_queue.py --name phase_helper --wait-primary，等待全部统一测速完成后加入质量评测。
速度 sweep 完成且 phase_helper 运行后，在 GPU3 执行 supplement_fastest.py，记录仅按实测速度选择的额外完整任务；脚本等待当前策略结束后补测，再恢复共享队列。
本次 residual_queue.py、switch_quality_queue.py 等脚本仅记录实际调度与恢复，干净复现使用标准 role 即可。临时模型二进制可以按归档 policy + activation_scales.pt + canonical state 重建；代码、日志、结果、文档全部保留。
