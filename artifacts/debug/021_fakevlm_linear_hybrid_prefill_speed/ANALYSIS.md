# FakeVLM Linear Hybrid Prefill 速度分析

## 范围

本次 debug 运行测量了 FakeVLM 在真实 FakeClue 图像/文本输入下的 pure prefill 速度：

- 前向路径：`model(**inputs, use_cache=False)`。
- Batch size：`1, 2, 4, 8, 16`。
- 序列长度：经过 FakeVLM processor padding/truncation 后为 `1024` 个 text token。
- GPU：物理卡 `0,1`。
- 计时：`warmup=3`，`iters=10`。
- 压缩目标：仅 FakeVLM language-model 的 linear 层。Vision tower、multimodal projector、attention softmax、norms、embeddings 和 output head 不参与 hybrid policy。

所有 policy 应用均已完成，`skipped_linear_count=0`。

## 端到端结果

| Batch | Best uniform | Best uniform ms | Manual ms | Manual speedup | Latency-model ms | Latency-model speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `uniform_sparse_bf16` | 65.807 | 64.374 | 1.022x | 60.203 | 1.093x |
| 2 | `uniform_sparse_nvfp4` | 103.283 | 105.339 | 0.980x | 100.745 | 1.025x |
| 4 | `uniform_sparse_nvfp4` | 204.053 | 192.356 | 1.061x | 192.638 | 1.059x |
| 8 | `uniform_sparse_bf16` | 433.177 | 404.644 | 1.071x | 405.282 | 1.069x |
| 16 | `uniform_sparse_bf16` | 863.468 | 804.428 | 1.073x | 804.235 | 1.074x |

Latency-model policy 在大多数 batch 下是最优或并列最优的 hybrid。Manual-profile policy 在 batch 2 略差，因为其 shape 级别的计时为 `11008 x 4096` 形状选择了 `dense_nvfp4`，而全模型 best uniform 是全 `sparse_nvfp4`。

## Uniform 基线

| Batch | Dense BF16 ms | Dense NVFP4 ms | Sparse BF16 ms | Sparse NVFP4 ms |
|---:|---:|---:|---:|---:|
| 1 | 105.198 | 70.988 | 65.807 | 69.889 |
| 2 | 183.540 | 110.486 | 114.460 | 103.283 |
| 4 | 356.164 | 218.082 | 215.560 | 204.053 |
| 8 | 675.033 | 486.160 | 433.177 | 458.955 |
| 16 | 1318.521 | 990.030 | 863.468 | 927.380 |

Uniform sparse 方法在此 workload 下本身已经很强，这意味着 per-linear hybrid 选择相对于 best uniform 基线的提升空间有限。

## 策略组成

FakeVLM 暴露了 224 个选中的 language linear 层，仅涉及三种形状：

| Shape `(out_features, in_features)` | Count | 典型角色 |
|---|---:|---|
| `(4096, 4096)` | 128 | attention 投影 |
| `(11008, 4096)` | 64 | MLP up/gate 投影 |
| `(4096, 11008)` | 32 | MLP down 投影 |

Manual-profile policies：

| Batch | Backend 数量 |
|---:|---|
| 1 | `sparse_bf16:128`, `sparse_nvfp4:96` |
| 2 | `sparse_bf16:160`, `dense_nvfp4:64` |
| 4 | `sparse_bf16:160`, `sparse_nvfp4:64` |
| 8 | `sparse_bf16:160`, `sparse_nvfp4:64` |
| 16 | `sparse_bf16:160`, `sparse_nvfp4:64` |

Latency-model policies：

| Batch | Backend 数量 |
|---:|---|
| 1 | `sparse_bf16:160`, `sparse_nvfp4:64` |
| 2 | `sparse_nvfp4:192`, `sparse_bf16:32` |
| 4 | `sparse_bf16:160`, `sparse_nvfp4:64` |
| 8 | `sparse_bf16:160`, `sparse_nvfp4:64` |
| 16 | `sparse_bf16:160`, `sparse_nvfp4:64` |

在 batch 4 及以上，两条路线收敛到相同的 shape 级别决策：

- `(4096, 4096)` -> `sparse_bf16`
- `(11008, 4096)` -> `sparse_nvfp4`
- `(4096, 11008)` -> `sparse_bf16`

## 为什么相对于 Best Uniform 的加速比很小

这一结果对于 FakeVLM 是预期的，并不与之前的 LLaMA 结果矛盾。

主要的比较陷阱不仅仅是基线，还包括测量范围。之前的 `artifacts/results/benchmarks/hybrid/manual/prefill_only` 结果是 `M=16384` 下模块级 linear 延迟汇总；而本次 FakeVLM 运行测量的是全模型 prefill 前向。前者本质上是"linear-only"，后者则包含了所有非线性和多模态工作。

之前来自 `artifacts/results/benchmarks/hybrid/manual/prefill_only` 的 manual pure-prefill linear 汇总结果：

| Model | 范围 | Dense BF16 ms | Best uniform | Best uniform ms | Hybrid ms | Hybrid vs dense | Hybrid vs best uniform | Hybrid policy |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LLaMA-2-7B | linear-only aggregate | 908.333 | `sparse_bf16` | 468.078 | 413.905 | 2.195x | 1.131x | `sparse_bf16:160`, `sparse_nvfp4:64` |
| LLaMA-3.1-8B | linear-only aggregate | 984.436 | `sparse_bf16` | 506.285 | 405.372 | 2.428x | 1.249x | `sparse_bf16:160`, `sparse_nvfp4:64` |
| Qwen3.5-9B | linear-only aggregate | 972.656 | `sparse_bf16` | 512.769 | 427.241 | 2.277x | 1.200x | `sparse_bf16:136`, `sparse_nvfp4:96`, `marlin_nvfp4:16` |
| FakeVLM | full-model prefill | 1318.521 | `sparse_bf16` | 863.468 | 804.235 | 1.640x | 1.074x | `sparse_bf16:160`, `sparse_nvfp4:64` |

你对 LLaMA-2 的计算是正确的：`2.1945 / 1.9406 = 1.1309x` over best uniform。这比 FakeVLM 的 `1.0737x` over best uniform 要大。最重要的原因是上表中的 LLaMA/Qwen 是 linear-only 汇总，而 FakeVLM 的数据是全模型端到端 prefill。

之前的 prefill-decode 示例不是同一种 workload：

| Model | 场景 | Dense BF16 E2E ms | Best uniform | Best uniform E2E ms | Predictor hybrid E2E ms | Hybrid vs dense | Hybrid vs best uniform | Predictor policy |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LLaMA-2-7B | `normal_02`, `batch=1,input=16384,output=256` | 9103.932 | `marlin_nvfp4` | 7572.967 | 7364.498 | 1.236x | 1.028x | `marlin_nvfp4:128`, `dense_nvfp4/marlin_nvfp4:96` |
| Qwen3.5-9B | `normal_01`, `batch=1,input=16384,output=32` | 4136.767 | `sparse_bf16` | 3646.444 | 3682.884 | 1.123x | 0.990x | `marlin_nvfp4:56`, `dense_nvfp4/marlin_nvfp4:128`, `bf16:64` |

这些 normal 场景包含 decode，可以使用 `marlin_nvfp4` 或 `dense_nvfp4/marlin_nvfp4` 拆分 policy。本次 FakeVLM 运行有意排除了 decode，因此无法受益于 prefill/decode 拆分。

1. FakeVLM 的 language linear 形状种类非常少。

   Policy 只能在三种重复形状之间选择。一旦 best uniform 方法已经接近其中两种形状的最优 backend，hybrid 的剩余优化空间就很小。

2. Best uniform 基线本身已经是 sparse 且很强的。

   在每个 batch 中，best uniform 要么是 `uniform_sparse_bf16`，要么是 `uniform_sparse_nvfp4`。Hybrid 不是在跟 dense BF16 比较，而是在跟一个已经很强的单一 backend sparse 基线比较。

3. 端到端 FakeVLM prefill 包含了大量不在选中 language linear 层范围内的工作。

   Policy 不替换 vision tower、multimodal projector、attention softmax、norms、embeddings、output head、框架开销或输入/输出数据搬运。Linear-only 汇总计时显示的增益比最终端到端增益更大：

   | Batch | Manual linear speedup vs best uniform linear | Manual E2E speedup | Latency-model linear speedup vs best uniform linear | Latency-model E2E speedup |
   |---:|---:|---:|---:|---:|
   | 1 | 1.167x | 1.022x | 1.121x | 1.093x |
   | 2 | 1.132x | 0.980x | 1.036x | 1.025x |
   | 4 | 1.155x | 1.061x | 1.050x | 1.059x |
   | 8 | 1.132x | 1.071x | 1.188x | 1.069x |
   | 16 | 1.136x | 1.073x | 1.121x | 1.074x |

   Linear-only 的改进被 FakeVLM 前向中未被替换的部分稀释了。

4. 之前"较大"的 LLaMA/Qwen 效果来自于 linear-only 测量。

   在 `benchmarks/hybrid/manual/prefill_only` 中，LLaMA/Qwen 的 `prefill_ms` 是选中 linear 组延迟之和。它有意排除了 attention softmax、RoPE、norms、residuals、embeddings、调度/框架开销以及任何 VLM 图像路径。这使得 hybrid 增益能够直接转化为报告中的数字。而此处的 FakeVLM benchmark 测量的是真实模型前向，因此同样的 linear 层改进被非线性和多模态工作稀释了。

## Linear 与非 Linear 延迟占比

下表使用与 policy 构建相同的 shape 级 microbench 汇总来估计选中 linear 层的延迟，然后与实测的全模型端到端延迟进行比较。这是一个近似值，但对于理解实测前向中有多少部分不在选中 linear 层范围内是有用的。

Pure prefill 对比：

| Model / method | E2E ms | 估计选中 linear ms | Linear 占比 | 非 linear / 未建模 ms | 非 linear 占比 |
|---|---:|---:|---:|---:|---:|
| LLaMA-2 dense BF16, manual prefill table | 908.333 | 908.333 | 100.0% | 0.000 | 0.0% |
| LLaMA-2 best uniform `sparse_bf16`, manual prefill table | 468.078 | 468.078 | 100.0% | 0.000 | 0.0% |
| LLaMA-2 hybrid, manual prefill table | 413.905 | 413.905 | 100.0% | 0.000 | 0.0% |
| Qwen3.5 dense BF16, manual prefill table | 972.656 | 972.656 | 100.0% | 0.000 | 0.0% |
| Qwen3.5 best uniform `sparse_bf16`, manual prefill table | 512.769 | 512.769 | 100.0% | 0.000 | 0.0% |
| Qwen3.5 hybrid, manual prefill table | 427.241 | 427.241 | 100.0% | 0.000 | 0.0% |
| FakeVLM dense BF16 | 1318.521 | 917.192 | 69.6% | 401.329 | 30.4% |
| FakeVLM best uniform `sparse_bf16` | 863.468 | 481.236 | 55.7% | 382.232 | 44.3% |
| FakeVLM latency-model hybrid | 804.235 | 423.546 | 52.7% | 380.688 | 47.3% |

两点值得注意：

- LLaMA/Qwen manual prefill table 在构造上就是 linear-only 的，因此其报告的 hybrid 增益不会被非 linear 模型工作稀释。
- FakeVLM 的全模型 prefill 在 hybrid 替换后仍有约 `381 ms` 的非 linear/未建模延迟。这几乎是实测 `804 ms` 前向的一半。这种 Amdahl 式的稀释正是为什么同样的 `sparse_bf16:160 + sparse_nvfp4:64` 策略在 LLaMA linear-only 表中给出 `1.13x` over best uniform，而在 FakeVLM 全模型 prefill 中仅为 `1.07x`。

## 解读

- 最优实测点：`latency_model`，batch 16，`804.235 ms`，`19.895 samples/s`，`1.074x` over best uniform baseline。
- 最稳定的 policy：`sparse_bf16:160 + sparse_nvfp4:64`，两条路线在 batch `4, 8, 16` 均使用此策略。
- Best uniform 方法仍然具有竞争力；任何论文/报告都应同时展示 hybrid 相对于 dense BF16 和 best uniform 的加速比，以避免夸大收益。
- 对于本次 FakeVLM prefill-only 的范围，hybrid 的主要价值在于相对于强 sparse uniform 基线提供适度但可复现的增益，而非 LLaMA 那样的大幅跃升。

## 文件

- 完整速度表：`speed/prefill_speed.csv`
- 清理后的最新汇总：`summary/prefill_speed_summary.csv`
- 每个 batch 的 policy：`policies/<family>/batch_<N>/policy.csv`
- 候选表：`candidates/<family>/batch_<N>.csv`
- Prefill-decode 后续工作：`TODO_prefill_decode.md`