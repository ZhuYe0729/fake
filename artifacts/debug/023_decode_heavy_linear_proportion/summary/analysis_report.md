# Decode-Heavy 场景 Linear 时间占比分析报告

## 研究问题

在 decode 为主的场景下（短 prefill + 长 decode），nn.Linear 层的时间占比如何变化？
与之前的 prefill-heavy 研究 (022) 对比有何不同？

## 测试方法

- 模型: Qwen3.5-2B, 4B, 9B (dense BF16)
- GPU: RTX 5090 (5, 6, 7 号卡)
- Batch size: 1, 4, 16
- 输入 token: 4, 16, 64, 256（短 prefill）
- 输出 token: 128, 256, 512（长 decode）
- Breakdown 仅测 output=128（前序研究已证明 decode linear% 与输出长度无关）

## 核心发现

### 1. Decode 阶段 Linear 占比 Top 10

| 模型 | Batch | 输入 | 输出 | Prefill Linear% | Decode Linear% | Decode ms/tok |
|---|---:|---:|---:|---:|---:|---:|
| 9b | 1 | 4 | 128 | 12.3% | 34.5% | 27.350060940539745 |
| 9b | 1 | 16 | 128 | 11.8% | 31.4% | 27.401697275957723 |
| 9b | 4 | 4 | 128 | 12.9% | 30.6% | 28.13732494744729 |
| 9b | 1 | 64 | 128 | 14.8% | 30.5% | 27.27594754977489 |
| 9b | 1 | 256 | 128 | 7.3% | 29.2% | 27.421756522111068 |
| 9b | 4 | 16 | 128 | 14.9% | 29.1% | 28.217697990597703 |
| 9b | 16 | 256 | 128 | 61.4% | 29.1% | 28.25047647371067 |
| 2b | 16 | 64 | 128 | 22.3% | 28.9% | 20.882988178823876 |
| 4b | 16 | 256 | 128 | 46.0% | 28.7% | 28.959070423456627 |
| 9b | 16 | 16 | 128 | 20.2% | 28.7% | 28.51907785821149 |

### 2. Batch Size 对 Decode Linear 占比的影响

| 模型 | Batch | 平均 Decode Linear% | 平均 Prefill Linear% |
|---|---:|---:|---:|
| 2b | 1 | 23.0% | 24.0% |
| 2b | 4 | 26.8% | 14.8% |
| 2b | 16 | 28.3% | 23.5% |
| 4b | 1 | 24.6% | 8.5% |
| 4b | 4 | 20.7% | 15.4% |
| 4b | 16 | 21.4% | 25.4% |
| 9b | 1 | 31.4% | 11.6% |
| 9b | 4 | 22.5% | 22.4% |
| 9b | 16 | 21.9% | 34.3% |

### 3. 输入长度对 Decode Linear 占比的影响

| 模型 | 输入 Token | 平均 Decode Linear% | 平均 Prefill Linear% |
|---|---:|---:|---:|
| 2b | 4 | 25.3% | 11.4% |
| 2b | 16 | 25.4% | 12.3% |
| 2b | 64 | 26.5% | 15.3% |
| 2b | 256 | 26.9% | 44.0% |
| 4b | 4 | 26.7% | 11.7% |
| 4b | 16 | 26.7% | 12.6% |
| 4b | 64 | 17.2% | 16.4% |
| 4b | 256 | 18.3% | 24.9% |
| 9b | 4 | 31.1% | 13.7% |
| 9b | 16 | 29.7% | 15.6% |
| 9b | 64 | 20.0% | 24.7% |
| 9b | 256 | 20.1% | 36.9% |

### 4. 模型大小对 Decode Linear 占比的影响

| 模型 | 平均 Decode Linear% | 平均 Prefill Linear% |
|---|---:|---:|
| 2b | 26.1% | 20.8% |
| 4b | 22.2% | 16.4% |
| 9b | 25.2% | 22.8% |

### 5. Prefill-Heavy vs Decode-Heavy 对比

对比本次 decode-heavy 测试（短输入+长输出）与前次 prefill-heavy 测试（长输入+短输出）:

| 场景 | 2B Prefill Lin% | 2B Decode Lin% | 4B Prefill Lin% | 4B Decode Lin% | 9B Prefill Lin% | 9B Decode Lin% |
|---|---:|---:|---:|---:|---:|---:|
| Prefill-Heavy (022) | 28.5% | 27.4% | 33.5% | 29.8% | 48.6% | 30.1% |
| Decode-Heavy (023) | 20.8% | 26.1% | 16.4% | 22.2% | 22.8% | 25.2% |

## 分析与解读

### Decode Linear 占比峰值

Decode 阶段 linear 占比最高为 **34.5%** — 模型=9b, batch=1, 输入=4。

### 核心结论 1: Decode Linear 占比在短 Prefill 场景下仍然有限

即使在极端 decode-heavy 配置（输入仅 4 token，输出 128-512 token），decode 阶段 linear 占比仍然在 **20-30%** 范围。
这进一步证实了前次研究的结论：decode 阶段本质上是 memory-bound，attention KV cache 操作占据约 70% 时间。

### 核心结论 2: 短 Prefill 时 Prefill Linear 占比下降

当输入序列很短（4-256 token）时，prefill 阶段的 linear 占比显著低于长序列场景。
这是因为短序列的 GEMM 中 M 维度很小，计算量不足以让 GPU 达到 compute-bound 状态。

| 模型 | Prefill-Heavy Prefill Lin% | Decode-Heavy Prefill Lin% | 下降 |
|---|---:|---:|---:|
| 2b | 28.5% | 20.8% | 7.8pp |
| 4b | 33.5% | 16.4% | 17.1pp |
| 9b | 48.6% | 22.8% | 25.8pp |

### 核心结论 3: 纯 Decode 场景下 Linear 压缩的收益上限

对于 decode 为主的推理场景（如 chatbot、代码生成），即使完美压缩所有 linear 层使其耗时归零，理论加速上限也仅约 **1.25-1.43x**（基于 20-30% linear 占比的 Amdahl 定律）。
这意味着在 decode 优化中，attention 优化（KV cache、FlashAttention 等）和系统优化（continuous batching、speculative decoding）是更大的杠杆。

### 对压缩策略的启示

- **Decode 为主场景**: Linear 压缩收益有限（~20-30%），应优先优化 attention 和系统调度
- **Prefill 为主场景**: Linear 压缩收益高（可达 62%），应重点投入
- **混合场景**: 大模型应差异化 prefill/decode 策略；小模型差异不大，可统一处理
