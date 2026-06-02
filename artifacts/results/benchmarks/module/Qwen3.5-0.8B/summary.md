# Qwen3.5-0.8B Benchmark Summary

> **Model**: Qwen3.5-0.8B, dtype=bfloat16, attn=sdpa, RTX 5090.  
> **Architecture**: 24-layer decoder with hybrid attention — 18 GatedDeltaNet (linear) + 6 full-attention layers (every 4th).  
> **Test matrix**: batch_size ∈ {1,2,4,8,32,64} × input_tokens ∈ {128,512,4096,8192,16384} × output_tokens ∈ {32,512}

## Phase Status

| Phase | Description | Status | Rows |
|-------|-------------|--------|------|
| Phase 1 | Speed (no hooks, KV cache) | ✅ Complete | 60 (42 OK, 18 OOM) |
| Phase 2 | Coarse breakdown | ⚠️ Partial | 27 OK |
| Phase 3 | Fine breakdown | ❌ Not run | — |

---

## Phase 1: Speed Benchmark

### Prefill Latency (ms)

| bs | 128 tok | 512 tok | 4096 tok | 8192 tok | 16384 tok |
|----|---------|---------|----------|----------|-----------|
| 1  | 88.4 | 107.6 | 286.1 | 501.1 | 983.3 |
| 2  | 89.4 | 109.5 | 295.4 | 587.8 | 1410.3 |
| 4  | 89.2 | 111.4 | 446.0 | 997.2 | **OOM** |
| 8  | 91.3 | 119.3 | 882.1 | **OOM** | **OOM** |
| 32 | 106.2 | 406.2 | **OOM** | **OOM** | **OOM** |
| 64 | 201.1 | 865.8 | **OOM** | **OOM** | **OOM** |

**Key**: Prefill scales roughly O(n^0.6), not O(n). The 128× increase in tokens (128→16384) yields only 11× increase in latency. This confirms the GatedDeltaNet uses a parallel scan algorithm, not sequential recurrence.

### Decode per Token (ms)

| bs | 128 tok | 512 tok | 4096 tok | 8192 tok | 16384 tok |
|----|---------|---------|----------|----------|-----------|
| 1  | 19.81 | 19.91 | 19.82 | 20.12 | 19.98 |
| 2  | 20.71 | 20.71 | 20.67 | 21.11 | 20.67 |
| 4  | 20.71 | 20.69 | 20.71 | 20.66 | — |
| 8  | 20.71 | 21.05 | 20.77 | — | — |
| 32 | 20.95 | 20.85 | — | — | — |
| 64 | 20.80 | 20.70 | — | — | — |

**Key**: Decode latency is rock-stable at ~20ms per token, completely independent of input context length. This validates correct KV cache behavior — decode processes exactly 1 token per step regardless of how many came before.

### Throughput (tok/s, input=128)

| bs | tok/s | Efficiency |
|----|-------|------------|
| 1  | 50 | 100% |
| 2  | 97 | 96% |
| 4  | 193 | 96% |
| 8  | 386 | 95% |
| 32 | 1527 | 94% |
| 64 | 3077 | 95% |

Near-linear batch scaling from bs=1 to bs=64, maintaining ≥94% efficiency throughout.

### OOM Boundaries

OOM driven by full-attention layers (every 4th layer) whose O(n²) memory dominates:

| Max Input | Batch Sizes that work |
|-----------|----------------------|
| 16384 | bs=1,2 |
| 8192 | bs=1,2,4 |
| 4096 | bs=1,2,4,8 |
| 512 | all (1,2,4,8,32,64) |
| 128 | all |

---

## Phase 2: Coarse Breakdown (Partial)

Coverage: bs=1 (all input lengths), bs=2 (all), bs=4 (128 to 8192). Missing: bs≥8 results.

### Prefill: Component vs Input Length (bs=1)

| Component | in=128 | in=512 | in=4096 | in=8192 | in=16384 | Trend |
|-----------|--------|--------|---------|---------|----------|-------|
| hybrid_linear_attn_block | 77–83% | 78–80% | 86–87% | 87% | 85% | ↑ dominates at long context |
| full_attn_block | 4–7% | 4–5% | 3% | 3% | 5% | ↓ then stable |
| mlp_block | 5% | 6% | 4.5% | 5% | 5% | flat |
| norm | 4–5% | 3.5–4% | 1.5–2% | 1% | 1.2% | ↓ proportionally |
| lm_head | 0.3% | 0.6–0.9% | 2.4–2.7% | 3% | 3.4% | ↑ (vocab proj for longer seq) |
| all_linear | 6.6% | 9.0% | 10.3% | 10.8–11.3% | 11.3–11.5% | ↑ |
| other | 3–8% | 4–7% | 1.5–3% | 0.6–1.2% | 0.4–0.8% | ↓ |

**Key insight**: At short context (128 tok), hybrid linear attention takes ~80% of prefill. At long context (16384 tok), this grows to ~85% — the SSM scan is the **dominant bottleneck** for long-context prefill. Full attention (6 layers) accounts for only 3–5%.

### Prefill: Component vs Batch Size (input=128)

| Component | bs=1 | bs=2 | bs=4 |
|-----------|------|------|------|
| hybrid_linear_attn_block | 83% | 64–66% | 57% |
| full_attn_block | 4–7% | 5–6% | 7% |
| mlp_block | 5% | 12–13% | 15–17% |
| all_linear | 7% | 15–16% | 20–21% |

**Key insight**: Larger batch sizes reduce the SSM proportion and increase MLP/Linear proportions. GEMM-based operations (MLP, Linear projections) benefit more from batch parallelism than the SSM scan. At bs=4, all_linear reaches 21% (vs 7% at bs=1).

### Decode: Component Distribution (stable across configs)

| Component | Range | Notes |
|-----------|-------|-------|
| hybrid_linear_attn_block | 37–53% | Higher at short context, decreases with context length |
| full_attn_block | 10–15% | Stable |
| mlp_block | 12–27% | Increases with context length (larger KV cache states) |
| norm | 10–16% | Stable |
| all_linear | 18–32% | Increases with context length |
| other | 8–15% | Stable |

**Key insight**: Decode breakdown is much more balanced than prefill. The SSM scan (hybrid_linear_attn_block) drops from 83% (prefill) to ~40% (decode) at bs=1, because with KV cache only 1 token is processed. All components become more evenly distributed. As input context grows, mlp_block and all_linear proportions increase (24% and 28% at 16384 tok) because larger KV cache states mean more work in projections.

### Prefill vs Decode (bs=1, in=128)

| Component | Prefill | Decode |
|-----------|---------|--------|
| hybrid_linear_attn_block | 83.3% | 48.6% |
| full_attn_block | 4.0% | 12.9% |
| mlp_block | 4.7% | 12.3% |
| norm | 4.7% | 16.1% |
| lm_head | 0.3% | 0.9% |
| all_linear | 6.6% | 18.2% |
| other | 3.0% | 9.3% |

The SSM dominant in prefill (83%) drops to <50% in decode. Norm and mlp_block increase proportionally since decode processes only 1 token, making fixed per-layer overheads (norm) relatively larger.

---

## Metrics Glossary

### Phase 1: Speed Metrics (no hooks, real latency)

所有 Phase 1 指标都通过 `torch.cuda.Event` 在**无 hook** 条件下测量，反映真实 wall-clock latency。

| Metric | Unit | Measurement Method | Meaning |
|--------|------|-------------------|---------|
| `prefill_ms` | ms | CUDA event 包裹完整 prefill forward（`model(input_ids=full_input, use_cache=True)`） | 一次性处理全部 input tokens 并生成 KV cache 的耗时。包含 embed_tokens → 24 层 decoder → norm → lm_head 的完整 forward。 |
| `first_decode_ms` | ms | 同上，但只测第一个 decode step | 第一个 decode token 的耗时。可能略高于后续 token，因为包含一些 lazy init 开销。 |
| `decode_per_token_ms` | ms | 后续 decode steps 的**平均**时间（排除 first_decode） | Steady-state 单 token decode 耗时。KV cache 已就绪，每次只输入 1 个 token，输出 1 个 logits 向量。**这是衡量生成速度的核心指标。** |
| `decode_total_ms` | ms | 计算值：`decode_per_token_ms × (output_tokens - 1)` | 除第一个 token 外的总 decode 时间（估算值）。 |
| `tokens_per_sec` | tok/s | `batch_size × 1000 / decode_per_token_ms` | 吞吐量。batch>1 时表示所有 batch 序列的总生成速度。 |

**测量流程**（一个 iteration）：
```
Prefill: model(full_input, use_cache=True) → logits + past_key_values
  └─ CUDA event 包裹，记录 prefill_ms

Decode loop (output_tokens 步):
  step_0: model(next_token, past_key_values, use_cache=True) → first_decode_ms
  step_1..N: 同上，平均得到 decode_per_token_ms
```

### Phase 2: Coarse Breakdown Metrics (with hooks, percentage only)

所有 Phase 2 指标都是**百分比**（`_pct`），通过 forward hooks 在 prefill/decode 阶段分别收集。由于 hook 本身有 overhead（~6-9× 的绝对时间膨胀），**百分比才有意义，绝对 ms 值不可用于性能评估。**

以下为 coarse mode 的模块分类：

| Metric | Hook 位置 | 包含的子组件 | 说明 |
|--------|----------|-------------|------|
| `hybrid_linear_attn_block` | `layer.linear_attn`（整个 Qwen3_5GatedDeltaNet） | in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, act(SiLU), norm(RMSNormGated), SSM recurrence core, out_proj | **GatedDeltaNet 线性注意力的完整时间。** 18 层中每层各有一个。这是 prefill 的绝对瓶颈。 |
| `full_attn_block` | `layer.self_attn`（整个 Qwen3_5Attention） | q_proj, k_proj, v_proj, q_norm, k_norm, SDPA/FlashAttention kernel, o_proj | **标准 self-attention 的完整时间。** 6 层中每层各有一个（每 4 层出现一次）。包含 QKV 投影 + attention core + output 投影。 |
| `mlp_block` | `layer.mlp`（整个 Qwen3_5MLP） | gate_proj, up_proj, SiLU, down_proj | **FFN/MLP 的完整时间。** 所有 24 层各有一个。SwiGLU 结构：gate/up 投影 → SiLU → down 投影。 |
| `norm` | `layer.input_layernorm` + `layer.post_attention_layernorm` + `language_model.norm` | — | **所有 RMSNorm 的总时间。** 每层 2 个（pre-attn + post-attn）+ 最终 1 个 = 49 个 RMSNorm。 |
| `lm_head` | `model.lm_head` | — | **输出投影的时间。** 将 hidden_size 映射到 vocab_size (248320) 的 Linear 层，仅在 prefill 最后和每个 decode step 执行一次。 |
| `vision` | `model.visual.blocks[]` | — | **视觉编码器时间**（仅多模态模式）。当前测试未使用。 |
| `all_linear` | 全模型所有 `nn.Linear` | — | **所有 Linear 层的总时间**（跨所有 block 类型）。该值与上述 block 指标**有重叠**（Linear 已被包含在各 block 的 inclusive 时间里），不应与其他指标相加。它回答的问题是："GEMM 占了多少时间？" |
| `other` | `100% - sum(tracked)` | — | **未归类的时间。** 包含：hook overhead, Python dispatch overhead, RoPE 位置编码, KV cache update, attention mask 生成, causal mask 处理, embedding lookup, 各 block 中未单独 hook 的小 op 等。`other` 偏高可能表明 hook overhead 显著或存在未覆盖的计算路径。 |

**Coarse mode 中各指标的关系**：
```
total_prefill_time = embed_tokens
  + Σ_layer ( input_layernorm           → "norm"
             + [linear_attn | self_attn] → "hybrid_linear_attn_block" | "full_attn_block"
             + post_attention_layernorm → "norm"
             + mlp                       → "mlp_block" )
  + final_norm                          → "norm"
  + lm_head                             → "lm_head"
  + RoPE, mask, etc.                    → "other"

all_linear = Σ(所有 nn.Linear)   // 与上述指标有重叠，独立展示
```

### Phase 3: Fine Breakdown Metrics (planned, not run)

Fine mode 将 coarse mode 的 block 进一步拆分为子组件：

| Metric | 包含关系 | 说明 |
|--------|---------|------|
| `q_proj / k_proj / v_proj / o_proj` | ∈ `full_attn_block` | 标准 attention 中的 4 个 Linear 投影 |
| `in_proj_qkv / in_proj_z / in_proj_a / in_proj_b` | ∈ `hybrid_linear_attn_block` | GatedDeltaNet 的输入投影 |
| `out_proj` | ∈ `hybrid_linear_attn_block` | GatedDeltaNet 的输出投影 |
| `gate_proj / up_proj / down_proj` | ∈ `mlp_block` | SwiGLU MLP 的 3 个 Linear |
| `attn_core` | ∈ `full_attn_block` | **纯 attention 计算** = self_attn 总时间 − (q+k+v+o projection)。即 fused SDPA/FlashAttention kernel 的 QK^T + softmax + PV 部分。 |
| `hybrid_attn_core` | ∈ `hybrid_linear_attn_block` | **纯 SSM 计算** = linear_attn 总时间 − (in_proj_* + out_proj + conv1d + act + norm)。即 GatedDeltaNet 的状态空间递推核心。 |
| `hybrid_conv1d` | ∈ `hybrid_linear_attn_block` | GatedDeltaNet 中的 1D 时序卷积 |
| `activation` | ∈ `mlp_block` ∪ `hybrid_linear_attn_block` | SiLU 等激活函数 |

**Fine mode 中各指标的关系**：
```
full_attn_block = q_proj + k_proj + v_proj + q_norm + k_norm + attn_core + o_proj

hybrid_linear_attn_block = in_proj_qkv + in_proj_z + in_proj_a + in_proj_b
                         + conv1d + act + norm + hybrid_attn_core + out_proj

mlp_block = gate_proj + up_proj + act_fn + down_proj

all_linear = q_proj + k_proj + v_proj + o_proj
           + in_proj_qkv + in_proj_z + in_proj_a + in_proj_b + out_proj
           + gate_proj + up_proj + down_proj
           + lm_head + (vision Linear if multimodal)
```

### Measurement Caveats

1. **Speed ≠ Breakdown**: Phase 1 的绝对时间是无 hook 的真实性能。Phase 2/3 的绝对时间因 hook overhead 膨胀 6-9×，只取百分比进行分析。
2. **`other` 不是单一模块**：它包含多种未追踪的计算（RoPE, mask, embed, hook overhead 等），不应解释为某个具体瓶颈。
3. **CUDA 异步**: 所有时间通过 `torch.cuda.Event.record() + synchronize()` 测量，反映 GPU kernel 完成时间，不含 CPU 调度开销。
4. **Warmup 隔离**: warmup iterations 不计入统计，避免 CUDA JIT 编译和 cold cache 的影响。
5. **Prefill/Decode 分开**: 两阶段使用不同的 `collect_all()` 调用分别收集 hook 事件，互不污染。
6. **KV cache decode 校验**: decode 时每次只输入 1 个 token + `past_key_values`，decode/tok 稳定在 ~20ms 不随 input length 增长，证明不是 repeated prefill。

---

## Visualization Files

| File | Content |
|------|---------|
| `speed_analysis.png` | 4-panel: prefill vs input, decode vs batch, throughput vs batch, OOM heatmap |
| `prefill_scaling.png` | Prefill scaling with O(n) and O(n^0.6) references |
| `decode_scaling.png` | Throughput vs batch size with ideal linear reference |
| `breakdown_analysis.png` | 4-panel: prefill stacked bar, decode stacked bar, prefill trends, decode trends |
| `prefill_vs_decode_bar.png` | Side-by-side bar comparison of prefill vs decode time distribution |
| `visualize.py` | Run `python visualize.py` to regenerate all charts |

---

## Next Steps

- **Phase 2 completion**: Re-run coarse breakdown for bs≥8 (24 more configs)
- **Phase 3**: Fine breakdown — q/k/v/o_proj, gate/up/down_proj, attn_core, hybrid_attn_core, activation
- **Multi-model**: Extend to Qwen3.5-2B, 4B, 9B, 27B
- **Multimodal**: Vision encoder timing with real images
