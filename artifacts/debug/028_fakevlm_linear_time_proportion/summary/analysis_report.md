# FakeVLM Linear 时间占比分析报告

## 数据概况

- Workload 数: 6
- Linear 模块数: all=371, language=224, vision=144, projector=2

## 核心结果

| Workload | Batch | 输入 | 输出 | Prefill Linear% | Language% | Vision% | Projector% | Decode Linear% | Decode Language% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_01 | 1 | 16384 | 32 | 30.7% | 29.9% | 0.2% | 0.0% | 15.9% | 15.7% |
| normal_02 | 1 | 16384 | 256 | 30.7% | 29.9% | 0.2% | 0.0% | 15.8% | 15.6% |
| prefill_b16_i1024 | 16 | 1024 | 0 | 73.3% | 69.7% | 2.1% | 0.1% | 0.0% | 0.0% |
| prefill_b1_i1024 | 1 | 1024 | 0 | 74.1% | 65.0% | 7.4% | 0.2% | 0.0% | 0.0% |
| prefill_b4_i1024 | 4 | 1024 | 0 | 79.4% | 74.1% | 3.7% | 0.2% | 0.0% | 0.0% |
| prefill_b4_i4096 | 4 | 4096 | 0 | 58.0% | 56.2% | 0.7% | 0.0% | 0.0% | 0.0% |

## 解读要点

- Prefill 包含 vision tower、multimodal projector 和 language model，因此总 linear 占比应同时看 language/vision/projector 拆分。
- Decode 使用 KV cache 后通常只走 language model 路径，vision/projector linear 占比应接近 0；如果不为 0，需要优先检查模型 forward 是否重复传入图像特征。
- 和纯 LLM 对比时，FakeVLM prefill denominator 更大，因为图像编码和多模态投影也在完整 forward 内。
- 该报告使用 CUDA event hook 计时，适合分析比例趋势；严格 kernel attribution 仍应使用 nsys 交叉验证。