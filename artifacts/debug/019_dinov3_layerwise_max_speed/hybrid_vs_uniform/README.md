# DINOv3 Hybrid Comparison

Compares the existing hybrid run against single uniform methods at the complete overlapping batch size.

The hybrid result improves batch-32 throughput from the best uniform result, Sparse BF16 at 81.607 images/sec, to 86.362 images/sec. That is a 1.058x speedup over the best uniform DINOv3 baseline.

This gain is much smaller than the LLaMA-2 prefill-only hybrid results because DINOv3 has less exploitable layer-to-layer kernel diversity in the measured setup. The DINOv3 transformer blocks repeat the same small set of ViT projection shapes, and at batch 32 a single uniform backend, Sparse BF16, is already close to optimal for most of the runtime. The existing DINOv3 hybrid only switches part of the model to the faster backend, so it mostly captures a small residual improvement.

In the earlier LLaMA-2 prefill-only study, the mixed policies had a much wider useful design space: different projection groups had different best backends and the Pareto-selected policies combined Dense BF16, Dense NVFP4, and Sparse BF16. The recorded LLaMA-2 prefill points reached about 1.49x to 1.64x speedup versus dense while also outperforming the uniform sparse baselines in quality-sensitive comparisons. That larger spread leaves more room for hybrid routing to beat any one uniform method.

| Batch | Method | img/s | latency ms | speedup vs dense | speedup vs best uniform |
|---:|---|---:|---:|---:|---:|
| 32 | Dense FP32 | 14.657 | 2183.231555 | 1.000000 | 0.179605 |
| 32 | Dense NVFP4 | 67.115 | 476.796582 | 4.579041 | 0.822417 |
| 32 | Sparse BF16 | 81.607 | 392.121809 | 5.567783 | 1.000000 |
| 32 | Sparse NVFP4 | 73.866 | 433.215964 | 5.039640 | 0.905143 |
| 32 | Hybrid | 86.362 | 370.534180 | 5.892202 | 1.058267 |
