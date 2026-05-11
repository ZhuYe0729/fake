# 压缩实验结果总结

更新时间：2026-05-11

## 结果文件

- MaxViT dense 基线：`artifacts/results/maxvit_<variant>_dense/accuracy.csv`，`artifacts/results/maxvit_<variant>_dense/speed.csv`
- MaxViT 压缩结果：`artifacts/results/maxvit_<variant>_compressed/accuracy.csv`，`artifacts/results/maxvit_<variant>_compressed/speed.csv`
- DINOv3 dense 基线：`artifacts/results/dinov3_vit7b16_dense/accuracy.csv`，`artifacts/results/dinov3_vit7b16_dense/speed.csv`
- DINOv3 压缩结果：`artifacts/results/dinov3_vit7b16_compressed/accuracy.csv`，`artifacts/results/dinov3_vit7b16_compressed/speed.csv`

注意：早期 `maxvit_dense/*.csv` 中曾混入压缩结果，`maxvit_dense/` 和 `maxvit_compressed/` 现作为历史目录保留。新增 MaxViT 实验应优先使用 `maxvit_tiny_*`、`maxvit_small_*`、`maxvit_base_*`、`maxvit_large_*` 目录。

## 精度结果

### MaxViT

Dense baseline：Top-1 83.44%，Top-5 96.606%。

| method | Top-1 | Top-1 delta | Top-5 |
| --- | ---: | ---: | ---: |
| nvfp4 | 83.238% | -0.202 pp | 96.514% |
| unstructured_sparse | 74.876% | -8.564 pp | 93.868% |
| semi_structured_sparse | 63.832% | -19.608 pp | 87.490% |
| nvfp4_unstructured_sparse | 74.096% | -9.344 pp | 93.472% |
| nvfp4_semi_structured_sparse | 33.564% | -49.876 pp | 59.518% |

MaxViT 对结构化/半结构化剪枝非常敏感，尤其 `nvfp4_semi_structured_sparse` 下降很大。

### DINOv3 ViT-7B/16

Dense baseline：Top-1 88.048%，Top-5 98.404%。

| method | Top-1 | Top-1 delta | Top-5 |
| --- | ---: | ---: | ---: |
| nvfp4 | 88.116% | +0.068 pp | 98.412% |
| unstructured_sparse | 87.614% | -0.434 pp | 98.422% |
| semi_structured_sparse | 87.234% | -0.814 pp | 98.420% |
| nvfp4_unstructured_sparse | 87.638% | -0.410 pp | 98.418% |
| nvfp4_semi_structured_sparse | 84.868% | -3.180 pp | 97.854% |

DINOv3 的下降确实很小，但从当前代码和 metadata 看，不像是压缩没有生效。更可能的解释是：7B 模型冗余很大，线性 probe 分类任务对这类权重扰动比较鲁棒；同时剪枝分数使用了 `weight^2 * activation_hessian_diag`，不是随机剪枝。

## DINOv3 实现检查

当前 DINOv3 压缩范围：

- 选中 280 个 backbone 线性层。
- 对应 `40` 层 transformer，每层 `7` 个 projection：
  - `attention.k_proj`
  - `attention.v_proj`
  - `attention.q_proj`
  - `attention.o_proj`
  - `mlp.gate_proj`
  - `mlp.up_proj`
  - `mlp.down_proj`
- 没有压缩 patch embedding、LayerNorm、分类 head 等小模块。

metadata 检查结果：

| method | selected modules | selected weight params | target sparsity in selected weights | skipped |
| --- | ---: | ---: | ---: | ---: |
| nvfp4 | 280 | 6,710,886,400 | 0% | 0 |
| unstructured_sparse | 280 | 6,710,886,400 | 50% | 0 |
| semi_structured_sparse | 280 | 6,710,886,400 | 50% | 0 |
| nvfp4_unstructured_sparse | 280 | 6,710,886,400 | 50% | 0 |
| nvfp4_semi_structured_sparse | 280 | 6,710,886,400 | 50% | 0 |

没有发现明显的 no-op 问题：

- `unstructured_sparse` 和 `semi_structured_sparse` 都会把剪枝后的权重 copy 回模型。
- `semi_structured_sparse` 是每 4 个输入列保留 2 个，标准 2:4 形式。
- `nvfp4_semi_structured_sparse` 使用每 8 个元素按 pair 剪枝的变体，精度下降明显更大，也侧面说明压缩确实在改变权重。
- `nvfp4` 当前是 fake quantize：权重先量化到 FP4 码本，再反量化回 float tensor 保存。

## 压缩率估算

### 重要说明

当前保存的 `model.pt` 仍是普通 dense float `state_dict`。也就是说，当前 checkpoint 文件本身并没有真实 packed FP4 或 sparse 存储压缩；下面的压缩率是按理想运行时/存储格式估算。

### DINOv3

- backbone safetensors metadata：6,716,035,072 参数。
- classifier 总参数估算：6,724,228,072 参数。
- 被压缩目标权重：6,710,886,400 参数。
- 覆盖率：约 99.80%。
- 未压缩参数：约 13,341,672 参数。

因此：

- 50% sparse 方法在全模型层面的有效零比例约为 `0.5 * 99.80% = 49.90%`。
- 若只考虑理想“保留非零 FP32 权重、不计索引/metadata”，稀疏方法理论上接近 2.0x 参数存储压缩。
- `nvfp4` 若按 4-bit weight + FP16 scale 估算：
  - group size 16：目标权重约 5 bit/weight，整模型约 6.3x 存储压缩。
  - group size 32：目标权重约 4.5 bit/weight，整模型约 7.0x 存储压缩。
- `nvfp4 + sparse` 的真实压缩率取决于是否 packed、是否保存 mask/index、scale 如何按 sparse group 组织；当前代码还没有真实 packed 格式，因此不应报告实际文件压缩率。

### MaxViT

MaxViT 目标压缩权重为 27,885,568 个参数，目标模块内 sparse 方法同样为 50% 零。MaxViT 未压缩模块占比需要进一步统计完整模型参数后才能给出全模型压缩率；从结果看，它对半结构化剪枝比 DINOv3 敏感得多。

## 速度结果解释

当前 speed benchmark 只能说明“压缩后的 dense checkpoint 能正常 forward”，不能说明真实压缩加速。

原因：

- checkpoint 加载只是 `load_state_dict`。
- pruning 只是把权重置零后仍以 dense tensor 参与普通 PyTorch `Linear` / `Conv2d` 计算。
- nvfp4 是 fake quantize，保存的是反量化后的 float 权重。
- 没有替换成真实 sparse kernel、2:4 kernel 或 NVFP4 packed kernel。

因此当前 speed 结果不应作为压缩加速证据。若论文或报告中需要速度收益，需要实现真实运行时算子或 packed 权重路径。

当前 speed 数值可作为 smoke test：

| model | method | mean latency ms | images/sec |
| --- | --- | ---: | ---: |
| MaxViT | dense | 89.662 | 1427.577 |
| MaxViT | nvfp4 | 89.732 | 1426.473 |
| MaxViT | unstructured_sparse | 89.498 | 1430.201 |
| MaxViT | semi_structured_sparse | 89.801 | 1425.376 |
| MaxViT | nvfp4_unstructured_sparse | 90.375 | 1416.313 |
| MaxViT | nvfp4_semi_structured_sparse | 89.927 | 1423.384 |
| DINOv3 | dense | 108.282 | 9.235 |
| DINOv3 | nvfp4 | 106.266 | 9.410 |
| DINOv3 | unstructured_sparse | 105.165 | 9.509 |
| DINOv3 | semi_structured_sparse | 100.990 | 9.902 |
| DINOv3 | nvfp4_unstructured_sparse | 101.520 | 9.850 |
| DINOv3 | nvfp4_semi_structured_sparse | 102.813 | 9.726 |

这些差异主要反映普通 dense forward 的测量波动和权重数值变化，不代表压缩推理加速。

## 结论

- DINOv3 精度下降小是可能合理的；当前检查没有发现压缩未生效的明显实现错误。
- DINOv3 压缩覆盖率很高，约 99.80% 参数位于目标压缩线性层中。
- 当前 sparse 方法在目标权重内确实达到 50% 稀疏，全模型有效零比例约 49.90%。
- 当前 checkpoint 不是真实 packed/sparse 存储格式；速度结果只能作为 forward smoke test。
- 若后续要报告真实压缩率和速度收益，需要实现 packed FP4/sparse checkpoint 与对应推理 kernel。
