# 端到端压缩推理结果总结

更新时间：2026-05-16

## 结论概览

- DINOv3 ViT-7B/16 的三条端到端路径已经打通：dense fp32、CUTLASS dense NVFP4、CUTLASS sparse NVFP4 storage。
- DINOv3 sparse NVFP4 现在从 2.6G storage checkpoint 加载，加载时 pack 成真实 CUTLASS sparse runtime buffer，forward 使用真实 sparse NVFP4 kernel。
- MaxViT 四个 variant 的 dense、CUTLASS dense NVFP4 runtime checkpoint、CUTLASS sparse NVFP4 storage checkpoint 都已完成 speed/accuracy 测试。
- MaxViT small/base 的 CUTLASS dense NVFP4 已修复：跳过 first stage 中 `in_features=96` 的 unsupported dense NVFP4 Linear 后，accuracy 恢复正常。
- MaxViT CUTLASS 首版只替换 Linear；MBConv pointwise Conv2d 仍保持 dense，因此 skipped count 中包含 `unsupported_kind:conv2d`。

## Checkpoint 口径

| model | path | format | size |
| --- | --- | --- | ---: |
| DINOv3 dense NVFP4 | `artifacts/checkpoints/dinov3_vit7b16/cutlass_nvfp4_runtime/model.pt` | CUTLASS runtime packed | 3.54 GiB |
| DINOv3 sparse NVFP4 | `artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/model.pt` | CUTLASS sparse storage | 2.56 GiB |
| DINOv3 old fake sparse | `artifacts/checkpoints/dinov3_vit7b16/nvfp4_semi_structured_sparse/model.pt` | dense float fake checkpoint | 25.05 GiB |
| MaxViT tiny dense NVFP4 | `artifacts/checkpoints/maxvit_tiny/cutlass_nvfp4_runtime/model.pt` | CUTLASS runtime packed | 0.029 GiB |
| MaxViT tiny sparse NVFP4 | `artifacts/checkpoints/maxvit_tiny/cutlass_sparse_nvfp4_storage/model.pt` | CUTLASS sparse storage | 0.026 GiB |
| MaxViT small dense NVFP4 | `artifacts/checkpoints/maxvit_small/cutlass_nvfp4_runtime/model.pt` | CUTLASS runtime packed | 0.065 GiB |
| MaxViT small sparse NVFP4 | `artifacts/checkpoints/maxvit_small/cutlass_sparse_nvfp4_storage/model.pt` | CUTLASS sparse storage | 0.058 GiB |
| MaxViT base dense NVFP4 | `artifacts/checkpoints/maxvit_base/cutlass_nvfp4_runtime/model.pt` | CUTLASS runtime packed | 0.112 GiB |
| MaxViT base sparse NVFP4 | `artifacts/checkpoints/maxvit_base/cutlass_sparse_nvfp4_storage/model.pt` | CUTLASS sparse storage | 0.100 GiB |
| MaxViT large dense NVFP4 | `artifacts/checkpoints/maxvit_large/cutlass_nvfp4_runtime/model.pt` | CUTLASS runtime packed | 0.197 GiB |
| MaxViT large sparse NVFP4 | `artifacts/checkpoints/maxvit_large/cutlass_sparse_nvfp4_storage/model.pt` | CUTLASS sparse storage | 0.176 GiB |

## DINOv3 ViT-7B/16

### Accuracy

| method | checkpoint / loader | Top-1 | delta vs dense | Top-5 | replaced | skipped |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense fp32 | original dense | 88.048% | +0.000 pp | 98.404% | - | - |
| CUTLASS dense NVFP4 | real NVFP4 kernel | 88.146% | +0.098 pp | 98.418% | 280 | 0 |
| CUTLASS sparse NVFP4 | `cutlass_storage_packed_v1`, load-time pack | 84.546% | -3.502 pp | 97.744% | 280 | 0 |

### Speed

Batch sweep 使用 `WARMUP=5, ITERS=20`；accuracy throughput 来自完整 ImageNet eval。

| method | batch=1 latency | batch=1 img/s | best batch | best img/s | throughput speedup vs dense best |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense fp32 | 106.144 ms | 9.421 | 128 | 14.967 | 1.00x |
| CUTLASS dense NVFP4 | 38.093 ms | 26.251 | 8 | 81.746 | 5.46x |
| CUTLASS sparse NVFP4 storage | 39.218 ms | 25.499 | 8 | 87.387 | 5.84x |

当前 DINOv3 结论：dense NVFP4 基本无精度损失，并显著快于 fp32；sparse NVFP4 有约 3.5 pp Top-1 损失，但 storage checkpoint 真实压缩且 forward 使用真实 CUTLASS sparse kernel。

## MaxViT

### Accuracy

| variant | method | checkpoint / loader | Top-1 | delta vs dense | Top-5 | replaced | skipped | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| tiny | dense | original dense | 83.440% | +0.000 pp | 96.606% | - | - | OK |
| tiny | CUTLASS dense NVFP4 | `cutlass_runtime_packed_v1` | 83.290% | -0.150 pp | 96.584% | 88 | 22 | OK |
| tiny | CUTLASS sparse NVFP4 | `cutlass_storage_packed_v1` | 70.894% | -12.546 pp | 89.944% | 88 | 22 | OK |
| small | dense | original dense | 84.456% | +0.000 pp | 96.812% | - | - | OK |
| small | CUTLASS dense NVFP4 | `cutlass_runtime_packed_v1` | 84.332% | -0.124 pp | 96.770% | 76 | 34 | OK |
| small | CUTLASS sparse NVFP4 | `cutlass_storage_packed_v1` | 78.302% | -6.154 pp | 94.108% | 76 | 34 | OK |
| base | dense | original dense | 84.128% | +0.000 pp | 96.166% | - | - | OK |
| base | CUTLASS dense NVFP4 | `cutlass_runtime_packed_v1` | 84.826% | +0.698 pp | 96.958% | 180 | 60 | OK |
| base | CUTLASS sparse NVFP4 | `cutlass_storage_packed_v1` | 79.710% | -4.418 pp | 94.784% | 180 | 60 | OK |
| large | dense | original dense | 88.050% | +0.000 pp | 98.522% | - | - | OK |
| large | CUTLASS dense NVFP4 | `cutlass_runtime_packed_v1` | 88.022% | -0.028 pp | 98.514% | 192 | 48 | OK |
| large | CUTLASS sparse NVFP4 | `cutlass_storage_packed_v1` | 79.828% | -8.222 pp | 95.608% | 192 | 48 | OK |

### Speed

MaxViT tiny/small/base 使用 `224x224`，batch size 128；large 使用 `512x512`，batch size 16。

| variant | method | latency | img/s | speedup vs dense | replaced | skipped |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tiny | dense | 89.530 ms | 1429.696 | 1.00x | - | - |
| tiny | CUTLASS dense NVFP4 | 85.483 ms | 1497.377 | 1.05x | 88 | 22 |
| tiny | CUTLASS sparse NVFP4 | 83.310 ms | 1536.430 | 1.07x | 88 | 22 |
| small | dense | 145.967 ms | 876.910 | 1.00x | - | - |
| small | CUTLASS dense NVFP4 | 118.125 ms | 1083.594 | 1.24x | 76 | 34 |
| small | CUTLASS sparse NVFP4 | 115.244 ms | 1110.686 | 1.27x | 76 | 34 |
| base | dense | 256.369 ms | 499.281 | 1.00x | - | - |
| base | CUTLASS dense NVFP4 | 216.522 ms | 591.163 | 1.18x | 180 | 60 |
| base | CUTLASS sparse NVFP4 | 213.651 ms | 599.109 | 1.20x | 180 | 60 |
| large | dense | 262.528 ms | 60.946 | 1.00x | - | - |
| large | CUTLASS dense NVFP4 | 208.341 ms | 76.797 | 1.26x | 192 | 48 |
| large | CUTLASS sparse NVFP4 | 202.670 ms | 78.946 | 1.30x | 192 | 48 |

当前 MaxViT 结论：真实压缩 checkpoint 和真实 CUTLASS kernel speed 链路已经跑通。dense NVFP4 对四个 variant 的 accuracy 都正常，速度提升约 1.05x-1.26x；sparse NVFP4 对四个 variant 都有速度收益，但有明显精度下降。

## 已知限制和后续建议

- MaxViT CUTLASS 当前只替换 Linear，MBConv pointwise Conv2d 未进入 CUTLASS 压缩 kernel；skipped count 主要来自这些 Conv2d，small/base dense/sparse 还额外跳过 first stage 中 `in_features=96` 的 Linear shape。
- MaxViT small/base dense NVFP4 的旧异常来自过度替换 `K=96` Linear；当前结果已使用 `in_features % 64 == 0` guard 重新 prepare checkpoint 并重测。
- DINOv3 sparse storage checkpoint 已经显著小于 runtime sparse checkpoint，也远小于旧 fake dense checkpoint；这是目前最可信的“真实压缩存储 + 真实 kernel 推理”路径。
