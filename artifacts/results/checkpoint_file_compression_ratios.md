# Checkpoint File Compression Ratios

This report computes real file-size ratios as:

`dense source model bytes / checkpoint model.pt bytes`

Important: fake-quant/pruned checkpoints in this project usually store dense tensors in `state_dict`, so their real file-size ratio is close to `1x`. Only packed/runtime/storage checkpoints such as `cutlass_*` represent real storage-compressed checkpoint files.

## MaxViT Tiny

| Checkpoint | Type | Dense GiB | Checkpoint GiB | File Ratio | Note |
|---|---|---:|---:|---:|---|
| `nvfp4_4over6_semi_structured_sparse` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed1` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed2` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed3` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed1` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed2` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed3` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_semi_structured_sparse_seed1` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed2` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed3` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed1` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed2` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed3` | fake_dense_state_dict | 0.115 | 0.116 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `cutlass_nvfp4_runtime` | real_packed_checkpoint | 0.115 | 0.029 | 3.954x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_bf16_runtime` | real_packed_checkpoint | 0.115 | 0.040 | 2.857x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_nvfp4_storage` | real_packed_checkpoint | 0.115 | 0.026 | 4.423x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |

## MaxViT Small

| Checkpoint | Type | Dense GiB | Checkpoint GiB | File Ratio | Note |
|---|---|---:|---:|---:|---|
| `nvfp4` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_4over6_semi_structured_sparse` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed1` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed2` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed3` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed1` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed2` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed3` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_semi_structured_sparse` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed1` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed2` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed3` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed1` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed2` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed3` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `semi_structured_sparse` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `unstructured_sparse` | fake_dense_state_dict | 0.257 | 0.257 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `cutlass_nvfp4_runtime` | real_packed_checkpoint | 0.257 | 0.065 | 3.982x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_bf16_runtime` | real_packed_checkpoint | 0.257 | 0.090 | 2.864x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_nvfp4_storage` | real_packed_checkpoint | 0.257 | 0.058 | 4.462x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |

## MaxViT Base

| Checkpoint | Type | Dense GiB | Checkpoint GiB | File Ratio | Note |
|---|---|---:|---:|---:|---|
| `nvfp4` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_4over6_semi_structured_sparse` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed1` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed2` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed3` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed1` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed2` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed3` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_semi_structured_sparse` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed1` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed2` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed3` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed1` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed2` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed3` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `semi_structured_sparse` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `unstructured_sparse` | fake_dense_state_dict | 0.446 | 0.446 | 0.999x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `cutlass_nvfp4_runtime` | real_packed_checkpoint | 0.446 | 0.112 | 3.988x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_bf16_runtime` | real_packed_checkpoint | 0.446 | 0.155 | 2.867x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_nvfp4_storage` | real_packed_checkpoint | 0.446 | 0.100 | 4.471x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |

## MaxViT Large

| Checkpoint | Type | Dense GiB | Checkpoint GiB | File Ratio | Note |
|---|---|---:|---:|---:|---|
| `nvfp4` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_4over6_semi_structured_sparse` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed1` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed2` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_semi_structured_sparse_seed3` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed1` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed2` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse_seed3` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_semi_structured_sparse` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed1` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed2` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_semi_structured_sparse_seed3` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed1` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed2` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse_seed3` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `semi_structured_sparse` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `unstructured_sparse` | fake_dense_state_dict | 0.792 | 0.792 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `cutlass_nvfp4_runtime` | real_packed_checkpoint | 0.792 | 0.197 | 4.013x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_bf16_runtime` | real_packed_checkpoint | 0.792 | 0.275 | 2.878x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_nvfp4_storage` | real_packed_checkpoint | 0.792 | 0.176 | 4.506x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |

## DINOv3 ViT-7B/16

| Checkpoint | Type | Dense GiB | Checkpoint GiB | File Ratio | Note |
|---|---|---:|---:|---:|---|
| `nvfp4` | fake_dense_state_dict | 25.050 | 25.050 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_4over6_semi_structured_sparse` | fake_dense_state_dict | 25.050 | 25.050 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_4over6_unstructured_sparse` | fake_dense_state_dict | 25.050 | 25.050 | 1.000x | Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint. |
| `nvfp4_semi_structured_sparse` | fake_dense_state_dict | 25.050 | 25.050 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `nvfp4_unstructured_sparse` | fake_dense_state_dict | 25.050 | 25.050 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `semi_structured_sparse` | fake_dense_state_dict | 25.050 | 25.050 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `unstructured_sparse` | fake_dense_state_dict | 25.050 | 25.050 | 1.000x | Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression. |
| `cutlass_nvfp4_runtime` | real_packed_checkpoint | 25.050 | 3.541 | 7.075x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_bf16_runtime` | real_packed_checkpoint | 25.050 | 7.056 | 3.550x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_nvfp4_runtime` | real_packed_checkpoint | 25.050 | 3.736 | 6.704x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |
| `cutlass_sparse_nvfp4_storage` | real_packed_checkpoint | 25.050 | 2.564 | 9.768x | Real packed/runtime/storage checkpoint; file-size ratio is meaningful. |

