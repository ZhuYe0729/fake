# FakeVLM Uniform Accuracy Summary

| method | status | global_accuracy | total_right | total_wrong | replacement_backend | replaced_linear_count | activation_quant | calibration_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_bf16 | ok | 0.986400 | 4932 | 68 | torch_bf16 | 0 | none | False |
| sparse_bf16 | ok | 0.985200 | 4926 | 74 | cutlass_sparse_bf16_cusparselt | 224 | none | True |
| dense_nvfp4 | ok | 0.987000 | 4935 | 65 | cutlass_nvfp4_sm120 | 224 | dynamic_tensor_global_scale_online | False |
| sparse_nvfp4 | ok | 0.768600 | 3843 | 1157 | cutlass_sparse_nvfp4_sm120 | 224 | dynamic_tensor_global_scale_online | True |
| marlin_weight_only | ok | 0.987600 | 4938 | 62 | marlin_nvfp4_sm120 | 224 | none_bf16_activation | False |
| dense_nvfp4_prefill_marlin_decode | ok | 0.986800 | 4934 | 66 | dense_nvfp4_prefill_marlin_decode | 224 | dynamic_online_for_prefill_bf16_for_decode | False |
