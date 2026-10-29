# 新8bit真实模块接口（R5 gkP9 / R3 LrT2）

已冻结Llama-3.1权重，空闲卡eager B8/S128/O2、1GiB KV，仅检查metadata。共同FP4/INT8阶段map、SQ-INT8和FP8各128 Linear×2阶段，共768条；完整dtype/shape/stride/device在CSV。

| Policy | Phase | Internal weight method | Boundaries | Input dtype | Output dtype |
| --- | --- | --- | ---: | --- | --- |
| hardware_minimum_prefill_decode_phase_pair | prefill | int8 | 96 | BF16 | BF16 |
| hardware_minimum_prefill_decode_phase_pair | prefill | dense_nvfp4 | 32 | BF16 | BF16 |
| hardware_minimum_prefill_decode_phase_pair | decode | int8 | 96 | BF16 | BF16 |
| hardware_minimum_prefill_decode_phase_pair | decode | w4a16_ours | 32 | BF16 | BF16 |
| uniform_sq_int8 | prefill | sq_int8 | 128 | BF16 | BF16 |
| uniform_sq_int8 | decode | sq_int8 | 128 | BF16 | BF16 |
| uniform_fp8 | prefill | fp8 | 128 | BF16 | BF16 |
| uniform_fp8 | decode | fp8 | 128 | BF16 | BF16 |

这些已测接口均以BF16 tensor连接，内部压缩权重格式不等于跨层activation格式。内部激活量化、scale/metadata与GEMM在对应Linear画像中计时；阶段权重搬运及恢复另外报告，不能由dtype相同推断总转换开销为零。

这是短输入的实际接口验证，不是速度、精度或生产mixed调度测试。初次post-hook观察在完整计数断言失败；代码检查显示全局phase在最后一层内部切换，已改pre-hook记录入口阶段，原代码/失败日志保留。模型计算及源run017均未改变。

[768条CSV](results/interfaces.csv) · [审计](results/final_audit.json) · [固定设计](configs/design.json)
