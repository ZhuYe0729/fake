### 5.1 两份阶段表示常驻时的内存

**结论：** Qwen Balanced的GPU峰值比BF16低约20%，同时明确付出约8.24 GiB的Host阶段权重缓存。INT8更省GPU内存，DASO提供的是另一种质量–速度–内存折中。

| 方法 | GPU Torch峰值 GiB | Host阶段权重缓存 GiB |
| --- | ---: | ---: |
| BF16 | 21.38 | 0.00 |
| INT8 | 14.99 | 0.00 |
| DASO (Balanced) | 17.10 | 8.24 |
| DASO (Max-speed) | 17.31 | 8.76 |

<details>
<summary>测量说明与来源</summary>

与上一节相同的Qwen混合调度变体、KV6 GiB。GPU峰值是Torch分配峰值，包含KV/workspace等；Host列只是阶段adapter权重缓存，不是总进程RSS。Host/GPU两项不是可以相加的同一内存口径。

[原始内存字段、峰值与协议](responses/reviewer2/qwen_updated_mixed_run047/README.md) · [原主表三模型内存](responses/reviewer1/runtime_memory_night151/README.md)。

</details>

