# 050 Structural Proxy Scenarios Plan

## 目标

验证并设计能突出 layer depth 与 linear type 影响的特定场景，尽量减轻 local error sum 的影响；优先聚焦 sparse NVFP4。

## 假设

- sparse NVFP4 的 local error 使用 015 kernel-aware 数据。
- 场景设计以同 count、同 raw local error bin 为前提，通过选择 high/low structural proxy 的 module 组合放大 layer/type 差异。
- 先做离线分析和 policy 生成，不立即跑 GPU loss；如果离线设计质量足够，再跑真实 loss。

## 实施步骤

1. 验证 layer/type 系数与 local error 大小的关系  
   验证：输出 layer/type mean local error 与 fitted coefficient 的相关性。

2. 构造 sparse NVFP4 structural scenarios  
   验证：生成 matched pair policies，要求 count 和 raw local sum 接近，但 structural proxy gap 大。

3. 输出诊断报告  
   验证：报告中列出 raw gap、structural gap、composition 差异，以及建议是否值得跑真实 loss。
