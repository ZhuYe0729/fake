# R2：mixed/chunked执行与PD分离的回复证据

已有结果：047真实vLLM V1 chunked-prefill（token budget=1024）逐行阶段路由，9方法、2负载、3轮；本次只重算原始记录，没有新GPU实验。PD两worker/KV交接及错峰到达尚未实测。

## 中文回复草稿

我们同意continuous batching和chunked prefill会使同一次forward包含不同请求的Prefill与Decode token。原始整批阶段切换路径需要针对这种调度扩展，但阶段异构策略本身仍可执行：Linear计算按token行进行，可根据调度器的原始prompt长度、已计算位置和当前token位置划分两组行，分别调用相应阶段权重，再scatter回原顺序；Attention继续使用原请求边界和KV元数据，不改变Transformer结构。单token的prompt chunk仍归为Prefill，不能仅按query长度判定Decode。

我们已实现并测量这一扩展。实际运行中，标准B8/S2048/O64负载每个计时burst包含15次真正同时含两阶段token的forward；输入512/2048交替、输出256的异构负载每个burst包含10次。逐层gather/阶段kernel/scatter与同一行分组的独立参考计算完全一致，BF16 no-op在两负载各64窗上也一致。

| mixed负载 | 每burst真实mixed forward | DASO均衡 E2E×BF16 | DASO速度点 E2E×BF16 |
| --- | ---: | ---: | ---: |
| B8 / S2048 / O64 | 15 | 1.348 | 1.225 |
| B8 / S512、2048交替 / O256 | 10 | 1.013 | 1.210 |

每方法/负载三轮各5次正式burst，完整九方法和实际负载64窗NLL均保留。该实现使用双表示常驻的eager路径，包含额外权重驻留及gather/scatter成本，与主表单arena/Graph路径分开；所有请求在t0提交，是closed-loop burst，不是已完成的生产open-loop连续到达测试。局部路由一致不等于chunked与non-chunked全模型逐位相同，动态激活量化尺度可能随分组改变，故实际mixed NLL另测。

对于PD分离，策略也有自然的架构映射：Prefill worker采用Prefill配置，Decode worker采用Decode配置，按一致的KV dtype/layout和请求位置语义交接KV。vLLM官方已有通过两个实例和KV connector实现PD分离的机制；据此可讨论兼容性，但我们尚未完成该两worker实验，不能将现有单实例速度等同于PD系统端到端收益，KV传输和首个生成token归属仍需验证。

## 建议的最小补充（尚未执行）

1. 优先补真实错峰到达：复用异构512/2048输入、输出256及冻结权重，采用同一预先固定的请求/到达序列；chunk token budget取512/1024，对比BF16、SQ-INT8→W4A16、DASO两点。记录请求加入/离开后逐步mixed计数、TTFT/TPOT、吞吐和峰值显存，并在实际调度下检查NLL；速度只在持续空闲GPU上统一测量。这直接补现有all-at-t0与生产持续到达的差距。

2. PD分离作为次优先补充：同样两worker/GPU资源下，先跑BF16，再跑一个冻结DASO阶段配置；真实交接Prefill产生的KV，对齐最后prompt token/首个生成token、position与cache布局。对比单实例同策略的逐token logprob/NLL并记录KV字节、传输耗时、TTFT/TPOT及每worker内存。它验证集成与语义，不能预先保证吞吐提升。

## 官方资料与本地来源

- [vLLM 0.11.1 chunked prefill](https://docs.vllm.ai/en/v0.11.1/configuration/optimization/#chunked-prefill)：同batch混合Prefill/Decode是实际调度行为。
- [vLLM 0.11.1 PD分离](https://docs.vllm.ai/en/v0.11.1/features/disagg_prefill/)：两个实例经KV connector交接；该版本标注experimental。不将支持该形态解读为所有生产部署都使用PD分离。
- [完整九方法实测](../qwen_updated_mixed_run047/README.md) · [重算数据](all_results.csv) · [逐burst路由](routing.csv) · [来源核验](audit.json)
- [路由实现](../../../runs/047/code/mixed_runtime.py) · [主机/显存解释](../../reviewer1/phase_memory_reply/README.md)
