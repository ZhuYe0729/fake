# 164 — 两Llama补齐统一8bit-prefill/W4-decode对照

## 问题与边界
- 当前Llama-3.1参数扫描最佳生成Avg仍未超过plain INT8；原SQ-prefill/W4-decode已测，尚缺plain INT8和FP8这两种prefill对应的完整生成。这两个全局phase-map早已存在017/018候选池，不是改弱基线。
- 独立038（来源017）与039（来源018）补完整三种8bit-prefill/W4A16-decode比较。新两点是固定阶段控制，不标成DASO自动求解/框架优化结果，完整不利点保留。
- 037同输入基线重放核查通过后运行；若出现不一致，先定位，不直接用不同协议比较。

## 冻结与执行
- 两个新控制：全plain INT8 prefill＋全W4A16 decode；全FP8 prefill＋全W4A16 decode。共同比较BF16、plain INT8、SQ-INT8、FP8、既有SQ-prefill/W4-decode五参考，7个唯一map/模型。
- 两新控制各自重新测试原64窗口NLL、CNN/DM1000/DialogSum1500/IWSLT333完整生成；五参考按已核验同map/数值代码/输入协议复用NLL/生成。无新MC，不声称新Prefill-only优化。
- 全7map统一新三轮P+D速度，B8/S2048/O64、原KV6/10GiB、native decode Graph，3warmup+5timed及首尾BF16/同轮INT8。每模型首轮前固定空闲GPU/NUMA；两模型质量可并行、速度排他。
- 可复用night161的精确量化export硬链接缓存；method-map、源模型、量化代码和activation scales须一致。所有原代码、基线、结果和预算扫描不改。

## 验收
- 固定stage控制128模块映射、数据和配方SHA在自身GPU输出前冻结；真实NLL与完整生成/三轮速度全部有来源。
- 对照是否超过8bit或DASO由实测决定；无预算标签或strictε声明。报告三种统一prefill方案的完整得分，不能只保留较好的一种。
- 发布独立phase-control报告/CSV/范围和主表链接；若发现明显配置空间可用但代理未选中，记录为建模问题和后续依据，不把手工控制冒充原求解器成功。
