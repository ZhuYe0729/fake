# 一些可能需要补充的实验测试啥的
### 1.约束求解
这块也许可以在appendix中写一个算法伪代码。

### 2. 验证为何 E2E calibration 采用 PAVA，而非固定 non-Linear offset（不需要了）

#### 近期：motivation 小实验（优先）

- 使用现有 E2E calibration policies，计算 raw Linear proxy $x_j$、实测 E2E latency
  $y_j$ 与隐含开销 $C_j=y_j-x_j$。报告 $C_j$ 的范围、标准差或 IQR，并绘制 $C_j$ 对
  $x_j$ 的散点图；若其存在明显幅度和趋势，即可直观说明 $g(x)=x+C$ 的“固定残差”假设
  不成立。
- 在 $x_j$--$y_j$ 散点上同时画固定 offset 曲线
  $\hat y=x+\bar C$ 与 PAVA 拟合曲线。该图只作为方法动机：说明需要允许非仿射、但保序的
  校准；不将同一批数据上的拟合优劣表述为泛化结果。

#### 后续：完整验证（时间允许时）

- 对每条 E2E calibration policy 计算 raw Linear proxy $x_j$、实测 E2E latency
  $y_j$ 以及隐含开销 $C_j=y_j-x_j$。绘制 $(x_j,y_j)$ 散点和 $C_j$ 对 $x_j$
  的残差图，检查 $C_j$ 是否存在随 raw latency 变化的稳定趋势，而非仅是测量噪声。
- 在完全相同的 train/holdout split 上比较四种校准器的 holdout MAE/RMSE/MAPE：raw
  $\hat y=x$、固定 offset $\hat y=x+\bar C_{\mathrm{train}}$、affine
  $\hat y=\alpha x+\beta$ 和单调 PAVA $\hat y=g_{\mathrm{PAVA}}(x)$。
- 只有在 PAVA 对 holdout 有稳定增益时，才主张其必要性；若固定 offset 或 affine 已足够，
  应采用更简单的模型。另按方法比例及 prefill/decode 组成分组诊断 $C_j$：若残差主要
  与组成有关却和 $x_j$ 无关，则一维 PAVA 也有信息瓶颈，后续应考虑
  composition-aware E2E calibrator。


### 3. 补充说明约束优化求解开销和校准开销啥的


### 4. 对比下直接使用精确的local压缩产生的NLL独立影响预测最终NLL以及我们的方法的预测NLL。