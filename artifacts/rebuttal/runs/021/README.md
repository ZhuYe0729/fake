# Run021：五档质量预算（R1 AK6a / R5 gkP9）

规则和代码在017/018打开confirmation前冻结。沿用同一候选池、validation32与校准速度，扫描ε=.005/.01/.02/.04/.07；按mean+2SE+.005过滤（BF16身份界为0），再取校准E2E最快点。

不为每档新增训练或MILP提案。最终读取各source的64窗口confirmation，这是同一批样本上的多个预算点，不是新增20组独立测试。.01/.04的policy必须与原主表一致。ε单位为nats/token，校准速度不能替代正式测速。

[预登记规则](configs/grid_protocol.json) · [冻结代码SHA](configs/grid.json) · [执行代码](code/budget_grid.py)

程序等待原模型校准和审计；完成后结果位于results/budget_points.csv。
