## 2026-07-13 - Llama2 closure and Llama3.1 transfer playbook
- 开发目的：把已验证的 Llama2 两场景 Pareto 结论、边界和复现入口沉淀为可直接执行的 Llama3.1-8B-Instruct 准备指南。
- 修改内容：创建 plan 097 与分 gate 的执行手册；明确哪些模型可复用、哪些必须重新校准，要求 uniform 作为优化/测量候选，列出 architecture、质量、速度、求解、实测闭环与故障避免清单。
- 影响文件：`dev/plans/097_llama31_8b_pareto_readiness_plan.md`、`dev/097_llama2_closure_and_llama31_pareto_playbook.md`。
- 后续注意：启动 Llama3.1 实验时在新的 debug 根目录执行 Gate A，不能复用 Llama2 的 quality coefficients、E2E calibration 或 policy JSON。

## 2026-07-13 - Llama3.1 prefill-only Gates A/C scaffold and frozen inputs
- 开发目的：执行 plan 097 的第一阶段，为 Llama3.1-8B-Instruct 的 `b=8, input=2048, output=1` 建立独立、可验证的质量建模闭环。
- 修改内容：新建 `artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto/`；Gate A 输出动态架构/动作支持清单（128 fused modules、GQA QKV `6144x4096`、全部 640 个动作可预测）；生成 Llama3 tokenizer 的固定 100 WikiText blocks 和预注册 72-policy 54/18 split；实现局部误差、逐策略 NLL、严格 shard 合并和正值 local+global quality proxy 脚本。
- 影响文件：`038/.../README.md`、`scripts/` 下的 audit、inputs、local-errors、NLL、merge 与 fit 脚本。
- 后续注意：局部误差和 NLL 标签必须全部来自本目录；仅当 holdout 指标达标后，才实现/运行速度校准与 DP。GPU 仅使用 1--4。

## 2026-07-13 - Remove redundant dense NLL work
- 开发目的：缩短 72-policy 校准标签测量时间，同时保持每个策略与同一固定 dense 参考的可比性。
- 修改内容：NLL evaluator 支持读取已测的 `p00` dense reference；shard runner 强制先验证该参考存在，后续策略只进行一次压缩模型前向，而不重复 dense 前向。
- 影响文件：`038/.../scripts/evaluate_wikitext_nll.py`、`run_nll_shard.sh`。
- 后续注意：`p00.csv` 是唯一 dense reference，必须以同一 100 blocks、同一 tokenizer 和相同 eager BF16 设置生成。

## 2026-07-13 - Correct mixed-policy calibration coverage
- 开发目的：修复 Gate C holdout 的异常负相关，并保持严格的训练/holdout 隔离。
- 修改内容：发现原 `p37--p71` 的 modulo-5 公式只产生 5 个重复 mixed policy；改为五档压缩强度下的确定性独立模块 placement。`p00--p36` 的 uniform 与受控消融保持不变；将归档并重测 `p37--p71`，不能把旧重复标签用于质量模型。
- 影响文件：`038/.../scripts/generate_inputs.py`。
- 后续注意：新标签完成后重新运行 merge/fit；只有新的 18 个 holdout 指标达标，才通过 Gate C。

## 2026-07-13 - Gate C passed; Gate D speed calibration scaffolding
- 开发目的：将已经通过冻结 holdout 的 Llama3.1 质量代理与独立的 E2E 速度校准衔接。
- 修改内容：Gate C 修复后得到 holdout MAE 0.0771、RMSE 0.0910、Spearman 0.967；新增 12 点速度设计、Llama3 checkpoint export + 五次 fresh-process runner、以及 strict-heldout 单调校准拟合脚本。
- 影响文件：`038/.../scripts/build_speed_calibration_design.py`、`run_speed_calibration_point.sh`、`fit_speed_calibrator.py`。
- 后续注意：速度 runner 的 generate-only timing 不包含 export/conversion；只有单调校准在 heldout 上优于 raw dense-scale，才用于显示 E2E 轴。

## 2026-07-13 - Gate D passed and Gate E screening solve
- 开发目的：以已验证的两个 surrogate 生成待实测的 Llama3.1 prefill-only Pareto 候选。
- 修改内容：Gate D strict-heldout MAE 从 raw dense-scale 52.04ms 降至单调校准 11.30ms；固化 quality model 参数，新增 DP screening solver，修复 DP 非零误差被量化为零的问题；新增 closure runner，按“导出→5 次 E2E→删除可再生 checkpoint→100-block NLL”顺序避免磁盘再次耗尽。
- 影响文件：`038/.../reports/quality/model.json`、`pareto/`、`scripts/{solve_predicted_pareto,run_measured_closure_point}.py/sh`。
- 后续注意：solver 输出只用于选点；论文图必须用 closure 的实测 speed/NLL 与 uniform references 重新求非支配集。

## 2026-07-13 - Gate F measured closure completed
- 开发目的：把 Llama3.1 solver screening 点转换为可报告的实测 prefill-only Pareto 数据。
- 修改内容：完成 points 3/5/7/9/11/13 的五次 fresh-process E2E 与 100-block NLL；新增 union report，和同 runner 实测的五个 uniform 基线共同做非支配筛选与绘图。
- 影响文件：`038/.../closure/`、`scripts/build_measured_pareto_report.py`、`report/measured_*`。
- 后续注意：该图是 WikiText measured closure；下游 ARC 任务仍是下一阶段 TODO，不能用预测或 calibration speed 替换实测坐标。

## 2026-07-13 - Start ARC downstream closure
- 开发目的：验证 measured NLL Pareto 趋势是否迁移到真实 prefill-only 下游任务。
- 修改内容：新增 Llama3 ARC-Challenge 0-shot answer-likelihood evaluator，采用原地安装压缩权重以避免 8B 双权重 OOM；计划在 dense、dense-NVFP4 和 measured union mixed frontier 上完整评测。
- 影响文件：`038/.../scripts/evaluate_arc_challenge.py`、`run_arc_challenge_point.sh`、`arc_challenge/`。
- 后续注意：ARC 结果必须与 `report/measured_nll_speed_summary.csv` 的同协议实测速配对汇总。

## 2026-07-13 - ARC downstream closure completed
- 开发目的：验证 Llama3.1 prefill-only measured NLL frontier 的下游可用性。
- 修改内容：完成 dense、dense-NVFP4 与 mixed points 3/5/9/11/13 的完整 1,172-example ARC-Challenge 0-shot answer-likelihood；新增与同协议实测速率配对的 ARC union Pareto 汇总和图。
- 影响文件：`038/.../arc_challenge/full/`、`scripts/build_arc_challenge_report.py`、`arc_challenge/report/`。
- 后续注意：只对已经完成 ARC 的 rows 计算该任务图；其它 uniform 方法尚未做 ARC，不应伪装为 task-level 比较点。
