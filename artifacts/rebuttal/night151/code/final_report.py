from pathlib import Path
import json,csv,shutil,datetime
base=Path('artifacts/rebuttal');night=base/'night151'
read=lambda p:json.loads(p.read_text())
for rid in ['016','017','018','019','020','021','022','023','024','025','026']:assert read(base/f'runs/{rid}/results/final_audit.json')['verified']
preserved=read(night/'final_preservation_audit.json');assert preserved['verified']
old=base/'NIGHT151_REPORT.md';archive=night/'reference/NIGHT151_TIMELINE.md'
if not archive.exists():shutil.copy2(old,archive)
lines=['# 本轮rebuttal实验总报告（五位人工reviewer）','',
 '更新：'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')+'。本轮0016–026内已启动的计算全部结束，各实验按自身规定范围验证通过；AI reviewer未列入任务。旧结果、失败和重复尝试均保留。','',
 '## 可直接查看的主表','',
 '| 模型 | 完整主表 | 交付范围 |',
 '| --- | --- | --- |',
 '| Qwen3-8B | [19行汇总](responses/reviewer2/qwen3_table2_run016/README.md) | 015原17行+016新DASO-8；跨campaign、各自BF16归一化，逐项来源保留 |',
 '| Llama-3.1-8B-Instruct | [15行完整表](responses/reviewer2/llama31_8b_instruct_table2/README.md) | 统一新路径，四MC+三生成+三轮速度 |',
 '| Llama-2-7B-Chat | [15行完整表](responses/reviewer2/llama2_7b_chat_table2/README.md) | 同上；MHA主测速KV10GiB，Llama-3.1为6GiB |','',
 '三个表共539个指标单元已核对有限值和来源。两Llama包含原uniform点、原DASO-5、plain INT8、SQ-INT8、FP8、HAWQ-FP4 PTQ适配、简单phase组合、共同硬件端点及DASO-8。所有速度为GPU0/NUMA0、同模型固定负载与BF16首尾锚点；完整任务允许空闲多卡各一模型并行。','',
 '## 最有用的新证据','',
 '### R2 Exi5 / R3 LrT2：六种有目的的负载','',
 '[完整66行](responses/reviewer2/workloads_run019/README.md) · [汇总图PDF](responses/reviewer2/workloads_run019/qwen019_workloads.pdf) · [SVG](responses/reviewer2/workloads_run019/qwen019_workloads.svg)','',
 '固定中心配置迁移，六个负载全部保留，不按新结果重选。DASO-8 ε=.04的E2E数值最快5/6，含恢复cycle最快4/6；B1/S512/O256保留W4A16更快的反例。六个64窗口NLL mean+2SE均小于.04，但窗口跨负载共用，不称六次独立泛化保证。','',
 '| B / S / O | DASO E2E × | 最快非DASO基线 × | DASO cycle × | DASO ΔNLL ±2SE |',
 '| --- | ---: | ---: | ---: | ---: |']
for row in csv.DictReader((night/'workload_run019_summary.csv').open()):
 lines.append(f'| {row["batch"]}/{row["prompt"]}/{row["output"]} | {float(row["daso_e2e_speedup"]):.3f} | {float(row["best_baseline_speedup"]):.3f} | {float(row["daso_cycle_speedup"]):.3f} | {float(row["daso_delta_nll"]):.5f} ± {float(row["daso_two_se"]):.5f} |')
lines+=['','### R1 AK6a / R5 gkP9：预算与合理的样本量调整','',
 '[两Llama五预算20点](responses/reviewer1/five_budget_llama_night151/README.md)完成全部正式速度，均值0超限；预算共享原确认集、配置存在重复，不当作20个独立测试。原始代理不准的点仍保留，不能把validation后的结果归给原DP天然保证。','',
 '[026独立跟进](responses/reviewer2/validation_size_run026/README.md)：固定14个原主表PD incumbent，先复现32窗口选择，再增加32个不重叠validation窗口，规则仍mean+2SE+.005，之后才打开64个新的确认窗口。Llama-2 ε=.04从1.533×的INT8改选1.789×的共同硬件端点，新确认ΔNLL=.01219±.01254（2SE），未超.04；ε=.01仍BF16。','',
 '这是一项有界incumbent样本量研究：速度/三生成引用018相同有效map的已审计结果，不是全候选重新优化或新增独立下游测试。原主表不替换；新点DialogSum并未同时提升。新增17个评分叶子进程合计507.14秒，含初始化，不是GPU-active或并行总墙钟。','',
 '### R4 QboE：简单剪枝比较','',
 '[完整四方法、全部任务和速度](responses/reviewer4/pruning_llama31_run020/README.md)。相同2:4结构与SparseBF16 kernel；Wanda使用原记录的128块训练数据重建及逐层传播，magnitude无需校准。','',
 '| 方法 | MMLU % | WikiText word PPL | Prefill × | P+D E2E × | 本轮准备s |',
 '| --- | ---: | ---: | ---: | ---: | --- |']
for row in csv.DictReader((base/'runs/020/results/pruning_table.csv').open()):
 method=row['method'];cost='—' if method=='BF16' else '历史状态复用，未恢复可信原计时' if method=='SparseGPT 2:4' else f'{read(base/"runs/020/results"/f"canonical_{method.split()[0].lower()}.json")["total_seconds_including_save"]:.2f}'
 lines.append(f'| {method} | {float(row["mmlu"]):.3f} | {float(row["wikitext_word_ppl"]):.3f} | {float(row["prefill_ttft_speedup"]):.3f} | {float(row["pd_e2e_speedup"]):.3f} | {cost} |')
lines+=['','SparseGPT在MMLU/PPL领先；Wanda在CNN/DM和IWSLT分数更高，全部生成结果保留。相近速度来自共享kernel，不能归因于剪枝算法。原SG校准tensor未保留，重建只核验加载规则、token总数与起点，未声称对不可用原tensor做过字节比较。','',
 '### R3 LrT2 / R5 gkP9：预测、求解、接口与开销','',
 '- [三模型建模表/图](responses/reviewer3/modeling_night151/README.md)：新实测LUT+预先固定BF16残差的六场景E2E MAPE约1.00%–1.71%；Qwen全43场景，Llama核心31/29场景，未测点不混入。原Hybrid广shape留出另列，不能用LUT结果冒充未知shape预测。质量代理MAE覆盖完整冻结池，不读取确认集重新拟合。',
 '- [18行内存/cycle](responses/reviewer1/runtime_memory_night151/README.md)：初始化、跨场景进程Torch峰值、生成结束NVML采样值、GPU Linear storage/arena、host cache和恢复分列。NVML不是连续真实瞬时峰值，arena不是总模型显存，cycle包含恢复。',
 '- [16个DP网格点+8个inverse一致性检查](responses/reviewer5/solver_grid_run023/README.md)：2000-bin原模块phase-pair取整相对精确代理成本损失约1.43%–73.44%；20000 bins改善但未总消除。只证明固定加和代理求解行为，不证明真实质量或硬SLA。',
 '- [768条实际接口](responses/reviewer5/interfaces_run025/README.md)：共同FP4/INT8阶段map、SQ-INT8、FP8在B8/S128/O2实测Linear边界均BF16，shape/stride完整保存。内部量化在Linear画像中计费，阶段权重搬运另报；接口dtype相同不能推导总转换开销为零。',
 '- [完整叶子成本口径](night151/COST_REPORT.md)：profile、Hessian、标签、拟合、validation、导出、正式测量与评测分列，缓存不是从零免费。进程壁钟之和不是GPU-active或并行总时长。',
 '- [024局部特征尝试](runs/024/REPORT.md)：仅恢复新8bit逐模块误差，四组旧holdout仅一组略改善、三组变差，全部保留且未替换方法。','',
 '## 解释边界与尚未覆盖项','',
 '- HAWQ为固定候选上的Hessian/ILP PTQ适配，完整CNN QAT/TVM没有移植；[signed/clip/absolute对照](responses/reviewer2/hessian_variants_llama_night151/README.md)全部保留。Llama-3.1 prefill的clip/absolute NLL更好，不能由对signed的结果宣称击败所有Hessian变体。',
 '- DASO有明确质量–速度折中，部分选择与uniform/共同端点重合，不能把所有收益归因于代理或宣称所有主表指标都最优。NLL预算不保证MC/ROUGE/BLEU；严格预算仍有BF16回退。',
 '- 完整30/32B、PRO6000新增baseline与生产open-loop continuous serving未完成；旧run006是mixed closed-loop burst，不能当生产在线SLA证明。两个Llama同候选中心之外的部分正式速度仍标pending，NLL已有；未铺满全部模型×负载×预算的笛卡尔积。',
 '- 同一运行时内的公平对照、候选扩展、精确求解、validation和运行时修复分别标注。全部539个主表指标可追溯，Qwen19行是跨campaign汇总。','',
 '## 完整性与过程','',
 f'001–020的{sum(r["checked_file_hashes"] for r in preserved["runs"]):,}个归档文件哈希及记录的symlink复核0错误，见[保存审计](night151/final_preservation_audit.json)。各新run均保留原代码、数据、日志、CSV、失败和审计范围。',
 '020并行切换的一份旧Wanda CNN/DM重复计算保留在previous目录，两套1000条答案字节/token/分数完全一致，未按分数选取。019/020的−15控制器重启记录及MC门槛/接口记录器的早期错误都未删除。','',
 '[实验总清单（按reviewer对应）](EXPERIMENT_CHECKLIST.md) · [reviewer证据索引](responses/README.md) · [详细时间线](night151/reference/NIGHT151_TIMELINE.md) · [开发记录](../../dev/impls/151_rebuttal_night_continuation_impl.md)','']
old.write_text('\n'.join(lines).replace('0016–026','016–026'))
print('Wrote final self-contained night report')
