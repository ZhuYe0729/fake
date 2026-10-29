"""Report raw DP, exact search, validation selection and actual inverse SLAs."""
import csv
import hashlib
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import RUN,write_json


def main():
    selected=json.loads((RUN/'configs/selected_policies.json').read_text());design=json.loads((RUN/'configs/design.json').read_text())
    assert json.loads((RUN/'results/pipeline.json').read_text())['status']=='complete'
    assert hashlib.sha256((RUN/'results/validation_selection_audit.json').read_bytes()).hexdigest()==selected['selection_audit_sha256']
    assert hashlib.sha256((RUN/'configs/design.json').read_bytes()).hexdigest()==selected['rules_sha256']
    audit=json.loads((RUN/'results/validation_selection_audit.json').read_text());assert not audit['test_results_present_before_selection']
    assert all((RUN/f'results/test_complete/{pid}.json').exists() and (RUN/f'results/test_speed_complete/{pid}.json').exists() for pid in selected['policies'])
    rows=[];rng=np.random.default_rng(144941);bootstrap=rng.integers(0,64,(4000,64));cache={};timing_samples={}
    for point in selected['points']:
        pid=point['policy_id'];scenario=point['scenario'];row={k:v for k,v in point.items() if k not in ['solver','validation']}
        if pid is None:rows.append(row);continue
        key=(pid,scenario)
        if key not in cache:
            measured=json.loads((RUN/f'results/test/{pid}/{scenario}.json').read_text())
            base=json.loads((RUN/f'results/test/uniform_dense_bf16/{scenario}.json').read_text())
            latency=json.loads((RUN/f'results/test_speed/{pid}/{scenario}.json').read_text())
            baseline=json.loads((RUN/f'results/test_speed/uniform_dense_bf16/{scenario}.json').read_text())
            assert measured['quality_score_batch']==8 and len(measured['blocks'])==64
            assert latency['gpu']=='2' and all(x['generated_tokens']==8*(1 if scenario=='prefill_only' else 64) for x in latency['iterations'])
            delta=np.array([a['avg_nll']-b['avg_nll'] for a,b in zip(measured['blocks'],base['blocks'],strict=True)])
            ci=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            cache[key]={'measured_delta_nll':float(delta.mean()),'delta_nll_ci_low':float(ci[0]),'delta_nll_ci_high':float(ci[1]),
                'measured_ms':latency['median_ms'],'bf16_ms':baseline['median_ms'],'speedup':baseline['median_ms']/latency['median_ms'],
                'quality_blocks':64,'quality_score_batch':8,'speed_gpu':latency['gpu']}
            timing_samples[key]=[r['elapsed_ms'] for r in latency['iterations'] if not r['warmup']]
            cache[key]['max_timed_ms']=max(timing_samples[key]);cache[key]['min_timed_ms']=min(timing_samples[key])
        row.update(cache[key])
        if 'quality_budget' in point:row['budget_violation']=max(0.,row['measured_delta_nll']-point['quality_budget'])
        if point['family']=='exact inverse':
            fraction=float(point['label'].rsplit('_r',1)[1]);baseline=json.loads((RUN/f'results/calibration_speed/uniform_dense_bf16/{scenario}.json').read_text())
            row['actual_latency_sla_ms']=baseline['median_ms']*fraction
        if 'actual_latency_sla_ms' in row:
            row['actual_sla_violation_ms']=max(0.,row['measured_ms']-row['actual_latency_sla_ms'])
            row['timed_sla_violations']=sum(v>row['actual_latency_sla_ms'] for v in timing_samples[key])
            row['timed_sla_samples']=len(timing_samples[key])
            row['max_timed_sla_violation_ms']=max(0.,row['max_timed_ms']-row['actual_latency_sla_ms'])
        if 'validation' in point:row.update({'validation_'+k:v for k,v in point['validation'].items() if k not in ['policy_id','scenario']})
        rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (RUN/'results/summary_final.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    groups=defaultdict(list)
    for r in rows:
        if 'quality_budget' in r:groups[r['scenario'],r['family']].append(r)
    quality=[{'scenario':scenario,'family':family,'points':len(values),
        'violations':sum(r['budget_violation']>0 for r in values),'max_violation':max(r['budget_violation'] for r in values)}
        for (scenario,family),values in groups.items()]
    sla=[]
    for family in ['exact inverse','inverse plus validation']:
        for scenario in ['prefill_only','prefill_decode']:
            values=[r for r in rows if r['family']==family and r['scenario']==scenario]
            feasible=[r for r in values if not r.get('infeasible')]
            sla.append({'scenario':scenario,'family':family,'requests':len(values),'selected':len(feasible),
                'infeasible':len(values)-len(feasible),'actual_sla_violations':sum(r['actual_sla_violation_ms']>0 for r in feasible),
                'max_sla_violation_ms':max([r['actual_sla_violation_ms'] for r in feasible] or [0.]),
                'timed_sample_violations':sum(r['timed_sla_violations'] for r in feasible),
                'timed_samples':sum(r['timed_sla_samples'] for r in feasible)})
    write_json(RUN/'results/quality_sla_audit.json',{'quality':quality,'latency_sla':sla})
    pool=json.loads((RUN/'configs/candidate_pool.json').read_text());solver=json.loads((RUN/'results/exact_solver_audit.json').read_text())
    pareto=json.loads((RUN/'results/surrogate_audit.json').read_text())
    assert solver['passed'] and pareto['passed']
    cutoff=(RUN/'configs/selected_policies.json').stat().st_mtime_ns;preparation=[]
    for path in (RUN/'logs').glob('*.command.json'):
        stamp=int(path.name.removesuffix('.command.json').rsplit('_',1)[1])
        if stamp>=cutoff:continue
        command=json.loads(path.read_text());argv=command['command'];stage=None
        if str(argv[1]).endswith('export_model.py'):stage='candidate_export'
        elif '--mode' in argv and argv[argv.index('--mode')+1] in ['calibration_speed','validation']:stage=argv[argv.index('--mode')+1]
        if stage is None:continue
        completion=json.loads(path.with_name(path.name.replace('.command.json','.completion.json')).read_text())
        assert completion['returncode']==0
        preparation.append({'stage':stage,'start_ns':stamp,'seconds':completion['seconds'],'gpu':command['gpu'],'source':str(path.relative_to(RUN))})
    costs=[]
    for stage in ['candidate_export','calibration_speed','validation']:
        a=[r for r in preparation if r['stage']==stage]
        costs.append({'stage':stage,'commands':len(a),'sum_process_seconds':sum(r['seconds'] for r in a),
            'elapsed_wall_envelope_seconds':(max(r['start_ns']/1e9+r['seconds'] for r in a)-min(r['start_ns']/1e9 for r in a)) if a else 0.,
            'gpus':sorted(set(r['gpu'] for r in a))})
    write_json(RUN/'results/verification_costs.json',{'stages':costs,'records':preparation,
        'wall_until_selection_seconds':(cutoff/1e9-min(r['start_ns']/1e9 for r in preparation)),
        'scope':'Additional candidate export, calibration speed and validation before immutable selection. Process elapsed includes CPU/I/O/init/evaluation, not GPU-active kernel seconds. Concurrent stage envelopes overlap; do not sum them as serial wall time. Final held-out test and downstream tasks excluded.'})
    lines=['# Run005：数值网格、反向约束和独立可行性验证','',
        f'候选池在验证前冻结：{len(pool["policies"])}份唯一策略、{len(pool["points"])}个原始比较点。保持run002质量拟合与run003完整forward成本不变。验证后选择写入并哈希，随后才执行最终test。','',
        '## 前向/反向求解','',
        '原2000-bin DP与real-valued one-hot MILP对照。前向min预测Linear延迟、约束预测ΔNLL；反向min预测ΔNLL、约束预测Linear延迟。P+D在无switch交互项的同一代理下拆成288个phase-module，数值质量系数缩放改善求解容差。原DP仍按144个phase-pair动作计算。',
        f'{len(solver["exhaustive_checks"])}个随机小实例双向穷举检查、{len(pareto["checks"])}个前向↔反向Pareto复核通过。其最优性只针对冻结可加代理，不保证真实NLL或TTFT/E2E。','',
        '| 场景 | 质量预算 | 原DP预测Linear ms | 精确预测Linear ms | DP/精确 |','| --- | ---: | ---: | ---: | ---: |']
    for scenario in ['prefill_only','prefill_decode']:
        for budget in design['quality_budgets']:
            dp=next(r for r in pool['points'] if r['scenario']==scenario and r['family']=='original DP' and r['quality_budget']==budget)
            exact=next(r for r in pool['points'] if r['scenario']==scenario and r['family']=='exact forward' and r['quality_budget']==budget)
            lines.append(f'| {scenario} | {budget:g} | {dp["predicted_linear_ms"]:.2f} | {exact["predicted_linear_ms"]:.2f} | {dp["predicted_linear_ms"]/exact["predicted_linear_ms"]:.3f} |')
    lines+=['','原网格逐模块上取整可累计显著保守误差：uniform SQ-INT8的prefill预测ΔNLL原始约0.001799，网格后约0.028430；P+D分别约−0.001479与0.036625。精确解消除此数值漏选，但不修复质量代理偏差。','',
        '## 实际质量和速度','',
        '验证、最终test分别为独立WikiText validation/test split各64个非重叠窗口。prefill和P+D质量均按目标B8实际执行；prefill计2047个prompt target，P+D计64个teacher-forced target。原003的prefill采用B1，本轮B8结果应单独比较。速度统一GPU2、B8/S2048/O1或64、6GiB BF16 KV、2 warmups+5 timed，校准/最终输入seed分开。','',
        'DASO+validation使用预先固定的mean ΔNLL + 2 paired SE + 0.005 guard，BF16锚点实际差值恒为0。质量可行的候选中选择校准实测延迟最小者。该额外验证阶段不属于原文实现，不将guard称为形式化泛化保证。原DP、精确解及所有预定预算均保留，不用test反馈挑点。','',
        '| 场景 | 方法 | 质量预算 | 实测ΔNLL | 描述性95%区间 | 实测ms | 相对BF16 | 超预算 |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        if r.get('infeasible') or r['family'] in ['exact inverse','inverse plus validation']:continue
        budget=f'{r["quality_budget"]:g}' if 'quality_budget' in r else '—'
        violation=f'{r["budget_violation"]:.5f}' if 'budget_violation' in r else '—'
        lines.append(f'| {r["scenario"]} | {r["label"]} | {budget} | {r["measured_delta_nll"]:.5f} | [{r["delta_nll_ci_low"]:.5f}, {r["delta_nll_ci_high"]:.5f}] | {r["measured_ms"]:.1f} | {r["speedup"]:.3f}× | {violation} |')
    lines+=['','## 实际质量预算违约','',
        '| 场景 | 版本 | 点数 | 超预算点数 | 最大超预算 |','| --- | --- | ---: | ---: | ---: |']
    for r in quality:lines.append(f'| {r["scenario"]} | {r["family"]} | {r["points"]} | {r["violations"]} | {r["max_violation"]:.6f} |')
    lines+=['','## 反向SLA：代理约束与实际约束分别检查','',
        '三个SLA比例为BF16校准实测的95%/80%/65%。raw inverse只约束预测Linear成本；verification inverse在候选中先要求校准中位延迟×1.05不超过实际SLA，再选择guarded validation损失最小者。表中“不可行”明确保留，不静默回退到违约配置。最终新seed速度才用于审计，同时报告中位数与全部5个timed样本，不能用中位数满足代替逐请求硬SLA保证。','',
        '| 场景 | 方法 | SLA ms | 最终中位ms | 最慢timed ms | 中位超出ms | 违约timed/5 | 实测ΔNLL |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if r['family'] not in ['exact inverse','inverse plus validation']:continue
        if r.get('infeasible'):lines.append(f'| {r["scenario"]} | {r["label"]} | {r["actual_latency_sla_ms"]:.1f} | 不可行 | — | — | — | — |')
        else:lines.append(f'| {r["scenario"]} | {r["label"]} | {r["actual_latency_sla_ms"]:.1f} | {r["measured_ms"]:.1f} | {r["max_timed_ms"]:.1f} | {r["actual_sla_violation_ms"]:.1f} | {r["timed_sla_violations"]}/5 | {r["measured_delta_nll"]:.5f} |')
    lines+=['','## 成本与适用范围','',
        '- 额外候选验证成本包含模型导出、初始化、64块validation评分与实测速度，完整命令/退出耗时位于logs。该步骤不是免费，也不应计为纯MILP搜索成本。',
        '- 36/64/80层搜索计时与内存记录在exact_solver_audit.json；动作表扩展来自8B，不能作为30B模型质量或端到端部署证据。',
        '- 质量区间为4000次配对窗口bootstrap描述性区间；窗口在各split内不重叠，但仍不提供分布无关的质量/SLA保证。',
        '- 同步同质batch、eager、无chunked prefill；真实mixed-token结果另见run006。uniform SQ/FP8完整下游任务的既有证据见run002，新策略不能自动继承其MMLU/ROUGE结论。','',
        '| 额外离线阶段 | 命令数 | 累计进程s | 该阶段wall包络s | 使用GPU |','| --- | ---: | ---: | ---: | --- |']
    for r in costs:lines.append(f'| {r["stage"]} | {r["commands"]} | {r["sum_process_seconds"]:.1f} | {r["elapsed_wall_envelope_seconds"]:.1f} | {r["gpus"]} |')
    lines+=['',
        '进程累计时间含I/O与初始化，不是纯GPU-active time。三个阶段并行重叠，wall包络不能直接相加。这里只计额外验证选择成本，原SparseGPT/质量训练标签/shape profiling成本仍需另计；最终test和run010下游任务不属于部署前选择成本。','',
        '![独立测试质量—速度](results/figures/quality_speed.png)','',
        '[完整数表](results/summary_final.csv) · [选择规则](configs/design.json) · [固定选择](configs/selected_policies.json) · [SLA/质量审计](results/quality_sla_audit.json) · [数值检查](results/surrogate_audit.json)。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    figures=RUN/'results/figures';figures.mkdir(exist_ok=True)
    colors={'uniform':'#777777','original DP':'#dd8e27','exact forward':'#4c7fb6','DASO plus validation':'#bb3f45'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,scenario in zip(axes,['prefill_only','prefill_decode'],strict=True):
        for family,color in colors.items():
            values=[r for r in rows if r['scenario']==scenario and r['family']==family]
            ax.scatter([r['measured_delta_nll'] for r in values],[r['speedup'] for r in values],label=family,c=color,s=45)
        ax.axhline(1,color='black',lw=.7);ax.axvline(.01,color='black',ls=':',lw=.7);ax.set_title(scenario);ax.set_xlabel('Held-out measured delta NLL');ax.set_ylabel('Speedup over BF16');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'quality_speed.{ext}',dpi=180)
    plt.close(fig)
    write_json(RUN/'results/report_complete.json',{'complete':True,'points':len(rows),'unique_test_policy_scenarios':len(cache),
        'quality':quality,'latency_sla':sla,'selection_before_test':True,'all_quality_score_batch':8,'all_final_speed_gpu':2})


if __name__=='__main__':main()
