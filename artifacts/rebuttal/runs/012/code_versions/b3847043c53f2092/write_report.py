"""Summarize complete placement controls and preselected latency constraints."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def median(xs):return statistics.median(xs) if xs else None


def main():
    assert read(RUN/'results/worker.json')['status']=='complete'
    design=read(RUN/'configs/design.json');selected=read(RUN/'configs/sla_selected.json')
    assert selected['frozen_before_test'] and selected['design_sha256']==digest(RUN/'configs/design.json')
    inherited=read(RUN/'reference/inherited_quality.json')
    for q in inherited['records']:assert digest(Path(q['source']))==q['sha256']
    for pid,info in design['sources'].items():assert digest(RUN/f'configs/policies/{pid}.json')==info['policy_sha256']
    copies=[]
    for i in range(4):
        r=read(RUN/f'results/copy/{i}.json');assert r['data_equal']
        for size in [64,512]:
            values=[s for s in r['samples'] if not s['warmup'] and s['mib']==size];assert len(values)==20
            copies.append({'process':i,'node':r['node'],'mib':size,'median_gpu_ms':median([s['gpu_ms'] for s in values]),
                'median_wall_ms':median([s['wall_ms'] for s in values]),'median_gib_per_second':median([s['gib_per_second'] for s in values])})
    rows=[];raw={};hashes={};invalid_observer=[]
    files=sorted((RUN/'results/speed').glob('*/*/*.json'));assert len(files)==28
    for p in files:
        r=read(p);node=r['node'];number=r['round'];pid=r['policy'];place=r['placement_after_init']
        assert place['cpu_affinity']==list(range(0,8) if node==0 else range(32,40))
        assert r['seed']==(144983 if number<2 else 144995) and r['gpu']=='0'
        valid=r.get('first_token_observer')=='engine_core_before_FINAL_ONLY_filter'
        if not valid:invalid_observer.append(str(p.relative_to(RUN)))
        for c in r['cases']:
            values=[s for s in c['samples'] if not s['warmup']];assert len(c['samples'])==11 and len(values)==8
            assert all(s['generated_tokens']==8*c['output'] for s in values)
            key=(node,number,pid,c['scenario']);raw[key]=values
            hashes[key]={s['output_sha256'] for s in values};assert len(hashes[key])==1
            row={'node':node,'round':number,'split':r['split'],'seed':r['seed'],'policy':pid,'scenario':c['scenario'],
                'median_ms':median([s['e2e_ms'] for s in values]),'min_ms':min(s['e2e_ms'] for s in values),
                'max_ms':max(s['e2e_ms'] for s in values),'restore_ms':median([s['restore_ms'] for s in values]),
                'cycle_ms':median([s['cycle_ms'] for s in values]),'timed_iterations':8,'valid_first_token_observer':valid,
                'ttft_observed_ms':median([s['ttft_observed_ms'] for s in values]) if valid else None,
                'decode_remainder_ms':median([s['decode_remainder_ms'] for s in values]) if valid else None,
                'sm_mhz':median([s['gpu_after']['sm_mhz'] for s in values]),
                'memory_mhz':median([s['gpu_after']['memory_mhz'] for s in values]),
                'host_cpu_khz':median([s['gpu_after']['host_cpu_khz'] for s in values if 'host_cpu_khz' in s['gpu_after']]),
                'temperature_c':median([s['gpu_after']['temperature_c'] for s in values])}
            rows.append(row)
    lookup={(r['node'],r['round'],r['policy'],r['scenario']):r for r in rows}
    equivalence=[]
    for pid in design['policies']:
        for scenario in ['prefill_only','prefill_decode']:
            for a,b in [(0,1),(2,3)]:
                same=hashes[0,a,pid,scenario]==hashes[0,b,pid,scenario]
                equivalence.append({'policy':pid,'scenario':scenario,'round_pair':[a,b],'same_token_ids':same})
    controls=[]
    for i in [0,1]:
        for pid in design['remote_policies']:
            for scenario in ['prefill_only','prefill_decode']:
                local=lookup[0,i,pid,scenario];remote=lookup[1,i,pid,scenario]
                same=hashes[0,i,pid,scenario]==hashes[1,i,pid,scenario]
                equivalence.append({'policy':pid,'scenario':scenario,'round':i,'node_pair':[0,1],'same_token_ids':same})
                controls.append({'round':i,'policy':pid,'scenario':scenario,'local_ms':local['median_ms'],
                    'remote_ms':remote['median_ms'],'remote_over_local':remote['median_ms']/local['median_ms'],'same_output_tokens':same})
    ranking=[]
    for i in range(4):
        base=lookup[0,i,'uniform_dense_bf16','prefill_decode']['median_ms']
        d=lookup[0,i,'transfer_daso5_measured_prefill_decode_04','prefill_decode']['median_ms']
        for pid in design['remote_policies'][1:]+['hawq_fp4_phase_pair_signed_prefill_decode_q0.01']:
            h=lookup[0,i,pid,'prefill_decode']['median_ms']
            ranking.append({'round':i,'split':'calibration' if i<2 else 'test','hawq_policy':pid,
                'bf16_ms':base,'daso5_ms':d,'hawq_ms':h,'daso5_speedup':base/d,'hawq_speedup':base/h,
                'hawq_over_daso_ms':h/d,'daso_faster':d<h})
    sla=[]
    for point in selected['points']:
        record=dict(point)
        if point['policy'] is not None:
            samples=[s for i in [2,3] for s in raw[0,i,point['policy'],'prefill_decode']]
            values=[s['e2e_ms'] for s in samples];assert len(values)==16
            record.update({'test_median_ms':median(values),'test_max_ms':max(values),'timed_samples':16,
                'sample_sla_violations':sum(v>point['sla_ms'] for v in values),'median_sla_violation':median(values)>point['sla_ms'],
                'physical_measurement_ids':[f'node0/round{i}/{point["policy"]}/iteration{s["iteration"]}' for i in [2,3] for s in raw[0,i,point['policy'],'prefill_decode']]})
        sla.append(record)
    for name,values in [('copy_summary',copies),('process_summary',rows),('placement_controls',controls),('ranking',ranking)]:
        with (RUN/f'results/{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,list(values[0]));writer.writeheader();writer.writerows(values)
    write(RUN/'results/sla_audit.json',{'points':sla,'selection_sha256':digest(RUN/'configs/sla_selected.json'),
        'counting':'Same selected policy can be checked against multiple SLA thresholds/rules. physical_measurement_ids identify repeats; do not add checks as independent measurements.'})
    write(RUN/'results/output_equivalence.json',{'checks':equivalence,'all_passed':all(r['same_token_ids'] for r in equivalence)})
    lines=['# Run012：排名重测、运行波动与实测约束','',
        '原001–011与模型策略不变。GPU0、B8/S2048、O1/O64、6GiB BF16 KV、eager、无chunked/prefix cache；所有导出完成后才做正式速度，一次只运行本实验。node0 CPU0–7、membind0；remote为CPU32–39、membind1。未改全局GPU时钟/CPU governor。','',
        '## 质量validation究竟是什么','',
        '原005先用代理生成38份候选，再在独立64窗口validation上测真实策略NLL。质量guard=mean paired delta+2SE+.005，按预算筛选后取校准速度最快点。冻结选择后，另一份64窗口test只作评价。8预算点是4预算×2场景，对应5个选定policy/scenario，包含SQ重复选择和BF16回退；0超限是该流程的经验结果，不是原质量拟合已经修好，也不是所有预算都有压缩收益。',
        '本轮没有新质量标签/质量重新拟合；下述时延选择只用原005 validation统计。旧test NLL仅作为既有质量上下文，不用于重新选择。','',
        '## 1. 放置与拷贝控制','',
        '| ABBA进程 | 内存/CPU节点 | MiB | GPU copy ms | wall ms | GiB/s |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in copies:lines.append(f'| {r["process"]} | {r["node"]} | {r["mib"]} | {r["median_gpu_ms"]:.3f} | {r["median_wall_ms"]:.3f} | {r["median_gib_per_second"]:.2f} |')
    lines+=['',
        'node0/node1的512MiB copy均约50GiB/s，单纯NUMA导致大块H2D带宽骤降的假设没有获得支持。完整模型还同时受CPU放置、时钟、分配与流水线影响；下表测的是CPU/内存联合放置效应，不将其全部归因于内存节点。','',
        '| round | 策略 | 场景 | local ms | remote ms | remote/local | 输出token相同 |','| --- | --- | --- | ---: | ---: | ---: | --- |']
    for r in controls:lines.append(f'| {r["round"]} | {r["policy"]} | {r["scenario"]} | {r["local_ms"]:.1f} | {r["remote_ms"]:.1f} | {r["remote_over_local"]:.3f} | {r["same_output_tokens"]} |')
    lines+=['','## 2. 独立进程中的排名','',
        '每个进程3 warmups+8 timed。round0/1使用完全相同的原011 seed144983，round2/3使用完全相同的新seed144995；各轮六策略输入相同、顺序交错，分开比较进程重复与输入变化。下表每行是一个独立进程中位数的配对比较，不把同进程8次当作8个独立campaign。','',
        '| round | 分区 | HAWQ点 | DASO5 ms | HAWQ ms | DASO/BF16 | HAWQ/BF16 | HAWQ/DASO ms |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in ranking:lines.append(f'| {r["round"]} | {r["split"]} | {r["hawq_policy"]} | {r["daso5_ms"]:.1f} | {r["hawq_ms"]:.1f} | {r["daso5_speedup"]:.3f}× | {r["hawq_speedup"]:.3f}× | {r["hawq_over_daso_ms"]:.3f} |')
    lines+=['',
        '首批首token观察使用LLMEngine.step末端，但LLM.generate强制FINAL_ONLY，因此这些未标记版本的TTFT字段实际是完成时刻，全部排除分解。E2E/restore与输出token不受影响，原文件保留。修正后在EngineCore原始new_token_ids进入输出过滤前观测，并断言首token早于完成；process_summary.csv有valid_first_token_observer标记。','',
        '## 3. 5%与经验上界：校准和test严格分开','',
        '原14%示例是1.740s实测校准→1.985s实测test，5%检查为1.740×1.05≤1.921s。它是实测状态/工作负载变化，不直接等于模型预测误差。原raw inverse还另有Linear代理未覆盖整模型代价的问题，两者需要分别处理。','',
        '本轮round0/1提供每候选16个校准timed值；预先固定median×1.05与max_observed×1.02两条规则。SLA为新BF16校准中位的95/80/65%。在原BF16/inverse_r0.65/exact_q0.02三候选中，按原validation guarded质量最小选可行者，全部选择在round2之前冻结；round2/3只检查结果。max_observed不是未来最大值的数学上界，经验规则均不构成硬实时保证。','',
        '| 规则 | BF16比例 | SLA ms | 选择 | test中位 ms | test最大 ms | 超时检查/16 |','| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in sla:
        if r['infeasible']:lines.append(f'| {r["rule"]} | {r["fraction"]:.2f} | {r["sla_ms"]:.1f} | 候选池内不可行 | — | — | — |')
        else:lines.append(f'| {r["rule"]} | {r["fraction"]:.2f} | {r["sla_ms"]:.1f} | {r["policy"]} | {r["test_median_ms"]:.1f} | {r["test_max_ms"]:.1f} | {r["sample_sla_violations"]}/16 |')
    lines+=['',
        '相同policy被不同阈值或规则选中会重复检查同一批计时，physical_measurement_ids提供去重依据。该小候选池用于可靠性验证，不代替完整38候选优化，也不把池内不可行写成全局不可能。BF16是质量回退，但可能不满足比BF16更紧的时延目标。','',
        '处理路径是：先稳定执行/计时，再用代理产生候选，真实validation检查决策可行性，以校准误差/波动设计保守界，最后独立test报告违约与不可行。更准的预测有助于减少验证成本；有限验证本身不能保证任意未来输入分布。','',
        '![独立进程速度复核](results/figures/repeated_pd.png)','',
        '[逐进程完整CSV](results/process_summary.csv) · [排名CSV](results/ranking.csv) · [放置对照](results/placement_controls.csv) · [SLA审计](results/sla_audit.json) · [冻结设计](configs/design.json)。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    figures=RUN/'results/figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for pid,label in [(design['policies'][1],'DASO5'),(design['policies'][2],'HAWQ allocated'),(design['policies'][3],'NV4/W4 min-cost')]:
        times=[lookup[0,i,pid,'prefill_decode']['median_ms'] for i in range(4)]
        axes[0].plot(range(4),times,marker='o',label=label)
    axes[0].set_xticks(range(4),['Cal 0','Cal 1','Test 2','Test 3']);axes[0].set_ylabel('Process median E2E (ms)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for pid in design['remote_policies']:
        values=[r['remote_over_local'] for r in controls if r['policy']==pid and r['scenario']=='prefill_decode']
        axes[1].plot([0,1],values,marker='o',label='DASO5' if pid==design['remote_policies'][0] else 'NV4/W4 min-cost')
    axes[1].axhline(1,color='black',lw=.7);axes[1].set_xticks([0,1]);axes[1].set_xlabel('Matched round');axes[1].set_ylabel('Remote / local E2E');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'repeated_pd.{ext}',dpi=180)
    plt.close(fig)
    result={'complete':True,'model_processes':28,'case_records':56,'model_timed_iterations':448,'copy_processes':4,
        'copy_timed_iterations':160,'fresh_quality_labels':0,'same_output_checks':len(equivalence),
        'same_output_all_passed':all(r['same_token_ids'] for r in equivalence),'early_first_token_observer_excluded':invalid_observer,
        'selection_frozen_before_test':True,'node0_model_campaigns':4}
    write(RUN/'results/report_complete.json',result);print(json.dumps(result))


if __name__=='__main__':main()
