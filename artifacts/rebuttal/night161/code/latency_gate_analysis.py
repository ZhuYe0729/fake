"""Preregistered first-round measurement gate, checked on two later rounds."""
import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT.parent/'runs/041';OUT=ROOT/'latency_validation'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def save(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def choose(candidates,cap):
    passing=[r for r in candidates if r['guard_ms']<=cap]
    return min(passing,key=lambda r:(r['objective'],r['median_ms'],r['policy']))if passing else None


def self_check():
    rows=[{'policy':'slow','objective':-1.,'median_ms':95.,'guard_ms':101.},
        {'policy':'quality','objective':0.,'median_ms':80.,'guard_ms':90.},
        {'policy':'fast','objective':1.,'median_ms':70.,'guard_ms':80.}]
    assert choose(rows,100)['policy']=='quality'
    assert choose(rows,85)['policy']=='fast'
    assert choose(rows,79)is None
    assert choose(rows,90)['policy']=='quality'
    tie=[{**rows[1],'policy':'slower','median_ms':81.},rows[1]]
    assert choose(tie,100)['policy']=='quality'
    return {'verified':True,'checks':5,'no_feasible_returns_none':True,'quality_objective_first':True,'boundary_inclusive':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);check=self_check()
    if args.freeze:
        assert not list((RUN/'results/measurements').glob('*.json'))
        assert not(OUT/'preregistered.json').exists()
        write(OUT/'self_checks.json',check)
        write(OUT/'preregistered.json',{'created_unix':time.time(),'code_sha256':sha(Path(__file__)),
            'design_sha256':sha(RUN/'configs/design.json'),'plan_sha256':sha(ROOT.parents[2]/'dev/plans/171_preregistered_latency_gate_plan.md'),
            'guard_multiplier':1.05,'calibration_rounds':[0],'confirmation_rounds':[1,2],
            'fractions':[.95,.8,.65],'metrics':['e2e_ms','cycle_ms'],'scope':'Preregistered split analysis, no new GPU acquisition.'})
        print('Frozen latency gate before all041 measurements',flush=True);return
    seal=read(OUT/'preregistered.json');assert seal['code_sha256']==sha(Path(__file__))
    assert seal['design_sha256']==sha(RUN/'configs/design.json')
    if args.wait:
        while not(RUN/'results/final_audit.json').exists():time.sleep(20)
    assert read(RUN/'results/final_audit.json')['verified']
    design=read(RUN/'configs/design.json');sources={};loaded={}
    def samples(ri,pid,output,key):
        path=RUN/f'results/measurements/matched_round{ri}_{pid}.json'
        assert path.stat().st_mtime>seal['created_unix']
        if(ri,pid)not in loaded:
            loaded[ri,pid]=read(path);sources[str(path)]=sha(path)
        case=next(c for c in loaded[ri,pid]['cases']if c['output']==output)
        values=[r[key]for r in case['samples']if not r['warmup']];assert len(values)==8
        return values
    candidates=[];selected=[]
    groups=sorted(set((p['scenario'],p['phase_mode'],p['allocator'])for p in design['points']))
    for scene,mode,allocator in groups:
        pool=[p for p in design['points']if(p['scenario'],p['phase_mode'],p['allocator'])==(scene,mode,allocator)]
        assert len(pool)==3;output=1 if scene=='prefill_only'else 64
        for key in seal['metrics']:
            bf=statistics.median(samples(0,'uniform_dense_bf16',output,key)+samples(0,'uniform_dense_bf16_bracket_end',output,key))
            local=[]
            for p in pool:
                values=samples(0,p['policy_id'],output,key)
                item={'scenario':scene,'phase_mode':mode,'allocator':allocator,'metric':key,'policy':p['policy_id'],
                    'source_r':p['latency_fraction'],'objective':p['solver']['objective'],
                    'median_ms':statistics.median(values),'maximum_ms':max(values),'guard_ms':max(values)*seal['guard_multiplier']}
                local.append(item);candidates.append(item)
            for fraction in seal['fractions']:
                cap=bf*fraction;winner=choose(local,cap)
                original=next(p for p in pool if p['latency_fraction']==fraction)
                selected.append({'scenario':scene,'phase_mode':mode,'allocator':allocator,'metric':key,'target_r':fraction,
                    'calibration_bf16_ms':bf,'target_ms':cap,'old_absolute_e2e_target_ms':original['actual_e2e_target_ms'],
                    'original_policy':original['policy_id'],'selected_policy':winner['policy']if winner else None,
                    'feasible_count':sum(r['guard_ms']<=cap for r in local),'guard_multiplier':seal['guard_multiplier']})
    assert len(selected)==36 and not any(ri>0 for ri,_ in loaded)
    payload={'selection':selected,'preregistered_sha256':sha(OUT/'preregistered.json'),
        'calibration_sha256':dict(sources),'confirmation_read':False}
    path=OUT/'selection.json'
    if path.exists():assert read(path)==payload
    else:write(path,payload)
    frozen=sha(path);save(OUT/'calibration_candidates.csv',candidates)
    # Confirmation raw records and the final quality/speed CSV are first read here.
    quality={r['policy']:r for r in csv.DictReader((RUN/'results/matched_points.csv').open())}
    results=[];details=[]
    for index,choice in enumerate(selected):
        scene=choice['scenario'];output=1 if scene=='prefill_only'else 64;key=choice['metric'];cap=choice['target_ms']
        row={**choice,'selection_feasible':choice['selected_policy']is not None}
        for role,pid in [('original',choice['original_policy']),('selected',choice['selected_policy'])]:
            if pid is None:
                row.update({role+'_misses':None,role+'_samples':0,role+'_median_ms':None,role+'_delta_nll':None});continue
            values=[]
            for ri in [1,2]:
                current=samples(ri,pid,output,key);values+=current
                details += [{'selection_label':index,'role':role,'round':ri,'sample':j,'policy':pid,'metric':key,
                    'target_ms':cap,'actual_ms':value,'violation':value>cap}for j,value in enumerate(current)]
            row.update({role+'_misses':sum(v>cap for v in values),role+'_samples':len(values),
                role+'_median_ms':statistics.median(values),role+'_delta_nll':float(quality[pid]['delta_nll'])})
        results.append(row)
    assert sha(path)==frozen
    dest=ROOT.parent/'responses/reviewer5/latency_validation_night161';dest.mkdir(parents=True,exist_ok=True)
    import shutil
    for name,values in [('latency_validation.csv',results),('confirmation_samples.csv',details),('calibration_candidates.csv',candidates)]:
        save(OUT/name,values);shutil.copy2(OUT/name,dest/name)
    for name in ['selection.json','preregistered.json','self_checks.json']:shutil.copy2(OUT/name,dest/name)
    feasible=[r for r in results if r['selection_feasible']]
    lines=['# 预先固定的实测延迟校验（R5 gkP9 / R3 LrT2）','',
        '041三轮原18点完整保留，零新增GPU测量。首个速度输出前冻结规则/源码SHA，round0校验，round1/2共16个timed样本检查。分析在原始测量收集完后执行：selection先于confirmation读取保存，但不是第二轮采集前发生的在线干预。','',
        '按场景/阶段/分配器分六池，每池只有三个既有incumbent。物理cap=r×round0该场景BF16首尾16样本中位数，E2E/cycle分别定义；与原041绝对E2E cap不同，不能用新cap宣称旧上限通过。候选round0最大值×1.05须通过；再按各自原solver objective、校验中位数、policy id选择。无通过点明确不可行，不能用更慢BF16冒充满足速度预算。','',
        f'36个标签中{len(feasible)}个可行，{len(results)-len(feasible)}个不可行；可行标签涉及{len(set(r["selected_policy"]for r in feasible))}个唯一map。重复选点和复用测试样本不视为独立成功次数。5%为固定经验余量，不提供覆盖定理或生产SLA保证；ΔNLL复用旧64个已查看窗口。','',
        '| 场景/阶段 | 分配器 | 指标 | r | cap ms | 选择 | 可行候选 | 原点超限/16 | 校验后超限/16 | 选择ΔNLL |',
        '| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in results:
        value='—'if r['selected_policy']is None else f'{r["selected_delta_nll"]:.5f}'
        lines.append(f'| {r["scenario"]}/{r["phase_mode"]} | {r["allocator"]} | {r["metric"]} | {r["target_r"]:.2f} | {r["target_ms"]:.2f} | {r["selected_policy"]or"不可行"} | {r["feasible_count"]} | {r["original_misses"]} | {r["selected_misses"]if r["selection_feasible"]else"—"} | {value} |')
    lines+=['','[完整36标签](latency_validation.csv) · [全部校验候选](calibration_candidates.csv) · [逐样本确认](confirmation_samples.csv) · [选择](selection.json) · [规则冻结](preregistered.json) · [原041结果](../../reviewer2/llama31_matched_allocator_run041/README.md) · [审计](provenance.json)','']
    (dest/'README.md').write_text('\n'.join(lines))
    sources[str(RUN/'results/matched_points.csv')]=sha(RUN/'results/matched_points.csv')
    audit={'complete':True,'verified':True,'labels':36,'feasible_labels':len(feasible),'infeasible_labels':36-len(feasible),
        'unique_selected_maps':len(set(r['selected_policy']for r in feasible)),'preregistered_sha256':sha(OUT/'preregistered.json'),
        'selection_sha256':frozen,'source_sha256':sources,'parent_audit_sha256':sha(RUN/'results/final_audit.json'),
        'code_sha256':sha(Path(__file__)),'delivery_sha256':{p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'}}
    write(OUT/'final_audit.json',audit);write(dest/'provenance.json',audit)
    print('Published latency-gate analysis',len(feasible),'/36 feasible labels',flush=True)


if __name__=='__main__':main()
