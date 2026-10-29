"""Complete original18 matched points with actual speed and SLA diagnostics."""
import csv
import hashlib
import math
import shutil
import statistics
from pathlib import Path

from launch import RUN,REPO,TEMP,read,write


def sha(path):
    with path.open('rb')as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    design=read(RUN/'configs/design.json');device=read(RUN/'configs/sweep_formal_device.json')
    for section in ['protected_sha256','nll_sources_sha256']:
        for name,digest in design[section].items():assert sha(Path(name))==digest,name
    for pid,digest in design['policy_sha256'].items():assert sha(RUN/f'configs/policies/{pid}.json')==digest
    for name,digest in read(RUN/'reference/source_model_fingerprint.json')['files_sha256'].items():assert sha(Path(name))==digest,name
    records={};details=[];rows=[];nll_blocks={}
    for ri,order in enumerate(design['rounds']):
        for label in order:
            record=read(RUN/f'results/measurements/matched_round{ri}_{label}.json');records[ri,label]=record
            assert record['placement']['cpu_affinity']==device['cpu_affinity']
            assert all(s['gpu_before']['physical_gpu_index']==device['physical_gpu']
                for c in record['cases']for s in c['samples'])
            assert record['input_sha256']==design['input_sha256']
            assert all(v['exact_equal']for v in record['exact_weight_audit'].values())
            for case in record['cases']:
                assert sum(not s['warmup']for s in case['samples'])==8
                assert all(math.isfinite(s['e2e_ms'])and s['e2e_ms']>0 for s in case['samples'])
    for p in design['points']:
        pid=p['policy_id'];scene=p['scenario'];output=1 if scene=='prefill_only'else 64
        nll=read(RUN/f'results/confirmation/{pid}/{scene}.json');base=read(RUN/f'results/confirmation/uniform_dense_bf16/{scene}.json')
        assert nll['data_sha256']==base['data_sha256']and len(nll['blocks'])==len(base['blocks'])==64
        delta=[a['avg_nll']-b['avg_nll']for a,b in zip(nll['blocks'],base['blocks'],strict=True)];nll_blocks[p['label']]=delta
        cap=p['actual_e2e_target_ms'];timings=[];samples=[];miss_relative=0
        for ri in range(3):
            actual=next(c for c in records[ri,pid]['cases']if c['output']==output)
            current=[s for s in actual['samples']if not s['warmup']];samples+=current
            anchors=[s for name in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']
                for c in records[ri,name]['cases']if c['output']==output for s in c['samples']if not s['warmup']]
            assert len(current)==8 and len(anchors)==16
            bf=statistics.median(s['e2e_ms']for s in anchors);relative_cap=p['latency_fraction']*bf
            row={'label':p['label'],'policy':pid,'scenario':scene,'round':ri,'target_ms':cap,'relative_target_diagnostic_ms':relative_cap,
                'e2e_ms':statistics.median(s['e2e_ms']for s in current),'cycle_ms':statistics.median(s['cycle_ms']for s in current),
                'restore_ms':statistics.median(s['restore_ms']for s in current)}
            row['speedup']=bf/row['e2e_ms'];row['target_misses']=sum(s['e2e_ms']>cap for s in current)
            row['relative_target_misses_diagnostic']=sum(s['e2e_ms']>relative_cap for s in current);miss_relative+=row['relative_target_misses_diagnostic']
            timings.append(row);details.append(row)
        mean=statistics.mean(delta);se=statistics.stdev(delta)/8
        rows.append({'label':p['label'],'policy':pid,'scenario':scene,'phase_mode':p['phase_mode'],'allocator':p['allocator'],
            'r':p['latency_fraction'],'predicted_linear_ms':p['solver']['predicted_linear_ms'],'linear_limit_ms':p['linear_limit_ms'],
            'original_target_ms':cap,'delta_nll':mean,'paired_se':se,'e2e_ms':statistics.median(t['e2e_ms']for t in timings),
            'e2e_speedup':statistics.median(t['speedup']for t in timings),'speedup_min':min(t['speedup']for t in timings),
            'speedup_max':max(t['speedup']for t in timings),'e2e_target_misses':sum(s['e2e_ms']>cap for s in samples),
            'relative_target_misses_diagnostic':miss_relative,'cycle_ms':statistics.median(t['cycle_ms']for t in timings),
            'cycle_vs_e2e_cap_misses_diagnostic':sum(s['cycle_ms']>cap for s in samples),'timed_samples':len(samples)})
    pairs=[]
    for d in rows:
        if d['allocator']!='daso':continue
        h=next(x for x in rows if(x['scenario'],x['phase_mode'],x['r'],x['allocator'])==(d['scenario'],d['phase_mode'],d['r'],'hawq_signed'))
        dif=[a-b for a,b in zip(nll_blocks[d['label']],nll_blocks[h['label']],strict=True)]
        pairs.append({'scenario':d['scenario'],'phase_mode':d['phase_mode'],'r':d['r'],
            'daso_minus_hessian_nll':statistics.mean(dif),'paired_se':statistics.stdev(dif)/8,
            'daso_e2e_speedup':d['e2e_speedup'],'hessian_e2e_speedup':h['e2e_speedup'],
            'daso_target_misses':d['e2e_target_misses'],'hessian_target_misses':h['e2e_target_misses']})
    assert len(rows)==18 and len(pairs)==9 and all(r['timed_samples']==24 for r in rows)
    for name,data in [('matched_points.csv',rows),('speed_rounds.csv',details),('paired_allocators.csv',pairs)]:save(RUN/'results'/name,data)
    dest=REPO/'artifacts/rebuttal/responses/reviewer2/llama31_matched_allocator_run041';dest.mkdir(parents=True,exist_ok=True)
    lines=['# Llama-3.1：同候选分配与反向约束（R2 Exi5 / R5 gkP9）','',
        '原017全部18个固定map，BF16/INT8/NVFP4候选、同Linear上限与MILP，只改变质量代理。HAWQ为signed Hessian/ILP的PTQ硬件移植；不称CNN QAT/TVM原栈。NLL按SHA复用原64窗口，质量不重新拟合；全部有利/不利点保留。','',
        f'统一GPU{device["physical_gpu"]}/NUMA{device["numa_node"]}，B8/S2048/O1或64，6GiB KV、native decode Graph，三轮各三warmup/八timed，BF16首尾锚点。表中三轮中位数，范围是重复范围。原绝对E2E上限不修改；current-BF16比例诊断与cycle恢复诊断另列CSV，不构成新SLA保证。','',
        '| 场景/阶段 | r | 分配器 | ΔNLL | ±2SE | E2E ms | ×BF16 | 三轮范围 | 原目标ms | E2E超限/24 | Cycle ms |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in rows:lines.append(f'| {r["scenario"]}/{r["phase_mode"]} | {r["r"]:.2f} | {r["allocator"]} | {r["delta_nll"]:.5f} | {2*r["paired_se"]:.5f} | {r["e2e_ms"]:.2f} | {r["e2e_speedup"]:.3f} | {r["speedup_min"]:.3f}–{r["speedup_max"]:.3f} | {r["original_target_ms"]:.2f} | {r["e2e_target_misses"]} | {r["cycle_ms"]:.2f} |')
    lines+=['','[完整18点](matched_points.csv) · [三轮明细](speed_rounds.csv) · [配对代理比较](paired_allocators.csv) · [原配置](design.json) · [审计](provenance.json)','']
    (dest/'README.md').write_text('\n'.join(lines))
    for name in ['matched_points.csv','speed_rounds.csv','paired_allocators.csv']:shutil.copy2(RUN/'results'/name,dest/name)
    shutil.copy2(RUN/'configs/design.json',dest/'design.json')
    assert not list((TEMP/'checkpoints').rglob('*.safetensors'))
    audit={'complete':True,'verified':True,'points':18,'paired_groups':9,'timed_per_point':24,'formal_device':device,
        'protected_sha256':design['protected_sha256'],'nll_sources_sha256':design['nll_sources_sha256'],
        'design_sha256':sha(RUN/'configs/design.json'),
        'evidence_sha256':{p.name:sha(p)for p in(RUN/'results/measurements').glob('*.json')},
        'code_sha256':{str(p.relative_to(RUN/'code')):sha(p)for p in(RUN/'code').rglob('*')if p.is_file()},
        'delivery_sha256':{p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'},
        'retained_failures':[p.name for p in(RUN/'results').glob('failure_*.json')]}
    write(RUN/'results/final_audit.json',audit);write(dest/'provenance.json',audit)
    print('Audited041:18 matched maps,9 pairedgroups',flush=True)


if __name__=='__main__':main()
