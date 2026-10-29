"""Select on fresh validation only and report every confirmation outcome."""
import csv
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path

from launch import RUN,REPO,SPEC,read,write


def sha(path):
    with path.open('rb')as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def quality(pid,split,count):
    design=read(RUN/'configs/design.json');path=RUN/f'results/{split}/{pid}/prefill_decode.json'
    value=read(path);base=read(RUN/f'results/{split}/uniform_dense_bf16/prefill_decode.json')
    assert value['data_sha256']==base['data_sha256']==design['datasets'][split]['sha256']
    assert len(value['blocks'])==64 and value['quality_score_batch']==8
    assert all(b['token_count']==64 and math.isfinite(b['avg_nll'])for b in value['blocks'])
    delta=[a['avg_nll']-b['avg_nll']for a,b in zip(value['blocks'][:count],base['blocks'][:count],strict=True)]
    mean=statistics.mean(delta);se=statistics.stdev(delta)/math.sqrt(count)
    return {'delta_nll':mean,'paired_se':se,'guarded_delta_nll':0. if pid=='uniform_dense_bf16'else mean+2*se+.005}


def select():
    design=read(RUN/'configs/design.json');rows=[];chosen=[]
    for count in design['validation_sizes']:
        values={pid:quality(pid,'validation',count)for pid in design['policies']}
        for pid,value in values.items():rows.append({'validation_size':count,'policy':pid,
            'linear_ms':design['policies'][pid]['linear_ms'],**value})
        for epsilon in design['budgets']:
            feasible=[pid for pid,value in values.items()if value['guarded_delta_nll']<=epsilon]
            assert 'uniform_dense_bf16'in feasible
            pid=min(feasible,key=lambda p:(design['policies'][p]['linear_ms'],values[p]['guarded_delta_nll'],p))
            chosen.append({'validation_size':count,'epsilon':epsilon,'policy':pid,
                'feasible_candidates':len(feasible),'pool_size':len(values),'linear_ms':design['policies'][pid]['linear_ms'],
                'validation_delta_nll':values[pid]['delta_nll'],'validation_guard':values[pid]['guarded_delta_nll']})
    references=['uniform_dense_bf16','uniform_int8','uniform_sq_int8','uniform_fp8']
    union=sorted(set(references+[r['policy']for r in chosen]))
    save(RUN/'results/validation_candidates.csv',rows)
    write(RUN/'configs/fresh_selection.json',{'selected':chosen,'confirmation_policies':union,
        'written_at_unix':time.time(),'confirmation_read':False,'design_sha256':sha(RUN/'configs/design.json'),
        'validation_sha256':{pid:sha(RUN/f'results/validation/{pid}/prefill_decode.json')for pid in design['policies']},
        'rule':design['selection']})
    print('Frozen selection before own confirmation',len(chosen),'labels',len(union),'unique confirmation maps',flush=True)


def main():
    design=read(RUN/'configs/design.json');selection=read(RUN/'configs/fresh_selection.json')
    assert sha(RUN/'configs/fresh_selection.json')==read(RUN/'reference/selection_frozen.json')['sha256']
    assert selection['design_sha256']==sha(RUN/'configs/design.json')
    for section in ['protected_sha256']:
        for name,digest in design[section].items():assert sha(Path(name))==digest,name
    for name,digest in read(RUN/'reference/parent_audits.json').items():assert sha(Path(name))==digest,name
    for name,digest in read(RUN/'reference/source_model_fingerprint.json')['files_sha256'].items():assert sha(Path(name))==digest,name
    assert sha(RUN/'reference/profile.csv')==design['profile_sha256']
    assert sha(RUN/'data/activation_scales.pt')==design['activation_scales_sha256']
    assert sha(RUN/'reference/data_purge.json')==design['purge_sha256']
    for split,data in design['datasets'].items():assert sha(Path(data['file']))==data['sha256']
    for pid,entry in design['policies'].items():
        assert sha(RUN/f'configs/policies/{pid}.json')==entry['policy_sha256']
        assert sha(RUN/f'results/validation/{pid}/prefill_decode.json')==selection['validation_sha256'][pid]
    results={};unique=[]
    stamp=read(RUN/'reference/selection_frozen.json')['written_at_unix']
    for pid in selection['confirmation_policies']:
        a=read(RUN/f'results/representations/validation/{pid}.json');b=read(RUN/f'results/representations/confirmation/{pid}.json')
        assert a['sha256']==b['sha256'],pid
        path=RUN/f'results/confirmation/{pid}/prefill_decode.json';assert path.stat().st_mtime>stamp
        results[pid]=quality(pid,'confirmation',64)
        unique.append({'policy':pid,**results[pid],'linear_ms':design['policies'][pid]['linear_ms'],
            'origins':json.dumps(design['policies'][pid]['origins'],ensure_ascii=False),'representation_sha256':a['sha256']})
    rows=[]
    for choice in selection['selected']:
        actual=results[choice['policy']]
        rows.append({**choice,'confirmation_delta_nll':actual['delta_nll'],'confirmation_paired_se':actual['paired_se'],
            'confirmation_guard':actual['guarded_delta_nll'],'mean_budget_violation':actual['delta_nll']>choice['epsilon'],
            'guard_budget_violation':actual['guarded_delta_nll']>choice['epsilon']})
    assert len(rows)==8
    save(RUN/'results/quality_budget_audit.csv',rows);save(RUN/'results/unique_confirmation.csv',unique)
    dest=REPO/f'artifacts/rebuttal/responses/reviewer1/fresh_quality_run{RUN.name}';dest.mkdir(parents=True,exist_ok=True)
    lines=[f'# {SPEC["model_id"]}：固定池的新窗口质量预算复核（R1 AK6a / R5 gkP9）','',
        f'固定{len(design["policies"])}套既有map，未重拟合质量代理或调q/wP。以新validation32/64窗口的mean+2SE+.005保护值筛选，在四个实际epsilon下选冻结Linear LUT成本最低的incumbent；BF16保护值为0。仅有限池筛选，不称完整空间重新优化。','',
        '数据来自公开WikiText train，排除旧实验校准128个P+D/100个prefill样本和完整旧validation/test分词流中共享的任何连续16token片段。新validation64与confirmation64互不重叠，数据和种子先于GPU输出冻结。不是外部未见语料，不是新的下游任务盲测；NLL窗口SE和保护值均为经验诊断，不是泛化定理。','',
        f'选择文件在confirmation前冻结。八个预算标签实际对应{len(set(r["policy"]for r in rows))}个选择map，确认并集另含BF16和三个8bit参考，共{len(unique)}个map；重复选点不能当作独立成功次数。B8/S2048/O64、KV{SPEC["evaluation_kv_gib"]}GiB、原native decode Graph与数值代码，重导出权重完整SHA须与validation一致。','',
        '| Validation N | epsilon | 配置 | Validation保护值 | 新确认ΔNLL | ±2SE | 确认保护值 | 均值超限 | 保护值超限 |',
        '| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |']
    for r in rows:lines.append(f'| {r["validation_size"]} | {r["epsilon"]:.3f} | {r["policy"]} | {r["validation_guard"]:.5f} | {r["confirmation_delta_nll"]:.5f} | {2*r["confirmation_paired_se"]:.5f} | {r["confirmation_guard"]:.5f} | {r["mean_budget_violation"]} | {r["guard_budget_violation"]} |')
    lines+=['','原q/wP标签与配置来源见design.json/policy origins；本轮epsilon是validation实际筛选预算，不能把原偏好q和它混用。本轮没有新增GEN或速度；相同map的旧下游分数/速度见各origin run的完整报告与来源审计，不能称新独立结果。','',
        '[全部选择及超限](quality_budget_audit.csv) · [唯一确认map](unique_confirmation.csv) · [全部validation候选](validation_candidates.csv) · [冻结选择](selection.json) · [固定池与来源](design.json) · [数据排除](data_purge.json) · [审计](provenance.json)','']
    (dest/'README.md').write_text('\n'.join(lines))
    for name in ['quality_budget_audit.csv','unique_confirmation.csv','validation_candidates.csv']:shutil.copy2(RUN/'results'/name,dest/name)
    for src,target in [('configs/design.json','design.json'),('configs/fresh_selection.json','selection.json'),('reference/data_purge.json','data_purge.json')]:shutil.copy2(RUN/src,dest/target)
    audit={'complete':True,'verified':True,'selection_labels':8,'unique_selected_maps':len(set(r['policy']for r in rows)),
        'unique_confirmation_maps':len(unique),'mean_violations':sum(r['mean_budget_violation']for r in rows),
        'guard_violations':sum(r['guard_budget_violation']for r in rows),'selection_sha256':sha(RUN/'configs/fresh_selection.json'),
        'parent_audits_sha256':read(RUN/'reference/parent_audits.json'),'design_sha256':sha(RUN/'configs/design.json'),
        'code_sha256':{str(p.relative_to(RUN/'code')):sha(p)for p in(RUN/'code').rglob('*')if p.is_file()},
        'evidence_sha256':{str(p.relative_to(RUN)):sha(p)for split in ['validation','confirmation']for p in(RUN/f'results/{split}').glob('*/*.json')},
        'delivery_sha256':{p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'},
        'retained_failures':[p.name for p in(RUN/'results').glob('failure_*.json')]}
    write(RUN/'results/final_audit.json',audit);write(dest/'provenance.json',audit)
    print('Audited',RUN.name,'8 labels;',audit['mean_violations'],'mean budget violations;',len(unique),'unique confirmation maps',flush=True)


if __name__=='__main__':main()
