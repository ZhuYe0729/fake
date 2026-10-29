"""Audit and publish all nine maps across six frozen workloads."""
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
    for section in ['protected_sha256','input_sha256']:
        for name,digest in design[section].items():assert sha(Path(name))==digest,name
    for name,digest in read(RUN/'reference/source_model_fingerprint.json')['files_sha256'].items():assert sha(Path(name))==digest,name
    for pid,entry in design['policies'].items():
        policy=read(RUN/f'configs/policies/{pid}.json')
        import json
        assert hashlib.sha256(json.dumps(policy['method_map'],sort_keys=True).encode()).hexdigest()==entry['map_sha256']
    raw={}
    for ri,order in enumerate(design['rounds']):
        for label in order:
            record=read(RUN/f'results/speed/round{ri}/{label}.json');raw[ri,label]=record
            assert record['gpu']==str(device['physical_gpu'])
            assert record['placement']['cpu_affinity']==device['cpu_affinity']
            assert all(a['exact_equal']for a in record['exact_weight_audit'].values())
            assert len(record['cases'])==6
            for actual,case in zip(record['cases'],design['workloads'],strict=True):
                assert actual['input_sha256']==case['input_sha256']and actual['label']==case['label']
                assert actual['admission']['whole_batch_fits']
                assert len(actual['samples'])==8 and sum(not s['warmup']for s in actual['samples'])==5
                for sample in actual['samples']:
                    assert len(sample['output_token_ids'])==case['batch']
                    assert all(len(ids)==case['output']for ids in sample['output_token_ids'])
                    assert all(math.isfinite(sample[k])and sample[k]>0 for k in ['e2e_ms','ttft_ms','tpot_ms','cycle_ms'])
    rows=[];rounds=[]
    for method in design['rows']:
        pid=method['policy']
        for ci,case in enumerate(design['workloads']):
            value=read(RUN/f'results/quality/{pid}/{case["label"]}.json')
            baseline=read(RUN/f'results/quality/uniform_dense_bf16/{case["label"]}.json')
            assert value['quality_sha256']==baseline['quality_sha256']==case['quality_sha256']
            assert value['quality_batch']==case['batch']and value['admission']['whole_batch_fits']
            assert len(value['blocks'])==64 and all(b['token_count']==case['output']and math.isfinite(b['avg_nll'])for b in value['blocks'])
            delta=[a['avg_nll']-b['avg_nll']for a,b in zip(value['blocks'],baseline['blocks'],strict=True)]
            mean=statistics.mean(delta);se=statistics.stdev(delta)/8
            row={**method,'workload':case['label'],'batch':case['batch'],'prompt':case['prompt'],'output':case['output'],
                'delta_nll':mean,'paired_se':se,'guarded_delta_nll':0. if pid=='uniform_dense_bf16'else mean+2*se+.005,
                'mean_within_label_budget':mean<=method['budget']if method['budget']is not None else None,
                'guard_within_label_budget':mean+2*se+.005<=method['budget']if method['budget']is not None else None}
            per=[]
            for ri in range(3):
                values=raw[ri,pid]['cases'][ci]['medians']
                anchors=[raw[ri,name]['cases'][ci]['medians']for name in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']]
                int8=raw[ri,'uniform_int8']['cases'][ci]['medians']
                timed={k:statistics.mean(a[k]for a in anchors)if pid=='uniform_dense_bf16'else v for k,v in values.items()}
                record={'policy':pid,'method':method['method'],'workload':case['label'],'round':ri,**timed}
                for key in ['e2e_ms','ttft_ms','tpot_ms','cycle_ms']:
                    record[key+'_vs_bf16']=statistics.mean(a[key]for a in anchors)/timed[key]
                    record[key+'_vs_int8']=int8[key]/timed[key]
                per.append(record);rounds.append(record)
            keys=[k for k in per[0]if k not in ['policy','method','workload','round']]
            row.update({k:statistics.median(r[k]for r in per)for k in keys})
            row.update(e2e_vs_int8_min=min(r['e2e_ms_vs_int8']for r in per),e2e_vs_int8_max=max(r['e2e_ms_vs_int8']for r in per))
            rows.append(row)
    assert len(rows)==54
    save(RUN/'results/workloads.csv',rows);save(RUN/'results/speed_rounds.csv',rounds)
    lines=['# Qwen新选点的六负载迁移（R2 Exi5 / R3 LrT2 / R4 QboE）','',
        '九map、六个原019负载，固定032/033中心选点迁移，无逐负载重新优化。所有基线/点完整保留；q只是原中心代理标签，不是新负载质量保证。64窗口均来自既有已查看WikiText-test，不是新盲测。','',
        f'统一GPU{device["physical_gpu"]}/NUMA{device["numa_node"]}，CPU{device["cpu_affinity"]}；6GiB KV、native decode Graph。每map/轮初始化一次、固定顺序六case，每case三warmup/五timed，三轮顺序不同，BF16首尾锚点。数值为三轮中位数，范围非置信区间。E2E含prefill→decode切换，cycle额外计下一批prefill恢复。','']
    for case in design['workloads']:
        lines += [f'## {case["label"]}: B{case["batch"]}/S{case["prompt"]}/O{case["output"]}','',
            '| 方法 | ΔNLL | ±2SE | TTFT ×BF16 | TPOT ×BF16 | E2E ×BF16 | E2E ×INT8 | 三轮×INT8范围 | Cycle ×BF16 | 恢复ms | q均值/保护值通过 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |']
        for r in rows:
            if r['workload']!=case['label']:continue
            lines.append(f'| {r["method"]} | {r["delta_nll"]:.5f} | {2*r["paired_se"]:.5f} | {r["ttft_ms_vs_bf16"]:.3f} | {r["tpot_ms_vs_bf16"]:.3f} | {r["e2e_ms_vs_bf16"]:.3f} | {r["e2e_ms_vs_int8"]:.3f} | {r["e2e_vs_int8_min"]:.3f}–{r["e2e_vs_int8_max"]:.3f} | {r["cycle_ms_vs_bf16"]:.3f} | {r["restore_ms"]:.2f} | {r["mean_within_label_budget"]}/{r["guard_within_label_budget"]} |')
        lines.append('')
    lines+=['[完整CSV](workloads.csv) · [三轮原始范围](speed_rounds.csv) · [冻结设计](design.json) · [来源审计](provenance.json)','']
    dest=REPO/'artifacts/rebuttal/responses/reviewer2/qwen_updated_workloads_run040';dest.mkdir(parents=True,exist_ok=True)
    for name in ['workloads.csv','speed_rounds.csv']:shutil.copy2(RUN/'results'/name,dest/name)
    shutil.copy2(RUN/'configs/design.json',dest/'design.json')
    (dest/'README.md').write_text('\n'.join(lines))
    (RUN/'REPORT.md').write_text('\n'.join(lines).replace('](workloads.csv)','](results/workloads.csv)').replace('](speed_rounds.csv)','](results/speed_rounds.csv)').replace('](design.json)','](configs/design.json)').replace('](provenance.json)','](results/final_audit.json)'))
    assert not list((TEMP/'checkpoints').rglob('*.safetensors'))
    evidence=[*list((RUN/'results/quality').glob('*/*.json')),*list((RUN/'results/speed').glob('*/*.json'))]
    audit={'complete':True,'verified':True,'maps':9,'workloads':6,'rows':54,'formal_device':device,
        'design_sha256':sha(RUN/'configs/design.json'),'protected_sha256':design['protected_sha256'],
        'source_model_fingerprint_sha256':sha(RUN/'reference/source_model_fingerprint.json'),
        'code_sha256':{str(p.relative_to(RUN/'code')):sha(p)for p in(RUN/'code').rglob('*')if p.is_file()},
        'evidence_sha256':{str(p.relative_to(RUN)):sha(p)for p in evidence},
        'retained_failures':[p.name for p in(RUN/'results').glob('failure_*.json')],
        'delivery_sha256':{p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'}}
    write(RUN/'results/final_audit.json',audit);write(dest/'provenance.json',audit)
    print('Audited040: all54 workload rows',flush=True)


if __name__=='__main__':main()
