"""Publish complete baseline tasks and matched speed without changing old tables."""
import csv
import hashlib
import math
import shutil
import statistics
from pathlib import Path

from launch import RUN,REPO,SPEC,TEMP,read,write


def sha(path):
    with path.open('rb')as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    design=read(RUN/'configs/design.json');device=read(RUN/'configs/baseline_formal_device.json')
    mc_kv=read(RUN/'configs/full_task_protocol.json')['mc_kv_gib']
    memory=[read(p)for p in(RUN/'results').glob('last_memory_*.json')]
    task_memory=[m for m in memory if m['mode']=='tasks']
    assert task_memory and all(m['kv_cache_memory_bytes']==mc_kv*1024**3 for m in task_memory)
    gates=[read(p)for p in(RUN/'results/admission').glob('tasks_max_logprobs_*.json')]
    assert gates and all(g['passed']and g['finite_prompt_logprobs']and g['batch']==8 for g in gates)
    for section in ['protected_sha256','input_sha256','reused_quality_sha256']:
        for name,digest in design[section].items():assert sha(Path(name))==digest,name
    for pid,digest in design['policies'].items():assert sha(RUN/f'configs/policies/{pid}.json')==digest
    for name,digest in read(RUN/'reference/source_model_fingerprint.json')['files_sha256'].items():assert sha(Path(name))==digest,name
    records={};rounds=[];sources={};rows=[]
    for ri,order in enumerate(read(RUN/'configs/baseline_formal_rounds.json')):
        for label in order:
            value=read(RUN/f'results/measurements/baseline_round{ri}_{label}.json');records[ri,label]=value
            assert value['placement']['cpu_affinity']==device['cpu_affinity']
            assert all(s['gpu_before']['physical_gpu_index']==device['physical_gpu']
                for c in value['cases']for s in c['samples'])
            assert value['input_sha256']==sha(RUN/'configs/speed_inputs.json')
            assert all(v['exact_equal']for v in value['exact_weight_audit'].values())
            assert [c['output']for c in value['cases']]==[1,64]
            for case in value['cases']:
                assert sum(not s['warmup']for s in case['samples'])==5
                assert all(math.isfinite(s['e2e_ms'])and s['e2e_ms']>0 for s in case['samples'])
    labels={'uniform_dense_bf16':'BF16','uniform_int8':'INT8 W8A8','uniform_sq_int8':'SmoothQuant INT8',
        'uniform_fp8':'FP8 W8A8','uniform_sq_fp8':'SmoothQuant FP8 (alpha=.5)'}
    for pid in design['policies']:
        origin=RUN if pid=='uniform_sq_fp8'else RUN.parent/design['original_run']
        row={'model':SPEC['model_id'],'method':labels[pid],'policy':pid}
        for task,key,column,count in [('arc_challenge','acc_norm,none','arc_c',1172),('arc_easy','acc,none','arc_e',2376),
            ('winogrande','acc,none','wino',1267),('mmlu','acc,none','mmlu',14042)]:
            path=origin/f'results/tasks/{pid}/{task}/full/result.json';value=read(path)
            assert value['limit']is None and value['num_fewshot']==0 and value['batch_size']==8
            counts=value['raw_lm_eval']['n-samples'];assert sum(v['effective']for v in counts.values())==count
            assert all(v['effective']==v['original']for v in counts.values())
            row[column]=100*value['metrics'][key];sources[str(path)]=sha(path)
        row['prefill_avg_acc']=statistics.mean(row[k]for k in ['arc_c','arc_e','wino','mmlu'])
        for task,key,column,count in [('cnn_dm_1000','rougeL_percent','cnn_dm',1000),('dsum','rougeL_percent','dialogsum',1500),('IWSLT','sacre_bleu','iwslt',333)]:
            path=origin/f'results/generation/{pid}/{task}/metrics_0000_end.json';value=read(path)
            assert value['samples']==count and value['quality_batch']==SPEC['generation_batch']
            assert value['max_new_tokens']==256 and value['input_sha256']==sha(RUN/f'data/generation_{task}.jsonl')
            row[column]=value[key];sources[str(path)]=sha(path)
        row['pd_avg_score']=statistics.mean(row[k]for k in ['cnn_dm','dialogsum','iwslt'])
        for scene,output in [('prefill_only',1),('prefill_decode',64)]:
            path=origin/f'results/confirmation/{pid}/{scene}.json';value=read(path)
            basepath=RUN.parent/design['original_run']/f'results/confirmation/uniform_dense_bf16/{scene}.json';base=read(basepath)
            assert value['data_sha256']==base['data_sha256']and len(value['blocks'])==64
            dif=[a['avg_nll']-b['avg_nll']for a,b in zip(value['blocks'],base['blocks'],strict=True)]
            row[scene+'_delta_nll']=statistics.mean(dif);row[scene+'_paired_se']=statistics.stdev(dif)/8
            sources[str(path)]=sha(path);sources[str(basepath)]=sha(basepath)
            local=[]
            for ri in range(3):
                actual=next(c for c in records[ri,pid]['cases']if c['output']==output)
                anchors=[s for name in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']
                    for c in records[ri,name]['cases']if c['output']==output for s in c['samples']if not s['warmup']]
                current=[s for s in actual['samples']if not s['warmup']]
                timed=anchors if pid=='uniform_dense_bf16'else current
                keys=['e2e_ms']if output==1 else['ttft_ms','tpot_ms','e2e_ms','restore_ms','cycle_ms']
                record={'policy':pid,'scenario':scene,'round':ri}
                for key in keys:
                    record[key]=statistics.median(s[key]for s in timed)
                    if key!='restore_ms':record[key+'_vs_bf16']=statistics.median(s[key]for s in anchors)/record[key]
                local.append(record);rounds.append(record)
            if output==1:row['prefill_ttft_speedup']=statistics.mean(r['e2e_ms_vs_bf16']for r in local)
            else:
                for key,column in [('ttft_ms','pd_ttft_speedup'),('tpot_ms','pd_tpot_speedup'),('e2e_ms','pd_e2e_speedup')]:
                    row[column]=statistics.mean(r[key+'_vs_bf16']for r in local)
                row['pd_e2e_speedup_min']=min(r['e2e_ms_vs_bf16']for r in local);row['pd_e2e_speedup_max']=max(r['e2e_ms_vs_bf16']for r in local)
        assert all(math.isfinite(v)for k,v in row.items()if k not in ['model','method','policy'])
        rows.append(row)
    assert len(rows)==5
    save(RUN/'results/baselines.csv',rows)
    # O1 and O64 contain different fields; write a union rather than losing cycle data.
    fields=list(dict.fromkeys(k for r in rounds for k in r))
    with(RUN/'results/speed_rounds.csv').open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rounds)
    dest=REPO/f'artifacts/rebuttal/responses/reviewer2/{SPEC["model_id"]}_sqfp8_run{RUN.name}';dest.mkdir(parents=True,exist_ok=True)
    columns=['arc_c','arc_e','wino','mmlu','prefill_avg_acc','prefill_ttft_speedup','cnn_dm','dialogsum','iwslt','pd_avg_score','pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']
    lines=[f'# {SPEC["model_id"]}：完整SQ-FP8基线（R2 Exi5 / R3 LrT2）','',
        '固定原128-context activation scales、alpha=.5，已有sq_fp8路径，无alpha搜索或DASO第九候选拟合。四个原参考的完整质量按SHA复用，SQ-FP8全部任务新测；所有五map速度同轮重测。原主表不改，强基线结果完整保留。','',
        f'原MC batch8/{mc_kv}GiB KV、zero-shot全样本，原PMPD/BOS/greedy256/EOS生成batch{SPEC["generation_batch"]}、1000/1500/333全样本。速度GPU{device["physical_gpu"]}/NUMA{device["numa_node"]}、B8/S2048/O1或64、KV{SPEC["evaluation_kv_gib"]}GiB、native decode Graph；三轮3warmup/5timed、首尾BF16，各轮归一化后取三轮算术均值，范围另列CSV；每轮原始输出和权重精确校验保留。范围不是置信区间，Avg是对应任务未舍入分数的等权摘要。','',
        '| 方法 | ARC-C | ARC-E | Wino | MMLU | Prefill Avg | TTFT × | CNN/DM | DialogSum | IWSLT | P+D Avg | P+D TTFT × | TPOT × | E2E × |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append('| '+r['method']+' | '+' | '.join(f'{r[k]:.3f}'for k in columns)+' |')
    lines+=['','旧固定64窗口NLL及误差范围列于CSV，既有公开benchmark不是新盲测。E2E含prefill→decode，不含下一请求恢复；cycle在原始三轮CSV另列。','',
        '[完整CSV](baselines.csv) · [三轮速度](speed_rounds.csv) · [设计](design.json) · [来源](provenance.json)','']
    (dest/'README.md').write_text('\n'.join(lines))
    for name in ['baselines.csv','speed_rounds.csv']:shutil.copy2(RUN/'results'/name,dest/name)
    shutil.copy2(RUN/'configs/design.json',dest/'design.json')
    assert not list((TEMP/'checkpoints').rglob('*.safetensors'))
    audit={'complete':True,'verified':True,'maps':5,'formal_device':device,'quality_sources_sha256':sources,
        'protected_sha256':design['protected_sha256'],'design_sha256':sha(RUN/'configs/design.json'),
        'code_sha256':{str(p.relative_to(RUN/'code')):sha(p)for p in(RUN/'code').rglob('*')if p.is_file()},
        'evidence_sha256':{p.name:sha(p)for p in(RUN/'results/measurements').glob('*.json')},
        'delivery_sha256':{p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'},
        'retained_failures':[p.name for p in(RUN/'results').glob('failure_*.json')]}
    write(RUN/'results/final_audit.json',audit);write(dest/'provenance.json',audit)
    print('Audited',RUN.name,SPEC['model_id'],'all5 full baseline rows',flush=True)


if __name__=='__main__':main()
