"""Report every frozen point, including quality failures and reused sources."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from launch import RUN,SPEC,TEMP,read,write


def sha(path):
    with path.open('rb')as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(name,rows):
    with(RUN/'results'/name).open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    design=read(RUN/'configs/sweep_design.json')
    for name,digest in design['input_sha256'].items():assert sha(Path(name))==digest,name
    for name,digest in design['configuration_sha256'].items():assert sha(RUN/'configs'/name)==digest,name
    source_model=read(RUN/'reference/source_model_fingerprint.json')
    for name,digest in source_model['files_sha256'].items():assert sha(Path(name))==digest,name
    registry=read(RUN/'configs/policy_registry.json');points=read(RUN/'configs/sweep_points.json')
    for pid,entry in registry.items():assert sha(RUN/f'configs/policies/{pid}.json')==entry['policy_sha256']
    base=read(RUN/'results/confirmation/uniform_dense_bf16/prefill_decode.json')['blocks']
    nll={}
    reused=read(RUN/'configs/sweep_evaluation_jobs.json')['reused']
    for item in reused:assert sha(Path(item['source']))==item['sha256']
    for pid in registry:
        value=read(RUN/f'results/confirmation/{pid}/prefill_decode.json')
        assert len(value['blocks'])==64 and value['data_sha256']==design['dataset']['sha256']
        assert all(math.isfinite(b['avg_nll'])for b in value['blocks'])
        differences=[a['avg_nll']-b['avg_nll']for a,b in zip(value['blocks'],base,strict=True)]
        mean=statistics.mean(differences);se=statistics.stdev(differences)/math.sqrt(64)
        nll[pid]={'delta_nll':mean,'paired_se':se,'guarded_delta_nll':0. if pid=='uniform_dense_bf16' else mean+2*se+.005}
    generation=read(RUN/'configs/sweep_generation_jobs.json');metrics={};gen_sources={}
    for item in generation['reused']:
        path=Path(item['source']);assert sha(path)==item['sha256']
        metrics[item['policy'],item['dataset']]=read(path);gen_sources[str(path)]=sha(path)
    for item in generation['jobs']:
        path=RUN/f'results/generation/{item["policy"]}/{item["dataset"]}/metrics_0000_end.json'
        metrics[item['policy'],item['dataset']]=read(path);gen_sources[str(path)]=sha(path)
    gen={}
    for pid in registry:
        row={}
        for task,key,metric,count in [('cnn_dm_1000','cnn_dm','rougeL_percent',1000),
                ('dsum','dialogsum','rougeL_percent',1500),('IWSLT','iwslt','sacre_bleu',333)]:
            value=metrics[pid,task]
            assert value['samples']==count and value['input_sha256']==sha(RUN/f'data/generation_{task}.jsonl')
            assert value['quality_batch']==SPEC['generation_batch']
            assert math.isfinite(value[metric])and 0<=value[metric]<=100
            row[key]=value[metric]
        gen[pid]={**row,'generation_avg':statistics.mean(row.values())}
    rounds=read(RUN/'configs/sweep_formal_rounds.json')['rounds'];device=read(RUN/'configs/sweep_formal_device.json')
    speed_records=[];speed={};counts={}
    for i,order in enumerate(rounds):
        records={pid:read(RUN/f'results/measurements/sweep_formal_round{i}_{pid}.json')for pid in order}
        for record in records.values():
            assert record['input_sha256']==sha(RUN/'configs/speed_inputs.json')
            assert record['placement']['cpu_affinity']==device['cpu_affinity']
            assert all(v['exact_equal']for v in record['exact_weight_audit'].values())
            assert sum(not s['warmup']for s in record['cases'][0]['samples'])==5
            assert all(math.isfinite(record['cases'][0]['medians'][k])and record['cases'][0]['medians'][k]>0
                       for k in ['ttft_ms','tpot_ms','e2e_ms'])
        medians={p:v['cases'][0]['medians']for p,v in records.items()}
        for pid in registry:
            row={'policy':pid,'round':i,**medians[pid]}
            for metric in ['ttft_ms','tpot_ms','e2e_ms']:
                anchor=statistics.mean(medians[p][metric]for p in [order[0],order[-1]])
                row[metric+'_vs_bf16']=anchor/medians[pid][metric]
            row['e2e_vs_int8']=medians['uniform_int8']['e2e_ms']/medians[pid]['e2e_ms']
            speed_records.append(row)
    for pid in registry:
        records=[r for r in speed_records if r['policy']==pid]
        speed[pid]={key:statistics.mean(r[key]for r in records)for key in records[0]if key not in ['policy','round']}
        speed[pid].update(e2e_vs_int8_min=min(r['e2e_vs_int8']for r in records),
                          e2e_vs_int8_max=max(r['e2e_vs_int8']for r in records))
        mapping=read(RUN/f'configs/policies/{pid}.json')['method_map']
        counts[pid]={p:dict(Counter(v[p+'_method']for v in mapping.values()))for p in ['prefill','decode']}
        assert all(sum(v.values())==128 for v in counts[pid].values())
    save('sweep_formal_rounds.csv',speed_records)
    all_rows=[{'policy':pid,**gen[pid],**nll[pid],**speed[pid],
        'prefill_counts':json.dumps(counts[pid]['prefill'],sort_keys=True),
        'decode_counts':json.dumps(counts[pid]['decode'],sort_keys=True)}for pid in registry]
    save('sweep_all_results.csv',all_rows)
    by_id={r['policy']:r for r in all_rows}
    rows=[{'label':p['label'],'prefill_weight':p['prefill_weight'],'budget':p['budget'],
        'weighted_prediction':p['weighted_prediction'],'anchored_prediction':p['anchored_prediction'],
        **by_id[p['policy']], 'mean_within_nominal_budget':nll[p['policy']]['delta_nll']<=p['budget'],
        'guard_within_nominal_budget':nll[p['policy']]['guarded_delta_nll']<=p['budget']}for p in points]
    save('sweep_results.csv',rows)
    display=[(pid,by_id[pid])for pid in design['anchors']]+[(p['label'],by_id[p['policy']])for p in points]
    lines=[f'# {SPEC["model_id"]}：完整空间P+D预算与prefill权重', '',
        '沿用本模型八候选精度代理与实测LUT，去除旧截距并以BF16=0锚定；128模块独立选择64种阶段组合。全部预设标签在本轮GPU输出前冻结，同速解再最小化预测损失。', '',
        'q是代理/加权偏好预算，不是已验证的真实ΔNLL保证。全部点保留，不根据最终生成分数重新训练或声称通用最优。原Prefill-only及原15行主表不改，本轮没有新增MC任务。', '',
        f'速度统一GPU{device["physical_gpu"]}/NUMA{device["numa_node"]}，B8/S2048/O64，原KV{SPEC["evaluation_kv_gib"]}GiB、native decode Graph；三轮3warmup+5timed、BF16首尾锚点、同轮INT8。E2E计入prefill→decode切换，不计下一批恢复。', '',
        f'完整生成保留原PMPD/BOS、batch{SPEC["generation_batch"]}、greedy最多256新token/EOS；1000/1500/333全样本。相同map/源码/输入协议的{SPEC["source_run"]}结果按SHA复用，其他新测。Avg是ROUGE-L%/BLEU三项算术摘要，不是统一准确率。', '',
        '| 设置 | CNN/DM | DialogSum | IWSLT | Gen Avg | TTFT ms | TPOT ms | E2E ms | ×BF16 | ×INT8 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,row in display:
        lines.append('| '+label+' | '+' | '.join(f'{row[k]:.3f}'for k in
            ['cnn_dm','dialogsum','iwslt','generation_avg','ttft_ms','tpot_ms','e2e_ms','e2e_ms_vs_bf16','e2e_vs_int8'])+' |')
    lines+=['','## 真实质量与预算诊断','','既有WikiText test固定64窗口，非新盲测；保护值=mean+2SE+.005。不按NLL预先淘汰预定生成点。', '',
        '| 设置 | q | wP | 加权预测 | 实测ΔNLL | 保护值 | 均值通过 | 保护值通过 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in rows:
        lines.append('| '+r['label']+' | '+' | '.join(f'{r[k]:.5f}'for k in
            ['budget','prefill_weight','weighted_prediction','delta_nll','guarded_delta_nll'])+
            f' | {r["mean_within_nominal_budget"]} | {r["guard_within_nominal_budget"]} |')
    lines+=['','## 自动分配','','| 设置 | Prefill | Decode |','| --- | --- | --- |']
    for p in points:
        c=counts[p['policy']]
        lines.append('| '+p['label']+' | '+str(c['prefill'])+' | '+str(c['decode'])+' |')
    lines+=['','[全部配置与分项](results/sweep_all_results.csv) · [六标签](results/sweep_results.csv) · [三轮速度与范围](results/sweep_formal_rounds.csv) · [数据与参数冻结](configs/sweep_design.json)','']
    (RUN/'SWEEP_REPORT.md').write_text('\n'.join(lines))
    assert not list((TEMP/'checkpoints').rglob('*.safetensors'))
    evidence=[*list((RUN/'results/confirmation').glob('*/*.json')),
        *list((RUN/'results/generation').glob('*/*/*.json*')),*list((RUN/'results/measurements').glob('sweep_formal*.json'))]
    write(RUN/'results/sweep_final_audit.json',{'complete':True,'verified':True,'labels':len(points),
        'unique_maps':len(registry),'formal_device':device,'generation_sources_sha256':gen_sources,
        'reused_nll_sources':reused,'input_sha256':design['input_sha256'],
        'configuration_sha256':{str(p.relative_to(RUN)):sha(p)for p in(RUN/'configs').glob('*.json')},
        'code_sha256':{str(p.relative_to(RUN/'code')):sha(p)for p in(RUN/'code').rglob('*')if p.is_file()},
        'evidence_sha256':{str(p.relative_to(RUN)):sha(p)for p in evidence},
        'csv_sha256':{p.name:sha(p)for p in(RUN/'results').glob('sweep_*.csv')},
        'retained_failures':[p.name for p in(RUN/'results').glob('sweep_failure*.json')]})
    print('Audited',SPEC['model_id'],len(points),'labels',len(registry),'maps',flush=True)


if __name__=='__main__':main()
