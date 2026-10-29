"""Complete current-cohort Qwen scores; retain every earlier table and result."""
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from launch import RUN,read,write
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def save(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    design=read(RUN/'configs/design.json');legacy=Path(design['legacy_table']);sources={}
    for name,h in design['legacy_table_sha256'].items():assert sha(legacy/name)==h
    for path,h in design['input_sha256'].items():assert sha(Path(path))==h
    for name,h in design['numeric_sha256'].items():assert sha(RUN/'code'/name)==h
    for pid,h in design['policies'].items():assert sha(RUN/f'configs/policies/{pid}.json')==h
    device=read(RUN/'configs/baseline_formal_device.json');records={};rounds=[];deltas=[];result=[]
    for ri,order in enumerate(design['rounds']):
        for label in order:
            path=RUN/f'results/measurements/repeat_round{ri}_{label}.json';record=read(path);sources[str(path)]=sha(path)
            assert record['placement']['cpu_affinity']==device['cpu_affinity']
            assert record['input_sha256']==sha(RUN/'configs/speed_inputs.json')
            assert all(v['exact_equal']for v in record['exact_weight_audit'].values())
            for c in record['cases']:
                assert len([s for s in c['samples']if not s['warmup']])==5
                assert all(s['gpu_before']['physical_gpu_index']==device['physical_gpu']for s in c['samples'])
            records[ri,label]=record
    for item in design['rows']:
        pid=item['policy'];row={'method':item['method'],'policy':pid};scores=[]
        for task,key,column,count in [('cnn_dm_1000','rougeL_percent','cnn_dm',1000),('dsum','rougeL_percent','dialogsum',1500),('IWSLT','sacre_bleu','iwslt',333)]:
            path=RUN/f'results/generation/{pid}/{task}/metrics_0000_end.json';value=read(path)
            assert value['samples']==count and value['input_sha256']==sha(RUN/f'data/generation_{task}.jsonl')
            assert value['max_new_tokens']==256 and value['begin']==0 and value['end']is None
            answers=path.with_name('answers_0000_end.jsonl');current=[json.loads(line)for line in answers.read_text().splitlines()]
            assert len(current)==count;row[column]=value[key];scores.append(value[key]);sources[str(path)]=sha(path);sources[str(answers)]=sha(answers)
            old=design['previous_generation'].get(pid,{}).get(task)
            if old:
                previous=Path(old['file']);assert sha(previous)==old['sha256'];prior=read(previous)
                old_answers=previous.with_name('answers_0000_end.jsonl');archived=[json.loads(line)for line in old_answers.read_text().splitlines()]
                assert len(archived)==count and prior['input_sha256']==value['input_sha256']
                assert [(a['question_id'],a['reference'])for a in archived]==[(a['question_id'],a['reference'])for a in current]
                token_complete=all('generated_token_ids'in a for a in archived)
                deltas.append({'policy':pid,'task':task,'samples':count,'previous_score':prior[key],'repeat_score':value[key],
                    'difference':value[key]-prior[key],'changed_text':sum(a['prediction']!=b['prediction']for a,b in zip(archived,current)),
                    'token_archive_available':token_complete,
                    'changed_token_sequences':sum(a['generated_token_ids']!=b['generated_token_ids']for a,b in zip(archived,current))if token_complete else None,
                    'previous_source':str(previous)})
                sources[str(previous)]=sha(previous);sources[str(old_answers)]=sha(old_answers)
        row['pd_avg_score']=statistics.mean(scores)
        local=[]
        for ri in range(3):
            for output in [1,64]:
                value=next(c for c in records[ri,pid]['cases']if c['output']==output)
                anchors=[s for label in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']
                    for c in records[ri,label]['cases']if c['output']==output for s in c['samples']if not s['warmup']]
                actual=anchors if pid=='uniform_dense_bf16'else[s for s in value['samples']if not s['warmup']]
                int8=next(c for c in records[ri,'uniform_int8']['cases']if c['output']==output)['medians']
                keys=['e2e_ms']if output==1 else['ttft_ms','tpot_ms','e2e_ms','cycle_ms','restore_ms']
                r={'policy':pid,'round':ri,'output':output}
                for key in keys:
                    r[key]=statistics.median(s[key]for s in actual)
                    if key!='restore_ms':r[key+'_vs_bf16']=statistics.median(s[key]for s in anchors)/r[key]
                if output==64:r['e2e_vs_int8']=int8['e2e_ms']/r['e2e_ms'];r['cycle_vs_int8']=int8['cycle_ms']/r['cycle_ms']
                local.append(r);rounds.append(r)
        pd=[r for r in local if r['output']==64];pref=[r for r in local if r['output']==1]
        row['pd_map_o1_speedup_diagnostic']=statistics.mean(r['e2e_ms_vs_bf16']for r in pref)
        for key,column in [('ttft_ms','pd_ttft_speedup'),('tpot_ms','pd_tpot_speedup'),('e2e_ms','pd_e2e_speedup')]:
            row[column]=statistics.mean(r[key+'_vs_bf16']for r in pd)
        row.update({'e2e_vs_int8':statistics.mean(r['e2e_vs_int8']for r in pd),
            'e2e_vs_int8_min':min(r['e2e_vs_int8']for r in pd),'e2e_vs_int8_max':max(r['e2e_vs_int8']for r in pd),
            'cycle_vs_int8':statistics.mean(r['cycle_vs_int8']for r in pd),
            'e2e_ms':statistics.mean(r['e2e_ms']for r in pd),'restore_ms':statistics.mean(r['restore_ms']for r in pd)})
        result.append(row)
    assert len(result)==10 and len(deltas)==24
    save(RUN/'results/repeated_pd.csv',result);save(RUN/'results/previous_generation_differences.csv',deltas)
    fields=list(dict.fromkeys(k for r in rounds for k in r))
    with(RUN/'results/speed_rounds.csv').open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rounds)
    dest=RUN.parents[1]/'responses/reviewer2/qwen_full_repeat_run048';dest.mkdir(parents=True,exist_ok=True)
    lines=['# Qwen十配置统一完整生成重复（R2 Exi5 / R3 LrT2）','',
        '040九map加固定alpha=.5的SQ-FP8，全部当前共同数值/Graph路径。原完整1000/1500/333、native Qwen3 enable_thinking=False、greedy/EOS/max256、B8；全部方法重复，不根据有利结果删点。旧002与028 INT8曾出现分数差异，本轮记录新旧逐样本输出差异，不预设原因或排名。','',
        f'同GPU{device["physical_gpu"]}/NUMA{device["numa_node"]}、KV6GiB、B8/S2048/O64，三轮3warmup/5timed、首尾BF16；先各轮归一化再算术平均，三轮范围不是置信区间。E2E含P→D，完整cycle另含下一批恢复。十套P+D map的O1只作诊断，不当作独立Prefill-only质量配置。','',
        '| 方法 | CNN/DM | DialogSum | IWSLT | P+D Avg | TTFT × | TPOT × | E2E × | ×INT8范围 | Cycle ×INT8 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result:
        lines.append(f'| {r["method"]} | {r["cnn_dm"]:.3f} | {r["dialogsum"]:.3f} | {r["iwslt"]:.3f} | {r["pd_avg_score"]:.3f} | {r["pd_ttft_speedup"]:.3f} | {r["pd_tpot_speedup"]:.3f} | {r["pd_e2e_speedup"]:.3f} | {r["e2e_vs_int8_min"]:.3f}–{r["e2e_vs_int8_max"]:.3f} | {r["cycle_vs_int8"]:.3f} |')
    lines+=['','本轮没有重新拟合/选q或新增NLL预算确认；探索点的旧q是偏好预算，真实预算见各旧原点和新窗口复核，不能因生成分数高就标预算通过。HAWQ为PTQ硬件适配，硬件最小成本端点单列。全部原结果与差异保留，不能把一次重复替换为历史真值或推断通用最优。','',
        '[全精度结果](repeated_pd.csv) · [新旧输出差异](previous_generation_differences.csv) · [三轮速度](speed_rounds.csv) · [原始代码/输出](../../../runs/048/README.md) · [来源](provenance.json)','']
    (dest/'README.md').write_text('\n'.join(lines))
    for name in ['repeated_pd.csv','previous_generation_differences.csv','speed_rounds.csv']:shutil.copy2(RUN/'results'/name,dest/name)
    old=list(csv.DictReader((legacy/'main_table.csv').open()));before=[dict(r)for r in old];lookup={r['method']:r for r in old}
    for item,value in zip(design['rows'],result,strict=True):
        row={**lookup[item['prefill_reuse_method']],'method':value['method']+' (048 P+D repeat)'}
        row.update({k:value[k]for k in ['cnn_dm','dialogsum','iwslt','pd_avg_score','pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']});old.append(row)
    assert len(old)==34 and old[:24]==before
    master=RUN.parents[1]/'responses/reviewer2/qwen3_table2_night161';master.mkdir(parents=True,exist_ok=True);save(master/'main_table.csv',old)
    snapshot=master/'reference/original24';snapshot.mkdir(parents=True,exist_ok=True)
    for name in ['main_table.csv','README.md','provenance.json']:shutil.copy2(legacy/name,snapshot/name)
    table=['# Qwen完整Table2：原24行保留，追加048十行','',
        '新十行的P+D完整质量/速度统一重复；Prefill-only四项精度及独立速度保留旧执行版本的独立结果，未用048当前内核重测MC。DASO两场景各有独立配置，不能把本轮P+D map的O1诊断与旧MC拼接。旧24行全部数值保持。','',
        '| 方法 | ARC-C | ARC-E | Wino | MMLU | Prefill Avg | TTFT × | CNN/DM | DialogSum | IWSLT | P+D Avg | P+D TTFT × | TPOT × | E2E × |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    numeric=['arc_c','arc_e','wino','mmlu','prefill_avg_acc','prefill_ttft_speedup','cnn_dm','dialogsum','iwslt','pd_avg_score','pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']
    for row in old:table.append('| '+row['method']+' | '+' | '.join(f'{float(row[k]):.3f}'for k in numeric)+' |')
    table+=['','[CSV](main_table.csv) · [统一P+D重复与原始来源](../qwen_full_repeat_run048/README.md) · [旧24行完整保留](../qwen3_table2_run016/README.md)','']
    (master/'README.md').write_text('\n'.join(table))
    for path in(RUN/'code').rglob('*'):
        if path.is_file():sources[str(path)]=sha(path)
    audit={'complete':True,'verified':True,'maps':10,'full_generation_jobs':30,'formal_processes':33,'formal_device':device,
        'design_sha256':sha(RUN/'configs/design.json'),'source_sha256':sources,'retained_legacy_rows':24,
        'legacy_table_sha256':design['legacy_table_sha256'],'master_sha256':{p.name:sha(p)for p in master.iterdir()if p.is_file()and p.name!='provenance.json'},
        'delivery_sha256':{p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'}}
    write(RUN/'results/final_audit.json',audit);write(dest/'provenance.json',audit);write(master/'provenance.json',audit)
    print('Published048:10 current-cohort rows,old24 retained in34-row master',flush=True)


if __name__=='__main__':main()
