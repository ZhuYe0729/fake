"""Summarize the fixed pilot; never replace full-task main-table numbers."""
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,rows):
    with (RUN/'results'/name).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def timing(pid):
    source=RUN.parent/'019';rows=[]
    for campaign in range(3):
        folder=source/f'results/speed/round{campaign}'
        paths=[folder/f'{name}.json' for name in [pid,'uniform_dense_bf16','uniform_dense_bf16_bracket_end']]
        if not all(p.exists() for p in paths):continue
        cases=[next(c for c in read(p)['cases'] if c['label']=='center') for p in paths]
        samples=[[s for s in c['samples'] if not s['warmup']] for c in cases]
        assert len(samples[0])==5 and len(samples[1])+len(samples[2])==10
        anchors=samples[1]+samples[2]
        row={'campaign':campaign,'e2e_speedup':statistics.median(s['e2e_ms'] for s in anchors)/statistics.median(s['e2e_ms'] for s in samples[0])}
        for key in ['ttft_ms','tpot_ms','e2e_ms','restore_ms','cycle_ms']:
            row[key]=statistics.median(s[key] for s in samples[0])
        row['cycle_speedup']=statistics.median(s['cycle_ms'] for s in anchors)/row['cycle_ms']
        rows.append(row)
    return {key:statistics.median(r[key] for r in rows) for key in rows[0] if key!='campaign'}|{'campaigns':len(rows)} if rows else {}


def main():
    design=read(RUN/'configs/design.json');rows=[];sources={}
    while not all((RUN/f'full/results/generation/uniform_sq_int8/{task}/metrics_0000_end.json').exists()
                  for task in ['cnn_dm_1000','dsum','IWSLT']):
        status=RUN/'results/full_control_worker.json'
        if status.exists():assert read(status)['status']!='failed',read(status)
        time.sleep(5)
    labels={'sq_int8_w4a16':'SQ-INT8 prefill / W4A16 decode','uniform_sq_int8':'Uniform SQ-INT8',
        'uniform_int8':'Uniform INT8','old_daso7_e001':'Old DASO-7 epsilon=.01 (NVFP4 / SQ-INT8)'}
    for pid in design['policies']:
        row={'policy':pid,'method':labels[pid]}
        for dataset,key in [('cnn_dm_1000','cnn_dm'),('dsum','dialogsum'),('IWSLT','iwslt')]:
            folder=RUN/f'results/generation/{pid}/{dataset}'
            metrics=read(folder/'metrics_0000_end.json')
            assert metrics['samples']==128 and metrics['input_sha256']==design['datasets'][dataset]['subset_sha256']
            questions=[json.loads(line) for line in (RUN/f'data/generation_{dataset}.jsonl').read_text().splitlines()]
            answers=[json.loads(line) for line in (folder/'answers_0000_end.jsonl').read_text().splitlines()]
            assert len(answers)==128 and [a['question_id'] for a in answers]==[q['question_id'] for q in questions]
            assert [a['reference'] for a in answers]==[q['reference'] for q in questions]
            row[key]=metrics['sacre_bleu' if dataset=='IWSLT' else 'rougeL_percent']
            assert math.isfinite(row[key])
            sources[str((folder/'metrics_0000_end.json').relative_to(RUN))]=sha(folder/'metrics_0000_end.json')
        row['generation_avg_score']=statistics.mean(row[key] for key in ['cnn_dm','dialogsum','iwslt'])
        rows.append(row)
    save('generation_pilot.csv',rows)
    differences=[{'control':row['policy'],**{key:rows[0][key]-row[key] for key in ['cnn_dm','dialogsum','iwslt','generation_avg_score']}}
        for row in rows[1:]]
    save('candidate_minus_controls.csv',differences)
    speeds=[]
    for local,pid in [('sq_int8_w4a16','fixed_sq_int8_w4a16'),('uniform_sq_int8','uniform_sq_int8'),('old_daso7_e001','graph_phase_dense_nvfp4_sq_int8')]:
        values=timing(pid)
        if values:speeds.append({'policy':local,'source_policy':pid,**values})
    if speeds:save('same_campaign_speed.csv',speeds)
    nll=[]
    baseline=read(RUN.parent/'019/results/quality/uniform_dense_bf16/center.json')
    for local,pid in [('sq_int8_w4a16','fixed_sq_int8_w4a16'),('uniform_sq_int8','uniform_sq_int8'),('old_daso7_e001','graph_phase_dense_nvfp4_sq_int8')]:
        source=RUN.parent/f'019/results/quality/{pid}/center.json'
        data=read(source)
        assert data['quality_sha256']==baseline['quality_sha256']
        delta=[a['avg_nll']-b['avg_nll'] for a,b in zip(data['blocks'],baseline['blocks'],strict=True)]
        assert len(delta)==64
        nll.append({'policy':local,'delta_nll':statistics.mean(delta),'paired_two_se':2*statistics.stdev(delta)/8,
            'source_sha256':sha(source),'mean_exceeds_epsilon_001':statistics.mean(delta)>.01})
    save('same_input_nll.csv',nll)
    lines=['# SQ-INT8 prefill / W4A16 decode：固定对照与完整生成结果','',
        '每任务128个固定seed抽样输入，四配置全部新跑相同greedy/256上限/6GiB KV/native decode Graph。'
        '前表为已评估benchmark的探索性子集，不作为完整任务结果、独立确认集或统计显著性结论；后文另列新组合与SQ对照的完整生成。'
        '新组合与Uniform SQ-INT8固定相同prefill；与旧DASO点比较时两个阶段均改变。','',
        '| Method | CNN/DM ROUGE-L | DialogSum ROUGE-L | IWSLT BLEU | Generation Avg score |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:lines.append('| '+row['method']+' | '+' | '.join(f'{row[k]:.3f}' for k in ['cnn_dm','dialogsum','iwslt','generation_avg_score'])+' |')
    lines += ['', '平均分为ROUGE-L%与SacreBLEU的混合算术摘要；原始分项全部保留，不作为统一准确率。','',
        '## 中心负载速度（复用019同一map）','',
        'B8/S2048/O64，GPU0/NUMA0。各轮自身BF16首尾锚点，3warmup+5timed；'
        '表中注明实际可用campaign数，cycle包含恢复。生成耗时不当作基准速度。','',
        '| Policy | Campaigns | E2E × | Cycle × | TTFT ms | TPOT ms | Restore ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in speeds:lines.append('| '+row['policy']+f' | {row["campaigns"]} | '+' | '.join(f'{row[k]:.3f}' for k in ['e2e_speedup','cycle_speedup','ttft_ms','tpot_ms','restore_ms'])+' |')
    transfer=read(RUN/'reference/same_candidate_workload_transfer.json')
    lines += ['', '## 六负载固定配置迁移：相对全SQ-INT8', '',
        '全部六负载均来自019三轮。新组合与全SQ-INT8固定map比较，未逐负载重新搜索；数值大于1表示新组合更快。', '',
        '| B | S | O | Relative E2E speedup | Relative cycle speedup |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for row in transfer:lines.append(f'| {row["batch"]} | {row["prompt"]} | {row["output"]} | {row["relative_e2e_speedup"]:.3f} | {row["relative_cycle_speedup"]:.3f} |')
    lines += ['', '## 独立负载NLL记录（019同64窗口）', '',
        '| Policy | Measured ΔNLL | Paired ±2SE | Mean exceeds .01 |',
        '| --- | ---: | ---: | --- |']
    for row in nll:lines.append(f'| {row["policy"]} | {row["delta_nll"]:.5f} | {row["paired_two_se"]:.5f} | {row["mean_exceeds_epsilon_001"]} |')
    lines += ['', '新组合是人工固定phase对照，不因生成子集改善而自动成为满足ε=.01的DASO选点。NLL约束与生成分数分别报告。']
    historical=read(RUN/'reference/historical_generation.json')
    equivalence=[]
    lines += ['', '## 历史完整生成参考（不同运行时，单列）', '',
        '002为旧eager，015为旧DASO的Graph；以下完整生成只作历史参考，与上述128条新pilot分开。', '',
        '| Policy | CNN/DM 1000 | DialogSum 1500 | IWSLT 333 |',
        '| --- | ---: | ---: | ---: |']
    for item in historical:
        pid=item['policy']
        lines.append('| '+pid+' | '+' | '.join(f'{item["datasets"][task]["score"]:.3f}' for task in ['cnn_dm_1000','dsum','IWSLT'])+' |')
        for task,meta in item['datasets'].items():
            path=Path(meta['answers_source'])
            assert sha(Path(meta['metric_source']))==meta['metric_sha256']
            assert sha(path)==meta['answers_sha256']
            old=[json.loads(line) for line in path.read_text().splitlines()]
            old=[old[i] for i in design['datasets'][task]['indices']]
            new=[json.loads(line) for line in (RUN/f'results/generation/{pid}/{task}/answers_0000_end.jsonl').read_text().splitlines()]
            assert [a['question_id'] for a in new]==[a['question_id'] for a in old]
            equivalence.append({'policy':pid,'dataset':task,'samples':128,
                'exact_generated_token_matches':sum(a['generated_token_ids']==b['generated_token_ids'] for a,b in zip(new,old,strict=True))})
    (RUN/'results/legacy_output_comparison.json').write_text(json.dumps(equivalence,indent=2)+'\n')
    full=[];full_equivalence=[]
    for item in historical:
        pid=item['policy']
        row={'policy':pid,'runtime_source':'027 new Graph' if pid in ['sq_int8_w4a16','uniform_sq_int8'] else f'{item["source_run"]} archived full generation'}
        for task,key in [('cnn_dm_1000','cnn_dm'),('dsum','dialogsum'),('IWSLT','iwslt')]:
            if pid in ['sq_int8_w4a16','uniform_sq_int8']:
                folder=RUN/f'full/results/generation/{pid}/{task}'
                metric=folder/'metrics_0000_end.json';data=read(metric)
                original=design['datasets'][task]
                assert data['input_sha256']==original['source_sha256']
                questions=[json.loads(line) for line in Path(original['source']).read_text().splitlines()]
                answers=[json.loads(line) for line in (folder/'answers_0000_end.jsonl').read_text().splitlines()]
                assert data['samples']==len(questions)==len(answers)
                assert [a['question_id'] for a in answers]==[q['question_id'] for q in questions]
                assert [a['reference'] for a in answers]==[q['reference'] for q in questions]
                old=[json.loads(line) for line in Path(item['datasets'][task]['answers_source']).read_text().splitlines()]
                full_equivalence.append({'policy':pid,'dataset':task,'samples':len(answers),
                    'exact_generated_token_matches':sum(a['generated_token_ids']==b['generated_token_ids'] for a,b in zip(answers,old,strict=True))})
                row[key]=data['sacre_bleu' if task=='IWSLT' else 'rougeL_percent']
                sources[str(metric.relative_to(RUN))]=sha(metric)
            else:row[key]=item['datasets'][task]['score']
            assert math.isfinite(row[key])
        row['generation_avg_score']=statistics.mean(row[k] for k in ['cnn_dm','dialogsum','iwslt'])
        full.append(row)
    save('full_generation_comparison.csv',full)
    full_differences=[{'control':row['policy'],**{key:full[0][key]-row[key] for key in ['cnn_dm','dialogsum','iwslt','generation_avg_score']}}
        for row in full[1:]]
    save('full_candidate_minus_controls.csv',full_differences)
    outcomes={row['control']:{'all_three_full_task_scores_higher':all(row[k]>0 for k in ['cnn_dm','dialogsum','iwslt'])}
        for row in full_differences}
    (RUN/'results/observed_full_task_comparisons.json').write_text(json.dumps({
        'comparisons':outcomes,'scope':'Observed fixed-benchmark scores, not statistical significance or a new quality-constraint guarantee.'},indent=2)+'\n')
    (RUN/'results/full_legacy_output_comparison.json').write_text(json.dumps(full_equivalence,indent=2)+'\n')
    lines += ['', '## 新组合完整三任务与归档完整对照', '',
        '新组合与全SQ-INT8对照的1000/1500/333均在027 Graph新测。其余两行完整分数来自明确标注的历史归档；'
        '四配置另有上面的同128条新Graph pilot及逐token历史匹配计数，不把历史完整对照称为本轮全量重测。', '',
        '| Policy | Runtime source | CNN/DM1000 | DialogSum1500 | IWSLT333 | Generation Avg score |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in full:lines.append('| '+row['policy']+' | '+row['runtime_source']+' | '+' | '.join(f'{row[k]:.3f}' for k in ['cnn_dm','dialogsum','iwslt','generation_avg_score'])+' |')
    lines += ['', '普通INT8的旧015完整主表E2E为1.645×，属另一campaign参考，未冒充上述同轮测量。'
        '本pilot的四项选择题不重跑：新配置prefill等价于Uniform SQ-INT8，原完整MC结果另有来源。','',
        '[生成CSV](results/generation_pilot.csv) · [相对各对照的差值](results/candidate_minus_controls.csv) · '
        '[冻结设计与抽样索引](configs/design.json) · [来源速度019](../019/README.md)','']
    (RUN/'REPORT.md').write_text('\n'.join(lines))
    for rid,digest in design['protected_manifests_sha256'].items():assert sha(RUN.parent/rid/'results/manifest.json')==digest
    (RUN/'results/final_audit.json').write_text(json.dumps({'complete':True,'verified':True,'policies':4,'generation_tasks':12,
        'full_candidate_generation_tasks':3,'full_candidate_samples':[1000,1500,333],
        'fresh_full_control_generation_tasks':3,
        'samples_per_task':128,'design_sha256':sha(RUN/'configs/design.json'),'metric_sources_sha256':sources,
        'generation_csv_sha256':sha(RUN/'results/generation_pilot.csv'),
        'full_generation_csv_sha256':sha(RUN/'results/full_generation_comparison.csv'),
        'historical_reference_sha256':sha(RUN/'reference/historical_generation.json'),
        'full_control_amendment_sha256':sha(RUN/'configs/full_control_amendment.json'),
        'workload_transfer_sha256':sha(RUN/'reference/same_candidate_workload_transfer.json'),
        'source019_manifest_sha256':sha(RUN.parent/'019/results/manifest.json'),
        'code_sha256':{p.name:sha(p) for p in (RUN/'code').glob('*.py')},
        'scope':'Four-policy exploratory subset pilot plus full generation of the fixed candidate and SQ-INT8 control. Other archived full controls explicitly sourced; old main tables unchanged. Speed reused with explicit campaign coverage.'},indent=2)+'\n')
    print('Completed fixed four-policy generation pilot',differences,flush=True)


if __name__=='__main__':main()
