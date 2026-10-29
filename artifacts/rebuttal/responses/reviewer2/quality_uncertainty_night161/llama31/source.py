"""Paired descriptive intervals on observed generation benchmarks, CPU only."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
TASKS=[('cnn_dm_1000','rougeL_percent',1000),('dsum','rougeL_percent',1500),('IWSLT','sacre_bleu',333)]
N=5000


def resolve(rid,pid,task):
    root=RUNS/rid;path=root/f'results/generation/{pid}/{task}/metrics_0000_end.json'
    if path.exists():return path
    design=root/'configs/sweep_generation_jobs.json'
    if not design.exists():design=root/'configs/generation_jobs.json'
    item=next(r for r in read(design)['reused']if(r['policy'],r['dataset'])==(pid,task))
    path=Path(item['source']);assert sha(path)==item['sha256'];return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen3','llama31','llama2'],required=True);args=p.parse_args()
    os.sched_setaffinity(0,set(range(48,56)))
    if args.model=='qwen3':
        bases=[('INT8','032','uniform_int8'),('SQ-INT8','032','uniform_sq_int8')]
        points=[('w1_q.01','032','w1_e0.01'),('w32_q.03','033','w32_e0.03')]
    else:
        replay=read(RUNS/'037/results/final_audit.json')
        assert replay['all_token_sequences_exact']and replay['all_nll_within_original_gate'],'Resolve baseline replay differences first'
        rid='034'if args.model=='llama31'else'035'
        bases=[('INT8',rid,'uniform_int8'),('SQ-INT8',rid,'uniform_sq_int8'),('FP8',rid,'uniform_fp8')]
        points=[(x['label'],rid,x['policy'])for x in read(RUNS/rid/'configs/sweep_points.json')]
        if rid=='034':points += [(x['label'],'036',x['policy'])for x in read(RUNS/'036/configs/sweep_points.json')]
    methods=bases+points;sources={};scores={};draws={}
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    for ti,(task,key,count)in enumerate(TASKS):
        indices=np.random.default_rng(1632026+ti).integers(0,count,size=(N,count))
        expected=None
        for label,rid,pid in methods:
            metric=resolve(rid,pid,task);answers=metric.with_name('answers_0000_end.jsonl')
            value=read(metric);rows=[json.loads(v)for v in answers.read_text().splitlines()]
            assert len(rows)==value['samples']==count
            ids=[(r['question_id'],r['reference'])for r in rows]
            if expected is None:expected=ids
            assert expected==ids,(label,task)
            refs=[r['reference']for r in rows];preds=[r['prediction'].strip()or' 'for r in rows]
            if task=='IWSLT':
                bleu=BLEU(tokenize='13a')
                stats=np.asarray(bleu._extract_corpus_statistics(preds,[refs]),dtype=np.int64)
                actual=bleu._compute_score_from_stats(stats.sum(0).tolist()).score
                sampled=stats[indices].sum(1)
                samples=np.asarray([bleu._compute_score_from_stats(s.tolist()).score for s in sampled])
            else:
                stats=100*np.asarray([scorer.score(ref,pred)['rougeL'].fmeasure for ref,pred in zip(refs,preds)])
                actual=float(stats.mean());samples=stats[indices].mean(1)
            assert abs(actual-value[key])<1e-8,(task,label,actual,value[key])
            scores[label,task]=actual;draws[label,task]=samples
            sources[str(metric)]=sha(metric);sources[str(answers)]=sha(answers)
            print(args.model,task,label,'score',actual,'bootstrap complete',flush=True)
    output=[]
    for label,_,_ in points:
        for baseline,_,_ in bases:
            for task in [t for t,_,_ in TASKS]+['Gen Avg']:
                if task=='Gen Avg':
                    delta=statistics.mean(scores[label,t]-scores[baseline,t]for t,_,_ in TASKS)
                    values=sum(draws[label,t]-draws[baseline,t]for t,_,_ in TASKS)/3
                else:delta=scores[label,task]-scores[baseline,task];values=draws[label,task]-draws[baseline,task]
                low,high=np.quantile(values,[.025,.975])
                output.append({'model':args.model,'point':label,'baseline':baseline,'metric':task,
                    'observed_difference':delta,'percentile_low95':float(low),'percentile_high95':float(high),
                    'resamples':N,'seed_base':1632026,'selection_adjusted':False})
    dest=ROOT.parent/'responses/reviewer2/quality_uncertainty_night161'/args.model
    dest.mkdir(parents=True,exist_ok=True)
    with(dest/'paired_intervals.csv').open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    selected=[r for r in output if r['metric']=='Gen Avg'and any(s in r['point']for s in ['w1_q0.01','w1_q.01','w32_','w16_'])]
    fig,ax=plt.subplots(figsize=(8,max(3,len(selected)*.38)))
    for i,r in enumerate(selected):
        x=r['observed_difference'];lo=r['percentile_low95'];hi=r['percentile_high95']
        ax.hlines(i,lo,hi,color='#2469a0')
        ax.plot(x,i,'o',color='#2469a0')
    ax.set_yticks(range(len(selected)),[r['point']+' vs '+r['baseline']for r in selected])
    ax.axvline(0,color='gray',linestyle='--');ax.set_xlabel('Gen Avg difference; descriptive paired bootstrap95% interval')
    ax.set_title(args.model+': selected exploratory points (all comparisons inCSV)');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(dest/'macro_intervals.png',dpi=200,bbox_inches='tight');fig.savefig(dest/'macro_intervals.pdf',bbox_inches='tight');plt.close(fig)
    text=[f'# {args.model}生成质量的配对bootstrap区间','',
        '配置已经在这些benchmark上探索，以下为固定配置条件下的描述性paired bootstrap，未校正参数选择和多重比较，不能称为新的盲测或确证性显著优胜。原ROUGE-L%/corpus BLEU和完整样本数保持，BLEU按重采样corpus统计重算。', '',
        '5000次重采样，seed=1632026+数据集序号；每任务方法间使用相同索引。三任务独立抽样后差值等权形成原Gen Avg摘要，仍不是统一准确率或NLL预算。图聚焦原主表重点点，CSV保留全部比较。', '',
        '| 点 | 参考 | 指标 | 实测差 | 2.5% | 97.5% |','| --- | --- | --- | ---: | ---: | ---: |']
    for r in output:text.append(f'| {r["point"]} | {r["baseline"]} | {r["metric"]} | {r["observed_difference"]:.4f} | {r["percentile_low95"]:.4f} | {r["percentile_high95"]:.4f} |')
    text+=['','[CSV](paired_intervals.csv) · [重点点图](macro_intervals.png) · [PDF](macro_intervals.pdf) · [来源](provenance.json)','']
    (dest/'README.md').write_text('\n'.join(text))
    (dest/'source.py').write_bytes(Path(__file__).read_bytes())
    meta={'complete':True,'comparisons':len(output),'resamples':N,'seed_base':1632026,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'source_sha256':sources,'code_sha256':sha(Path(__file__)),
        'scope':'Observed benchmark, fixed-configuration descriptive bootstrap, not selection-adjusted or newblind test.'}
    meta['delivery_sha256']={p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'}
    (dest/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Bootstrap completed',args.model,len(output),'comparisons',dest,flush=True)


if __name__=='__main__':
    with(ROOT.parent/'night151/measurement.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_SH)
        main()
