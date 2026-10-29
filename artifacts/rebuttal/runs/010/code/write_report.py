"""Audit preregistered full task coverage and summarize unchanged metrics."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    catalog=read(RUN/'configs/task_catalog.json');rules=read(RUN/'configs/task_rules.json')
    assert digest(RUN/'configs/task_rules.json')==catalog['rules_sha256']
    assert digest(RUN.parent/'005/configs/selected_policies.json')==catalog['source_selection_sha256']
    for job in catalog['jobs']:assert read(RUN/f'results/complete/{job["id"]}.json')['complete']
    for pid,info in catalog['policies'].items():assert digest(RUN/f'configs/policies/{pid}.json')==info['policy_sha256']
    inherited=read(RUN/'results/inherited_tasks.json')['records']
    for record in inherited:
        target=RUN/record['target'];assert target.is_symlink()
        assert target.resolve()==Path(record['source']) and digest(target)==record['sha256']
    assert len(inherited)==18
    labels={'reference_bf16':'BF16','reference_sparsegpt':'SparseGPT 2:4',
        'pruning_magnitude':'Magnitude 2:4','pruning_wanda':'Wanda 2:4',
        'verified_c63754e060c4':'Verified prefill, epsilon=.01 (SQ INT8)',
        'verified_d1eeaeaf84b5':'Verified prefill, epsilon=.04 (heterogeneous)',
        'verified_7aafcc63be76':'Verified P+D, epsilon=.01 (BF16)',
        'verified_9e60153b6827':'Verified P+D, epsilon=.04 (heterogeneous)'}
    sizes={'wikitext':62,'winogrande':1267,'arc_easy':2376,'arc_challenge':1172,'mmlu':14042}
    metrics={'wikitext':'word_perplexity,none','winogrande':'acc,none',
        'arc_easy':'acc_norm,none','arc_challenge':'acc_norm,none','mmlu':'acc,none'}
    tasks=[]
    for path in sorted((RUN/'results/tasks').glob('*/*/full/result.json')):
        pid=path.relative_to(RUN/'results/tasks').parts[0];task=path.parent.parent.name;r=read(path)
        counts=r['raw_lm_eval']['n-samples'];effective=sum(x['effective'] for x in counts.values())
        assert r['limit'] is None and r['num_fewshot']==0
        assert all(x['effective']==x['original'] for x in counts.values()) and effective==sizes[task]
        assert len(counts)==(57 if task=='mmlu' else 1)
        assert r['batch_size']==(1 if task=='wikitext' else 4)
        value=r['metrics'][metrics[task]]*(1 if task=='wikitext' else 100)
        tasks.append({'policy':pid,'label':labels[pid],'task':task,'metric':metrics[task],
            'value':value,'units':'perplexity' if task=='wikitext' else 'percent',
            'samples':effective,'subtasks':len(counts),'batch_size':r['batch_size'],
            'elapsed_seconds':r['elapsed_seconds'],'inherited':path.is_symlink(),
            'source':str(path.relative_to(RUN)),'source_sha256':digest(path)})
    assert len(tasks)==30
    generation=[]
    for path in sorted((RUN/'results/generation').glob('*/*/metrics_0000_end.json')):
        pid=path.relative_to(RUN/'results/generation').parts[0];task=path.parent.name;r=read(path)
        reference=read(RUN/f'results/generation/verified_7aafcc63be76/{task}/metrics_0000_end.json')
        assert r['samples']=={'cnn_dm_1000':1000,'dsum':1500,'IWSLT':333}[task]
        assert r['input_sha256']==reference['input_sha256'] and r['empty_predictions']==0
        assert r['max_new_tokens']==256 and r['kv_cache_memory_bytes']==6*1024**3
        for metric in (['rougeL_percent','sacre_bleu'] if task=='IWSLT' else ['rougeL_percent']):
            generation.append({'policy':pid,'label':labels[pid],'task':task,'metric':metric,'value':r[metric],
                'delta_to_bf16':r[metric]-reference[metric],'samples':r['samples'],
                'inherited':path.is_symlink(),'source':str(path.relative_to(RUN)),'source_sha256':digest(path)})
    assert len(generation)==8
    for name,rows in [('likelihood_summary',tasks),('generation_summary',generation)]:
        with (RUN/f'results/{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    order=['reference_bf16','verified_c63754e060c4','verified_d1eeaeaf84b5','reference_sparsegpt','pruning_wanda','pruning_magnitude']
    matrix={(r['policy'],r['task']):r['value'] for r in tasks}
    lines=['# Run010：预先选定策略的完整下游任务','',
        '策略取自run005在最终test之前冻结的验证选择，预先固定epsilon=.01/.04的prefill与P+D代表点；没有根据下游得分换策略或删任务。Magnitude/Wanda固定补齐五个likelihood任务。','',
        '相同method-map且原canonical权重/scales来源完全相同的结果继承002，并逐文件校验SHA；新Wanda/magnitude权重全部重新评测，不能因稀疏格式相同而继承SparseGPT结果。18份继承结果、18份新结果，共36份完整任务结果。','',
        '## 完整likelihood任务','',
        '采用原002协议：zero-shot、max_model_len=2048、3GiB KV；WikiText滚动PPL为batch1，其余batch4。WikiText为完整原测试集的62篇文档，Winogrande1267、ARC-E2376、ARC-C1172、MMLU57科共14042题；全部effective=original，limit=None。ARC使用normalized accuracy，其余分类使用accuracy。','',
        '| 策略 | WikiText PPL ↓ | Winogrande % | ARC-E norm % | ARC-C norm % | MMLU % |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for pid in order:lines.append('| '+labels[pid]+' | '+' | '.join(f'{matrix[pid,t]:.3f}' for t in sizes)+' |')
    lines+=['','## 完整生成任务','',
        'CNN/DM1000、XSum原dsum子集1500、IWSLT333，均沿用002冻结输入、native Qwen3 non-thinking chat template、batch8、max_new_tokens=256、6GiB KV、允许EOS。生成长度随策略改变，因此generation_seconds不用于固定工作量加速比。epsilon=.01的验证选择回退BF16，epsilon=.04为异构策略。','',
        '| 策略 | 任务 | 指标 | 得分 | 对BF16变化（百分点） | 样本数 |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in generation:lines.append(f'| {r["label"]} | {r["task"]} | {r["metric"]} | {r["value"]:.4f} | {r["delta_to_bf16"]:+.4f} | {r["samples"]} |')
    lines+=['','## 解释与边界','',
        '- 两个prefill代表点在005的中心B8/S2048/O1负载分别为1.953×和2.213×；这里likelihood任务的batch/长度不同，不能把这两个速度称为完整MMLU任务的加速比。',
        '- 相同2:4 kernel下，简单剪枝方法具有相近速度而不同下游精度；Wanda/SparseGPT的完整任务显著好于magnitude，但均有明显质量下降。',
        '- epsilon=.04的P+D点在005固定B8/S2048/O64计时为1.358×，其三个完整生成任务ROUGE-L/BLEU变化另见表；不可把固定长度E2E速度与可变长度生成任务混为一个实验。',
        '- WikiText NLL质量预算不保证跨任务准确率。prefill异构点MMLU比BF16下降约1.77个百分点；完整表保留该代价。epsilon=.01的SQ点下降约0.26个百分点，并非异构配置在所有任务上都更好。',
        '- 本轮代表性确认不等同于对所有003负载、所有候选或HAWQ变体都做全任务；原001/002完整任务应共同引用。未运行完整30B模型。','',
        '![完整任务对照](results/figures/full_likelihood.png)','',
        '[likelihood完整CSV](results/likelihood_summary.csv) · [生成完整CSV](results/generation_summary.csv) · [预注册规则](configs/task_rules.json) · [选择/继承来源](configs/task_catalog.json) · [审计](results/task_audit.json)。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    directory=RUN/'results/figures';directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    short=['BF16','Verified .01\nSQ INT8','Verified .04\nheterogeneous','SparseGPT','Wanda','Magnitude']
    for ax,task in zip(axes,['mmlu','wikitext'],strict=True):
        ax.bar(range(len(order)),[matrix[p,task] for p in order])
        ax.set_xticks(range(len(order)),short,rotation=40,ha='right',fontsize=9)
        ax.set_ylabel('Accuracy (%)' if task=='mmlu' else 'Word perplexity (log scale)')
        if task=='wikitext':ax.set_yscale('log')
        ax.set_title('MMLU: all 14,042 questions' if task=='mmlu' else 'WikiText: full rolling evaluation')
        ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(directory/f'full_likelihood.{ext}',dpi=180)
    plt.close(fig)
    audit={'passed':True,'new_policy_jobs':4,'likelihood_result_files':30,'generation_result_files':6,
        'inherited_results_verified':18,'new_results':18,'mmlu_subjects_per_policy':57,
        'mmlu_questions_per_policy':14042,'full_sample_counts':sizes,
        'generation_samples_per_policy':{'cnn_dm_1000':1000,'dsum':1500,'IWSLT':333},
        'source_selection_sha256':catalog['source_selection_sha256'],'rules_sha256':catalog['rules_sha256'],
        'all_preregistered_jobs_complete':True,'selection_and_inherited_hashes_verified':True}
    for name in ['task_audit','report_complete']:(RUN/f'results/{name}.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
