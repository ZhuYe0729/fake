"""Compare exact baseline replays with identical original input/batch subsets."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

RUN=Path(__file__).resolve().parents[1]
DEST=RUN.parents[1]/'responses/reviewer2/llama_baseline_replay_run037'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lines=lambda p:[json.loads(v)for v in p.read_text().splitlines()]


def save(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def score(rows,task):
    preds=[r['prediction'].strip()or' 'for r in rows];refs=[r['reference']for r in rows]
    if task=='IWSLT':return corpus_bleu(preds,[refs]).score
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    return 100*statistics.mean(scorer.score(r,p)['rougeL'].fmeasure for r,p in zip(refs,preds))


def main():
    nll,gen,differences,sources=[],[],[],{}
    for job in read(RUN/'configs/jobs.json'):
        identity=RUN/f'results/source_identity/{job["source_run"]}_{job["policy"]}.json'
        for path,digest in read(identity)['files_sha256'].items():assert sha(Path(path))==digest,path
        sources[str(identity)]=sha(identity)
        old=RUN.parent/job['old_run'];new=RUN/'models'/job['model_id'];pid=job['policy']
        before=old/f'results/confirmation/{pid}/prefill_decode.json'
        after=new/f'results/confirmation/{pid}/prefill_decode.json'
        a,b=read(before),read(after)
        assert a['data_sha256']==b['data_sha256']and len(a['blocks'])==len(b['blocks'])==64
        ds=[y['avg_nll']-x['avg_nll']for x,y in zip(a['blocks'],b['blocks'])]
        nll.append({'model':job['model_id'],'policy':pid,'blocks':64,'mean_difference':statistics.mean(ds),
            'max_abs_block_difference':max(abs(x)for x in ds),
            'within_original_cache_gate':abs(statistics.mean(ds))<=1e-4 and max(abs(x)for x in ds)<=1e-3})
        sources[str(before)]=sha(before);sources[str(after)]=sha(after)
        for task in ['cnn_dm_1000','dsum','IWSLT']:
            before=old/f'results/generation/{pid}/{task}/answers_0000_end.jsonl'
            after=new/f'results/generation/{pid}/{task}/answers_0000_128.jsonl'
            a,b=lines(before)[:128],lines(after)
            assert len(a)==len(b)==128
            assert all(x['question_id']==y['question_id']and x['reference']==y['reference']and x['prompt_tokens']==y['prompt_tokens']for x,y in zip(a,b))
            equal=sum(x['generated_token_ids']==y['generated_token_ids']for x,y in zip(a,b))
            text_equal=sum(x['prediction']==y['prediction']for x,y in zip(a,b))
            old_score,new_score=score(a,task),score(b,task)
            gen.append({'model':job['model_id'],'policy':pid,'dataset':task,'samples':128,
                'exact_token_sequences':equal,'exact_prediction_strings':text_equal,
                'old_same_subset_score':old_score,'new_same_subset_score':new_score,'score_difference':new_score-old_score})
            for x,y in zip(a,b):
                if x['generated_token_ids']!=y['generated_token_ids']:
                    left,right=x['generated_token_ids'],y['generated_token_ids']
                    first=next((i for i,(p,q)in enumerate(zip(left,right))if p!=q),min(len(left),len(right)))
                    differences.append({'model':job['model_id'],'policy':pid,'dataset':task,
                        'question_id':x['question_id'],'first_different_token':first,'old_tokens':left,'new_tokens':right})
            sources[str(before)]=sha(before);sources[str(after)]=sha(after)
            assert sha(old/f'data/generation_{task}.jsonl')==sha(new/f'data/generation_{task}.jsonl')
    assert len(nll)==6 and len(gen)==18
    save(RUN/'results/nll_comparison.csv',nll);save(RUN/'results/generation_comparison.csv',gen)
    (RUN/'results/token_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    DEST.mkdir(parents=True,exist_ok=True)
    for name in ['nll_comparison.csv','generation_comparison.csv','token_differences.json']:
        (DEST/name).write_bytes((RUN/'results'/name).read_bytes())
    text=['# Llama8bit基线同输入重放（R2/R5）','',
        '两模型plain INT8、SQ-INT8、FP8；沿034/035使用相同数值代码和精确export缓存，仅隔离输出目录。NLL重放原64窗口，生成重放每任务最前128条，保持原batch8/4与分组、BOS/PMPD和256token/EOS。这里比较相同128子集，不用子集分数替代完整benchmark。','',
        f'共{sum(r["samples"]for r in gen)}条生成重放，逐token完全一致{sum(r["exact_token_sequences"]for r in gen)}条；全部NLL落在原cache gate内：{all(r["within_original_cache_gate"]for r in nll)}。完全一致率和数值容差只是复用可靠性证据，不是新泛化或新性能结论。','',
        '| 模型 | 方法 | NLL均值差 | 最大窗口差 | 原容差通过 |','| --- | --- | ---: | ---: | --- |']
    for r in nll:text.append(f'| {r["model"]} | {r["policy"]} | {r["mean_difference"]:.8f} | {r["max_abs_block_difference"]:.8f} | {r["within_original_cache_gate"]} |')
    text+=['','| 模型 | 方法 | 任务 | token完全一致/128 | 原128子集分数 | 新128子集分数 | 差值 |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in gen:text.append(f'| {r["model"]} | {r["policy"]} | {r["dataset"]} | {r["exact_token_sequences"]} | {r["old_same_subset_score"]:.5f} | {r["new_same_subset_score"]:.5f} | {r["score_difference"]:.5f} |')
    text+=['','任何差异均保留，不能据此选择更弱的基线。必要的完整基线复测需在定位输入/权重/运行时差异后独立开展。原017/018结果与各追加主表不改。','',
        '[NLL CSV](nll_comparison.csv) · [生成CSV](generation_comparison.csv) · [逐token差异](token_differences.json) · [原始代码/输出](../../../runs/037/README.md)','']
    (DEST/'README.md').write_text('\n'.join(text))
    meta={'complete':True,'verified_comparison':True,'nll_groups':6,'generation_groups':18,
        'all_token_sequences_exact':all(r['exact_token_sequences']==128 for r in gen),
        'all_nll_within_original_gate':all(r['within_original_cache_gate']for r in nll),'sources_sha256':sources,
        'code_sha256':{p.name:sha(p)for p in(RUN/'code').glob('*.py')},
        'retained_failures':[p.name for p in(RUN/'results').glob('failure*.json')]}
    meta['delivery_sha256']={p.name:sha(p)for p in DEST.iterdir()if p.is_file()and p.name!='provenance.json'}
    (DEST/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    (RUN/'results/final_audit.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Baseline replay comparisons complete:',meta['all_token_sequences_exact'],meta['all_nll_within_original_gate'],flush=True)


if __name__=='__main__':main()
