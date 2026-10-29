"""Publish complete original-protocol task scores; keep pending cells explicit."""
from pathlib import Path
import csv
import hashlib
import html
import json
import math

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/050'
OUT=BASE/'responses/reviewer2/qwen_original_quality_run050'
OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
DESIGN=read(RUN/'configs/design.json')
markers={j['id']:RUN/f"results/completed/{j['id']}.json" for j in DESIGN['jobs'] if (RUN/f"results/completed/{j['id']}.json").exists()}
cache=read(OUT/'cell_cache.json') if (OUT/'cell_cache.json').exists() else {}
cells={};sources={};completed=0
def source(path):sources[str(path.relative_to(BASE))]=sha(path);return path
def f(v):return '待测' if v is None else f'{float(v):.2f}'

for job in DESIGN['jobs']:
 marker=markers.get(job['id'])
 if marker is None:continue
 completed+=1;entry=read(source(marker))
 if job['kind']!='multiple_choice':continue
 key=job['id'];signature=entry['result_sha256']
 if key not in cache or cache[key]['signature']!=signature:
  path=RUN/entry['result'];assert sha(path)==signature;data=read(path)
  assert data['job']==job and data['complete'] and data['limit'] is None and data['num_fewshot']==0
  assert sum(r['effective'] for r in data['raw_lm_eval']['n-samples'].values())==job['samples']
  selected=data['raw_lm_eval']['results'].get(job['task']) or data['raw_lm_eval'].get('groups',{}).get(job['task'])
  value=float(selected[job['metric']]);assert value==float(data['metric_value']) and 0<=value<=1
  cache[key]={'signature':signature,'value':100*value,'sources':{str(path.relative_to(BASE)):signature},'samples':job['samples']}
 cells[job['batch'],job['policy'],job['task']]=cache[key]['value'];sources.update(cache[key]['sources'])

scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
for batch in DESIGN['batches']:
 for pid in ['bf16','pd_hawq_bal','pd_hawq_fast','pd_daso_bal','pd_daso_fast']:
  for task,count in DESIGN['generation_tasks'].items():
   jobs=[j for j in DESIGN['jobs'] if j['kind']=='generation' and j['batch']==batch and j['policy']==pid and j['task']==task]
   jobs.sort(key=lambda j:j['begin'])
   if not all(j['id'] in markers for j in jobs):continue
   entries=[read(markers[j['id']]) for j in jobs];signature=hashlib.sha256(json.dumps([m['result_sha256'] for m in entries]).encode()).hexdigest()
   key=f'generation_b{batch}_{pid}_{task}'
   if key not in cache or cache[key]['signature']!=signature:
    answers=[];files={};input_file=RUN/f'data/generation_{task}.jsonl';questions=[json.loads(x) for x in input_file.read_text().splitlines()]
    assert len(questions)==count and sha(input_file)==DESIGN['input_sha256'][input_file.name]
    for job,marker in zip(jobs,entries,strict=True):
     path=RUN/marker['result'];assert sha(path)==marker['result_sha256'];data=read(path)
     assert data['complete'] and data['job']==job and data['max_new_tokens']==256 and data['input_sha256']==sha(input_file)
     raw=RUN/data['answers'];values=[json.loads(x) for x in raw.read_text().splitlines()]
     expected=questions[job['begin']:job['end']]
     assert len(values)==len(expected)
     assert all(a['question_id']==b['question_id'] and a['reference']==b['reference'] for a,b in zip(values,expected,strict=True))
     answers.extend(values);files[str(path.relative_to(BASE))]=sha(path);files[str(raw.relative_to(BASE))]=sha(raw)
    assert len(answers)==count
    predictions=[a['prediction'].strip() or ' ' for a in answers];references=[a['reference'] for a in answers]
    value=corpus_bleu(predictions,[references]).score if task=='IWSLT' else 100*sum(scorer.score(r,p)['rougeL'].fmeasure for r,p in zip(references,predictions,strict=True))/count
    assert math.isfinite(value)
    cache[key]={'signature':signature,'value':value,'samples':count,'sources':files,
     'empty_predictions':sum(not a['prediction'].strip() for a in answers)}
   cells[batch,pid,task]=cache[key]['value'];sources.update(cache[key]['sources'])

pp={r['method']:r['policy'] for r in DESIGN['rows']['prefill_only']}
pd={r['method']:r['policy'] for r in DESIGN['rows']['prefill_decode']}
rows=[]
for batch in DESIGN['batches']:
 for method in pp:
  row={'batch':batch,'method':method}
  for column,task in [('arc_c','arc_challenge'),('arc_e','arc_easy'),('winogrande','winogrande'),('mmlu','mmlu')]:row[column]=cells.get((batch,pp[method],task))
  values=[row[k] for k in ['arc_c','arc_e','winogrande','mmlu']];row['prefill_avg']=sum(values)/4 if all(v is not None for v in values) else None
  for column,task in [('cnn_dm','cnn_dm_1000'),('dialogsum','dsum'),('iwslt','IWSLT')]:row[column]=cells.get((batch,pd[method],task))
  values=[row[k] for k in ['cnn_dm','dialogsum','iwslt']];row['pd_avg']=sum(values)/3 if all(v is not None for v in values) else None
  rows.append(row)
with (OUT/'quality_table.csv').open('w') as handle:
 writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
headers=[('method','方法'),('arc_c','ARC-C'),('arc_e','ARC-E'),('winogrande','Wino'),('mmlu','MMLU'),('prefill_avg','Prefill Avg'),('cnn_dm','CNN/DM'),('dialogsum','DialogSum'),('iwslt','IWSLT'),('pd_avg','P+D Avg')]
complete=completed==len(DESIGN['jobs']) and all(r['prefill_avg'] is not None and r['pd_avg'] is not None for r in rows)
lines=['# Qwen原协议下游精度（R2 / R3）','',
 f"**状态：{'全部完成' if complete else '运行中'}，{completed}/{len(DESIGN['jobs'])}个任务/分块完成。** 本页只展示精度。原主表完整任务、原prompt、生成上限256；分别测试B=8与B=4。",'',
 '这些是原协议任务分数，不是固定S/O下的任务分数。同一策略、batch的分数可供S=2048/4096、测速输出64/128行共用；不将共用值标作多次独立重测。另见[逐B/S/O NLL](../qwen_nearby_dual_scenario_run049/README.md)。','',
 '[分组HTML](table.html) · [全部CSV](quality_table.csv) · [运行状态](../../../runs/050/results/worker.json) · [任务设计](../../../runs/050/configs/design.json)','']
body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen原协议精度</title><style>body{font:15px system-ui;max-width:1400px;margin:25px auto;padding:0 16px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;text-align:right;border:1px solid #dce3e9;white-space:nowrap}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.daso{background:#eaf5fc}.hawq{background:#fff9ed}p{line-height:1.6}</style><h1>Qwen原协议下游精度</h1>',f'<p>进度{completed}/{len(DESIGN["jobs"])}。原prompt、完整样本、最大生成256；仅精度，速度留待夜间重测。S/O测速标签不构成新的质量任务数据。</p>']
for batch in DESIGN['batches']:
 subset=[r for r in rows if r['batch']==batch]
 lines +=[f'## Batch={batch}','','| '+' | '.join(label for _,label in headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
 body.append(f'<h2>Batch={batch}</h2><div class="scroll"><table><thead><tr><th rowspan="2">方法</th><th colspan="5">Prefill-only，准确率%</th><th colspan="4">P+D，原协议任务Score</th></tr><tr>'+''.join('<th>'+label+'</th>' for _,label in headers[1:])+'</tr></thead><tbody>')
 for row in subset:
  values=[row[k] if k=='method' else f(row[k]) for k,_ in headers]
  lines.append('| '+' | '.join(values)+' |')
  cls='daso' if row['method'].startswith('DASO') else 'hawq' if row['method'].startswith('HAWQ') else ''
  body.append('<tr class="'+cls+'">'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
 lines.append('');body.append('</tbody></table></div>')
lines +=['## 协议','',
 '- 原主表MC使用B=4、生成使用B=8；本轮将两类任务都分别跑B=8/4，其余任务、prompt及生成上限沿用原设置。不同batch的分数单独保留。',
 '- ARC-C acc_norm；ARC-E/Wino/MMLU acc，全部0-shot。样本量依次1172/2376/1267/14042，MC上下文2048，原生lm-eval/vLLM Prefill似然评测。',
 '- CNN/DM 1000、DialogSum 1500用原ROUGE-L；IWSLT 333用完整语料SacreBLEU。沿用原Qwen无thinking prompt和greedy，保留原输入token，最多生成256；原输入限制3840，本地数据实际最长2355。',
 '- 两场景Avg各为分项等权平均，沿用原主表。未完成分项不计入Avg。Prefill共同速度端点为同map，复用一次测量；BF16只是参照。',
 '- 沿用[主表已说明的配置与后端](../qwen3_main_table/README.md)，不按本轮精度重新选点。Prefill DASO均衡点选中全SQ-INT8；各方法、场景的具体map与组成见任务设计。',
 '- 生成分块边界按8对齐，保留原batch分组。最终BLEU由完整预测语料重算，不平均分块BLEU。失败/部分输出保留，不用于最终得分。',
 '- GPU空闲时逐卡加锁并行，启动核对CUDA/NVML UUID。运行耗时仅用于进度，不作为速度；旧主表和049结果不改写。','']
body.append('<p><a href="README.md">Markdown与协议</a> · <a href="quality_table.csv">CSV</a></p></html>')
(OUT/'README.md').write_text('\n'.join(lines));(OUT/'table.html').write_text(''.join(body))
(OUT/'cell_cache.json').write_text(json.dumps(cache,ensure_ascii=False,indent=2)+'\n')
audit={'verified':True,'complete':complete,'completed_jobs':completed,'planned_jobs':len(DESIGN['jobs']),'display_rows':10,
 'complete_prefill_rows':sum(r['prefill_avg'] is not None for r in rows),'complete_generation_rows':sum(r['pd_avg'] is not None for r in rows),
 'source_sha256':sources,'generation_max_new_tokens':256,'quality_inputs_shared_across_speed_S_O':True,'speed_measurements':0}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
