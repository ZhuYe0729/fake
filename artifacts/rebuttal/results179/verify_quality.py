"""Recompute original-protocol scores and verify publication provenance."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/050'
OUT=BASE/'responses/reviewer2/qwen_original_quality_run050'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
design=read(RUN/'configs/design.json');audit=read(OUT/'audit.json')
assert design['generation_max_new_tokens']==256 and not design['performance_measurements_enabled']
for name,digest in read(RUN/'configs/protocol_sha256.json').items():assert sha(RUN/'code'/name)==digest,name
for group in ['source_sha256','protected_sha256']:
 for path,digest in design[group].items():assert sha(Path(path))==digest,path
for path,digest in read(RUN/'reference/likelihood_inputs.json').items():assert sha(Path(path))==digest,path
for name,digest in design['input_sha256'].items():assert sha(RUN/'data'/name)==digest,name
for path,digest in audit['source_sha256'].items():assert sha(BASE/path)==digest,path
for pid,entry in read(RUN/'reference/checkpoint_reuse.json').items():
 source=Path(entry['source']).stat();target=Path(entry['target']).stat()
 assert source.st_ino==target.st_ino==entry['inode'] and source.st_dev==target.st_dev==entry['device']
 assert source.st_size==target.st_size==entry['bytes']
 config=Path(entry['target']).parent/'config.json';assert sha(config)==entry['config_sha256']
 mapping=read(config)['quantization_config']['method_map']
 assert mapping==read(RUN/f'configs/policies/{pid}.json')['method_map']
 assert hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()==design['policies'][pid]['map_sha256']

scores={};generations={};raw_samples={'multiple_choice':0,'generation':0};done=0;anchors={}
numeric=['quality_task.py','common.py','eightbit.py','arena_runtime.py','runtime_extensions.py','graph_bindings.py','int4_marlin.py']
for job in design['jobs']:
 marker=RUN/f"results/completed/{job['id']}.json"
 if not marker.exists() or str(marker.relative_to(BASE)) not in audit['source_sha256']:continue
 done+=1;entry=read(marker);path=RUN/entry['result'];assert sha(path)==entry['result_sha256']
 data=read(path);assert data['complete'] and data['quality_only'] and data['job']==job and entry['job']==job
 assert data['gpu_uuid']==entry['gpu_uuid'] and data['gpu']==entry['gpu'] and data['batch']==job['batch']
 assert data['policy_map_sha256']==design['policies'][job['policy']]['map_sha256']
 attempt=path.parent.name;command=RUN/f"logs/{job['id']}_{attempt}.command.json"
 info=read(command);assert read(command.with_name(command.name.replace('.command.json','.completion.json')))['returncode']==0
 assert info['job']==job and info['gpu_uuid']==entry['gpu_uuid']
 for name in numeric:
  assert info['code_sha256'][name]==sha(RUN/'code'/name),name
  assert sha(RUN/'code_versions'/info['code_version']/name)==info['code_sha256'][name]
 if job['kind']=='multiple_choice':
  assert data['num_fewshot']==0 and data['limit'] is None and data['max_model_len']==2048
  raw=data['raw_lm_eval'];metric=job['metric'].split(',')[0];values=[]
  assert all(v['num_fewshot']==0 for v in raw['configs'].values())
  for task,samples in raw['samples'].items():
   assert len(samples)==raw['n-samples'][task]['effective']
   assert len({s['doc_id'] for s in samples})==len(samples)
   values.extend(float(s[metric]) for s in samples)
  assert len(values)==job['samples'] and all(v in [0.,1.] for v in values)
  value=sum(values)/len(values)
  selected=raw['results'].get(job['task']) or raw.get('groups',{}).get(job['task'])
  assert math.isclose(value,float(selected[job['metric']]),abs_tol=1e-12)
  assert math.isclose(value,float(data['metric_value']),abs_tol=1e-12)
  scores[job['batch'],job['policy'],job['task']]=100*value;raw_samples['multiple_choice']+=len(values)
  if job['id']=='mc_b4_bf16_arc_easy':
   assert value==0.8345959595959596;anchors['bf16_arc_easy_exact']=True
 else:
  source=RUN/f"data/generation_{job['task']}.jsonl"
  inputs=[json.loads(x) for x in source.read_text().splitlines()][job['begin']:job['end']]
  answers=[json.loads(x) for x in (RUN/data['answers']).read_text().splitlines()]
  assert data['max_new_tokens']==256 and data['input_sha256']==sha(source)
  assert len(inputs)==len(answers)==data['samples']==job['end']-job['begin']
  assert job['begin']%8==0
  for expected,actual in zip(inputs,answers,strict=True):
   assert actual['question_id']==expected['question_id'] and actual['reference']==expected['reference']
   assert actual['prompt_tokens']==len(expected['prompt_token_ids'])<=3840
   assert len(actual['generated_token_ids'])<=256
  generations.setdefault((job['batch'],job['policy'],job['task']),[]).append((job['begin'],answers))
  raw_samples['generation']+=len(answers)
  if job['id']=='gen_b8_bf16_IWSLT_0000_0333':
   original=BASE/'runs/048/results/generation/uniform_dense_bf16/IWSLT/answers_0000_end.jsonl'
   old=[json.loads(x) for x in original.read_text().splitlines()]
   assert len(old)==333 and all(a['generated_token_ids']==b['generated_token_ids'] and a['prediction']==b['prediction'] for a,b in zip(answers,old,strict=True))
   anchors['bf16_iwslt_all_333_tokens_exact']=True

scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
original_generation_samples=0;original_generation_sources={}
for key,chunks in generations.items():
 chunks.sort();answers=[a for _,chunk in chunks for a in chunk]
 if len(answers)!=design['generation_tasks'][key[2]]:continue
 assert len({a['question_id'] for a in answers})==len(answers)
 if key[0]==8:
  policy=design['policies'][key[1]]
  original=BASE/'runs'/policy['source_run']/'results/generation'/policy['source_policy']/key[2]/'answers_0000_end.jsonl'
  previous=[json.loads(x) for x in original.read_text().splitlines()]
  assert len(previous)==len(answers)
  assert all(a['generated_token_ids']==b['generated_token_ids'] and a['prediction']==b['prediction'] for a,b in zip(answers,previous,strict=True))
  original_generation_samples+=len(answers);original_generation_sources[str(original.relative_to(BASE))]=sha(original)
 predictions=[a['prediction'].strip() or ' ' for a in answers];references=[a['reference'] for a in answers]
 scores[key]=corpus_bleu(predictions,[references]).score if key[2]=='IWSLT' else 100*sum(scorer.score(r,p)['rougeL'].fmeasure for r,p in zip(references,predictions,strict=True))/len(answers)

pp={r['method']:r['policy'] for r in design['rows']['prefill_only']};pd={r['method']:r['policy'] for r in design['rows']['prefill_decode']}
rows=list(csv.DictReader((OUT/'quality_table.csv').open()));assert len(rows)==10
assert len({(r['batch'],r['method']) for r in rows})==10
markdown=(OUT/'README.md').read_text();html=(OUT/'table.html').read_text()
for row in rows:
 batch=int(row['batch']);method=row['method'];expected={}
 for column,task in [('arc_c','arc_challenge'),('arc_e','arc_easy'),('winogrande','winogrande'),('mmlu','mmlu')]:expected[column]=scores.get((batch,pp[method],task))
 for column,task in [('cnn_dm','cnn_dm_1000'),('dialogsum','dsum'),('iwslt','IWSLT')]:expected[column]=scores.get((batch,pd[method],task))
 for column,fields in [('prefill_avg',['arc_c','arc_e','winogrande','mmlu']),('pd_avg',['cnn_dm','dialogsum','iwslt'])]:
  values=[expected[f] for f in fields];expected[column]=sum(values)/len(values) if all(v is not None for v in values) else None
 for column,value in expected.items():
  if value is None:assert row[column]==''
  else:
   assert math.isclose(float(row[column]),value,abs_tol=1e-10),(row,column,value)
   assert f'{value:.2f}' in markdown and f'{value:.2f}' in html
for path in [OUT/'README.md',RUN/'README.md']:
 for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
  assert (path.parent/target.split('#')[0]).exists(),(path,target)
assert audit['completed_jobs']==done and audit['speed_measurements']==0
if args.final:
 assert done==92 and audit['complete'] and read(RUN/'results/worker.json')['status']=='complete'
 assert raw_samples=={'multiple_choice':150856,'generation':28330}
 assert all(r['prefill_avg'] and r['pd_avg'] for r in rows) and len(anchors)==2
 assert original_generation_samples==14165
report={'verified':True,'final':args.final,'completed_jobs':done,'planned_jobs':92,'raw_samples':raw_samples,
 'recomputed_complete_metric_cells':len(scores),'display_rows':len(rows),'anchors':anchors,
 'original_inputs_and_protected_results_unchanged':True,'checkpoint_hardlink_identities_verified':8,
 'original_generation_limit':256,'speed_measurements':0,
 'b8_exact_original_generation_samples':original_generation_samples,'original_generation_source_sha256':original_generation_sources,
 'source_sha256':audit['source_sha256']}
(OUT/'delivery_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if 'source_sha256' not in k},ensure_ascii=False,indent=2))
