"""Independently derive per-device speed ratios from every raw timed sample."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/051'
OUT=BASE/'responses/reviewer2/qwen_quiet_speed_run051'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:list(csv.DictReader(p.open()))
parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[3,4]);parser.add_argument('--final',action='store_true');args=parser.parse_args()
design=read(RUN/'configs/design.json');audit=read(OUT/'audit.json');cards=[args.gpu] if args.gpu else design['cards']
for path,digest in audit['source_sha256'].items():assert sha(BASE/path)==digest,path
for path,digest in design['source_sha256'].items():assert sha(Path(path))==digest,path
for path,digest in read(RUN/'configs/protocol_sha256.json').items():assert sha(RUN/'code'/path)==digest,path
for entry in read(RUN/'reference/checkpoint_reuse.json').values():
 a=Path(entry['source']).stat();b=Path(entry['target']).stat()
 assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(entry['device'],entry['inode'],entry['bytes'])
 assert sha(Path(entry['target']).parent/'config.json')==entry['config_sha256']
cases={c['label']:c for c in design['workloads']}
for c in cases.values():
 path=Path(c['input']);assert sha(path)==c['input_sha256'];ids=read(path)
 assert len(ids)==c['batch'] and all(len(row)==c['prompt'] for row in ids)
markers={};raw={};timed=warmup=case_count=0
for path in (RUN/'results/completed').glob('*.json'):
 if str(path.relative_to(BASE)) not in audit['source_sha256']:continue
 marker=read(path)
 if marker['gpu'] not in cards:continue
 markers[path.stem]=marker;result=RUN/marker['result'];data=read(result)
 assert sha(result)==marker['result_sha256'] and data['complete'] and data['engine']==design['engine']
 gpu=marker['gpu'];assert data['gpu_uuid']==marker['gpu_uuid']==read(RUN/f'configs/device_gpu{gpu}.json')['uuid']
 assert int(data['gpu'])==gpu and data['round']==marker['round'] and data['policy']==marker['policy']
 assert data['placement']['cpu_affinity']==list(range(gpu*8,gpu*8+8))
 assert data['policy_map_sha256']==design['policies'][marker['policy']]['map_sha256']
 assert all(a['exact_equal'] for a in data['weight_audit'].values())
 log=RUN/marker['log'];command=read(log.with_suffix('.command.json'));completion=read(log.with_suffix('.completion.json'))
 assert completion['returncode']==0 and not completion['foreign_pids']
 for name,digest in command['code_sha256'].items():
  assert sha(RUN/'code'/name)==digest and sha(RUN/'code_versions'/command['code_version']/name)==digest
 for line in log.with_suffix('.monitor.jsonl').read_text().splitlines():
  observed=json.loads(line);assert observed['gpu']==gpu
  assert all(p['pid']==observed['expected_pid'] for p in observed['processes'])
 for cell in data['cases']:
  c=cell['case'];assert c==cases[c['label']];samples=cell['samples'];assert len(samples)==15
  for i,s in enumerate(samples):
   assert s['iteration']==i and s['warmup']==(i<5)
   assert s['e2e_ms']>0 and 0<s['ttft_ms']<=s['e2e_ms'] and s['restore_ms']>=0
   assert math.isclose(s['cycle_ms'],s['e2e_ms']+s['restore_ms'],abs_tol=1e-7)
   if c['output']>1:assert math.isclose(s['tpot_ms'],(s['e2e_ms']-s['ttft_ms'])/(c['output']-1),abs_tol=1e-7)
   else:assert s['tpot_ms'] is None
   assert s['gpu_before']['physical_gpu_index']==s['gpu_after']['physical_gpu_index']==gpu
   tokens=s['output_token_ids'];assert len(tokens)==c['batch'] and all(len(t)==c['output'] for t in tokens)
   assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest()==s['output_sha256']
  values={k:statistics.median(s[k] for s in samples if not s['warmup']) for k in cell['medians']}
  assert values==cell['medians'];raw[path.stem,c['label']]=values
  case_count+=1;timed+=10;warmup+=5

qualities={(r['batch'],r['method']):r for r in load(BASE/'responses/reviewer2/qwen_original_quality_run050/quality_table.csv')}
rows=[r for r in load(OUT/'comparison.csv') if int(r['gpu']) in cards];assert len(rows)==35*len(cards)
for row in rows:
 gpu=int(row['gpu']);w=row['workload'];stats=[];refs=[]
 q=qualities[row['batch'],row['method']];assert float(row['original_task_avg'])==float(q['prefill_avg' if row['scenario']=='prefill_only' else 'pd_avg'])
 for ri in range(3):
  ids=[f'g{gpu}_r{ri}_bf16_bracket_{s}' for s in ['start','end']];own=f"g{gpu}_r{ri}_{row['policy']}"
  if not all(i in markers for i in ids) or (row['policy']!='bf16' and own not in markers):continue
  reference={k:statistics.mean(raw[i,w][k] for i in ids) for k in raw[ids[0],w]}
  values=reference if row['policy']=='bf16' else raw[own,w]
  stats.append({**values,**{k.replace('_ms','_speedup'):reference[k]/values[k] for k in reference if k!='restore_ms'}})
  refs.append(abs(raw[ids[0],w]['e2e_ms']-raw[ids[1],w]['e2e_ms'])/reference['e2e_ms'])
 assert int(row['completed_rounds'])==len(stats)
 if not stats:
  assert not row['ttft_speedup'];continue
 for key in stats[0]:assert math.isclose(float(row[key]),statistics.median(s[key] for s in stats),abs_tol=1e-10)
 ratios=[s['e2e_speedup'] for s in stats];span=(max(ratios)-min(ratios))/statistics.median(ratios)
 assert math.isclose(float(row['relative_round_range']),span,abs_tol=1e-12)
 assert math.isclose(float(row['max_bf16_anchor_drift']),max(refs),abs_tol=1e-12)
 expected='需复核' if span>.05 or max(refs)>.05 else '三轮完成'
 if len(stats)==3:assert row['status']==expected
 if args.final or args.gpu:assert len(stats)==3
if args.final:
 assert len(markers)==54 and case_count==234 and timed==2340 and warmup==1170
 assert read(RUN/'results/worker.json')['status']=='complete' and audit['complete']
elif args.gpu:assert len(markers)==27 and case_count==117 and timed==1170
result={'verified':True,'final':args.final,'cards':cards,'complete_processes':len(markers),'case_rounds':case_count,
 'timed_samples':timed,'warmup_samples':warmup,'scene_rows_checked':len(rows),
 'unchanged_sources':True,'per_device_normalization_verified':True,'all_raw_samples_retained':True,
 'source_sha256':audit['source_sha256']}
name=f'gpu{args.gpu}_delivery_audit.json' if args.gpu else 'delivery_audit.json'
(OUT/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
