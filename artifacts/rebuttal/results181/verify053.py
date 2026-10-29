"""Recompute measurements and confirm sustained-idle/resource evidence."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/053';OUT=BASE/'responses/reviewer2/qwen_night_cohorts_run053'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:list(csv.DictReader(p.open()))
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['core_speed','long_speed','long_quality']);parser.add_argument('--coverage-phase',choices=['core_speed','long_speed']);parser.add_argument('--final',action='store_true');args=parser.parse_args()
D=read(RUN/'configs/design.json');W=D['watch'];audit=read(OUT/'audit.json')
protocol=read(RUN/'configs/protocol_sha256.json')
original=(BASE/'runs/052/code/experiment.py').read_text()
expected=original.replace(" parser.add_argument('--label',required=True)"," parser.add_argument('--label',required=True)\n parser.add_argument('--group')")
expected=expected.replace(' if not quality:\n  shift='," if args.group:\n  selected=design['cohort_groups'][args.group]['workloads'];cases=[c for c in cases if c['label'] in selected]\n  assert cases\n if not quality:\n  shift=")
assert (RUN/'code/experiment.py').read_text()==expected
for name in protocol:
 if name not in ['driver.py','experiment.py']:assert sha(RUN/'code'/name)==sha(BASE/'runs/052/code'/name)
for name,digest in audit['source_sha256'].items():assert sha(BASE/name)==digest,name
for name in audit['source_sha256']:
 if name.startswith(('runs/053/results/optional_continuations/','runs/053/results/coverage_dispatches/')) and name.endswith('.json'):
  request=read(BASE/name);assert sha(Path(request['entrypoint']))==request['entrypoint_sha256']
  assert request['deadline_unix']==W['deadline_unix'] and request['time_unix']<W['deadline_unix']
  assert request['code_version']==sha(RUN/'configs/protocol_sha256.json')[:16]
for name,digest in D['source_sha256'].items():assert sha(Path(name))==digest,name
for name,digest in read(RUN/'configs/protocol_sha256.json').items():assert sha(RUN/'code'/name)==digest,name
for info in read(RUN/'reference/checkpoint_reuse.json').values():
 a=Path(info['source']).stat();b=Path(info['target']).stat();assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(info['device'],info['inode'],info['bytes'])
 assert sha(Path(info['target']).parent/'config.json')==info['config_sha256']
cases={c['label']:c for c in D['workloads']}
for c in cases.values():
 p=Path(c['input']);assert sha(p)==c['input_sha256'];ids=read(p)
 assert len(ids)==c['batch'] and all(len(v)==c['prompt'] for v in ids)
def memory_valid(s):
 assert s['used_bytes']==s['total_bytes']-s['free_bytes']-s['reserved_bytes']
 assert s['reserved_bytes']>=0
def is_bare(s):
 memory_valid(s)
 return s['used_bytes']<W['maximum_idle_used_bytes'] and s['utilization_percent']<=5 and not s['processes']
records=[];raw={};jobs=timed=warmup=quality_blocks=0;quality_data={};seen_admissions=set();seen_calibrations=set();seen_cohorts=set()
for p in (RUN/'results/accepted').glob('*.json'):
 if str(p.relative_to(BASE)) not in audit['source_sha256']:continue
 m=read(p);records.append(m);phase=m['phase'];gpu=m['gpu'];admission=RUN/m['admission']
 if phase!='long_quality':
  group=D['cohort_groups'][m['group']];key=(phase,gpu,m['group'],m['round'])
  assert m['accepted'] and group['phase']==phase and m['round'] in [0,1,2] and key not in seen_cohorts;seen_cohorts.add(key)
  order=[p for p in D['rounds'][m['round']] if p!='bf16' and D['policies'][p]['scenario']==group['scenario']]
  assert [j['label'] for j in m['jobs']]==['bf16_bracket_start',*order,'bf16_bracket_end']
  assert all(j['group']==m['group'] for j in m['jobs'])
 if admission not in seen_admissions:
  a=read(admission);s=a['samples'];assert a['selected']['eligible'] and a['selected']['gpu']==gpu
  assert s[-1]['time']-s[0]['time']>=W['required_idle_seconds'] and all(y['time']-x['time']<=45 for x,y in zip(s,s[1:]))
  for v in s:
   memory_valid(v);assert v['gpu']==gpu and v['used_bytes']<W['maximum_idle_used_bytes'] and v['utilization_percent']<=5 and not v['visible_processes']
  assert is_bare(a['current']);seen_admissions.add(admission)
 for job in m['jobs']:
  jobs+=1;calpath=RUN/job['calibration'];cal=read(calpath)
  if calpath not in seen_calibrations:
   assert len(cal['samples'])==5 and cal['before_used_bytes']<W['maximum_idle_used_bytes']
   for v in cal['samples']:
    memory_valid(v);assert v['own_visible'] and not v['foreign_pids'] and v['unattributed_bytes']==max(0,v['used_bytes']-v['own_used_bytes'])
   gaps=[v['unattributed_bytes'] for v in cal['samples']]
   assert max(gaps)<=1024**3 and max(gaps)-min(gaps)<=64*1024**2
   assert cal['allowed_unattributed_bytes']==max(gaps)+D['memory_gap_allowance_bytes']
   assert len(cal['post_exit'])>=3 and all(is_bare(v) for v in cal['post_exit'][-3:]);seen_calibrations.add(calpath)
  file=RUN/job['result'];assert sha(file)==job['result_sha256'];data=read(file)
  assert data['complete'] and int(data['gpu'])==gpu and data['gpu_uuid']==cal['uuid']
  if phase!='long_quality':assert len(data['cases'])==len(group['workloads']) and {c['case']['label'] for c in data['cases']}==set(group['workloads'])
  assert data['engine']==D['engine'] and data['policy_map_sha256']==D['policies'][job['policy']]['map_sha256']
  argv=job['command'];option=lambda k:argv[argv.index(k)+1]
  assert option('--policy')==job['policy'] and option('--label')==job['full_label'] and int(option('--round'))==job['round']
  assert option('--mode')==('quality' if phase=='long_quality' else 'speed')
  assert int(option('--tier'))==(1 if phase=='core_speed' else 2)
  assert f'--physcpubind={gpu*8}-{gpu*8+7}' in argv and f'--membind={0 if gpu<4 else 1}' in argv
  assert Path(option('--checkpoint')).resolve()==(RUN/'data/work/checkpoints'/job['policy']).resolve()
  assert str((RUN/'code/experiment.py').resolve()) in [str(Path(v).resolve()) for v in argv if v.endswith('/experiment.py')]
  if phase!='long_quality':assert option('--group')==m['group']
  assert all(v['exact_equal'] for v in data['weight_audit'].values())
  assert data['placement']['cpu_affinity']==list(range(gpu*8,gpu*8+8))
  assert job['completion']['returncode']==0 and job['completion']['resource_interruption'] is None
  assert len(job['post_exit'])>=3 and all(is_bare(v) for v in job['post_exit'][-3:])
  for name,digest in job['code_sha256'].items():assert sha(RUN/'code'/name)==sha(RUN/'code_versions'/job['code_version']/name)==digest
  log=RUN/job['log'];bad=0;previous_own=None
  for line in log.with_suffix('.monitor.jsonl').read_text().splitlines():
   s=json.loads(line);memory_valid(s);assert s['gpu']==gpu and not s['foreign_pids']
   assert all(int(pid)==s['expected_pid'] for pid in s['processes'])
   own=s['processes'].get(str(s['expected_pid']),0);assert own==s['own_used_bytes']
   stable=own>0 and previous_own is not None and abs(own-previous_own)<=64*1024**2
   assert s['allocation_stable']==stable;previous_own=own
   assert s['unattributed_bytes']==max(0,s['used_bytes']-own)
   assert s['allowed_unattributed_bytes']==cal['allowed_unattributed_bytes']
   bad=bad+1 if s.get('allocation_stable',True) and s['unattributed_bytes']>s['allowed_unattributed_bytes'] else 0
   assert bad<D['resource_confirmation_samples']
  if phase=='long_quality':
   for c in data['cases']:
    value=read(RUN/c['quality_result']);assert value['complete'] and len(value['blocks'])==64
    quality_data[gpu,job['policy'],c['workload']]=value
    for i,block in enumerate(value['blocks']):
     assert block['block']==i and block['token_count']==value['case']['output']
     capture=RUN/f"results/captures/{job['full_label']}/{c['workload']}/{i:04d}.jsonl"
     tokens=[json.loads(x) for x in capture.read_text().splitlines()]
     assert len(tokens)==block['token_count'] and all(t['step']==j for j,t in enumerate(tokens))
     actual=-sum(t['logprob'] for t in tokens)/len(tokens);assert math.isclose(actual,block['avg_nll'],abs_tol=1e-12)
     quality_blocks+=1
  else:
   for c in data['cases']:
    spec=c['case'];assert spec==cases[spec['label']] and len(c['samples'])==15
    for i,s in enumerate(c['samples']):
     assert s['iteration']==i and s['warmup']==(i<5) and s['e2e_ms']>=s['ttft_ms']>0 and s['restore_ms']>=0
     assert math.isclose(s['cycle_ms'],s['restore_ms']+s['e2e_ms'],abs_tol=1e-7)
     if spec['output']>1:assert math.isclose(s['tpot_ms'],(s['e2e_ms']-s['ttft_ms'])/(spec['output']-1),abs_tol=1e-7)
     else:assert s['tpot_ms'] is None
     assert s['gpu_before']['physical_gpu_index']==s['gpu_after']['physical_gpu_index']==gpu
     ids=s['output_token_ids'];assert len(ids)==spec['batch'] and all(len(v)==spec['output'] for v in ids)
     assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==s['output_sha256']
    values={k:statistics.median(s[k] for s in c['samples'] if not s['warmup']) for k in c['medians']};assert values==c['medians']
    key=(phase,gpu,job['round'],job['label'],spec['label']);assert key not in raw;raw[key]=values;timed+=10;warmup+=5

qidx={(r['batch'],r['method']):r for r in load(BASE/'responses/reviewer2/qwen_original_quality_run050/quality_table.csv')}
nidx={(r['workload'],r['method']):r for r in load(BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049/comparison.csv')}
scene_rows=load(OUT/'comparison.csv');scene_idx={(r['phase'],int(r['gpu']),r['workload'],r['method']):r for r in scene_rows}
for row in scene_rows:
 phase=row['phase'];gpu=int(row['gpu']);w=row['workload'];stats=[];drifts=[]
 expected_avg=float(qidx[row['batch'],row['method']]['prefill_avg' if row['scenario']=='prefill_only' else 'pd_avg']);assert float(row['original_task_avg'])==expected_avg
 if phase=='core_speed':
  prior=nidx[w,row['method']]
  assert all(float(row[k])==float(prior[k]) for k in ['delta_nll','two_se']) and row['nll_source']=='049 same-load 64 windows'
 for ri in range(3):
  key=lambda label:(phase,gpu,ri,label,w)
  if key('bf16_bracket_start') not in raw:continue
  anchors=[raw[key('bf16_bracket_'+s)] for s in ['start','end']]
  reference={k:statistics.mean(a[k] for a in anchors) for k in anchors[0]}
  values=reference if row['policy']=='bf16' else raw[key(row['policy'])]
  stats.append({**values,**{k.replace('_ms','_speedup'):reference[k]/values[k] for k in reference if k!='restore_ms'}})
  drifts.append(abs(anchors[0]['e2e_ms']-anchors[1]['e2e_ms'])/reference['e2e_ms'])
 assert len(stats)==int(row['completed_rounds'])
 if not stats:
  assert row['status']=='待测' and row['e2e_speedup']=='';continue
 for k in stats[0]:assert math.isclose(float(row[k]),statistics.median(s[k] for s in stats),abs_tol=1e-10)
 ratios=[s['e2e_speedup'] for s in stats];span=(max(ratios)-min(ratios))/statistics.median(ratios)
 assert math.isclose(float(row['relative_round_range']),span,abs_tol=1e-12)
 if len(stats)==3:assert row['status']==('需复核' if span>.05 or max(drifts)>.05 else '三轮完成')
for row in load(OUT/'paired_comparisons.csv'):
 phase=row['phase'];g=int(row['gpu']);w=row['workload'];d=scene_idx[phase,g,w,row['method']];h=scene_idx[phase,g,w,row['baseline']]
 ratios={k:[] for k in ['e2e','cycle']}
 for ri in range(3):
  dk=(phase,g,ri,d['policy'],w);hk=(phase,g,ri,h['policy'],w)
  if dk in raw and hk in raw:
   for k in ratios:ratios[k].append(raw[hk][k+'_ms']/raw[dk][k+'_ms'])
 assert len(ratios['e2e'])==int(row['paired_rounds'])
 for k,v in ratios.items():
  assert math.isclose(float(row[k+'_gain_percent']),100*(float(d[k+'_speedup'])/float(h[k+'_speedup'])-1),abs_tol=1e-10)
  assert math.isclose(float(row['median_paired_'+k+'_gain_percent']),100*(statistics.median(v)-1),abs_tol=1e-10)
 assert int(row['faster_rounds'])==sum(v>1 for v in ratios['e2e']) and row['status']==d['status']
 assert math.isclose(float(row['min_paired_gain_percent']),100*(min(ratios['e2e'])-1),abs_tol=1e-10)
 assert math.isclose(float(row['max_paired_gain_percent']),100*(max(ratios['e2e'])-1),abs_tol=1e-10)
panels={'B=8 / 输入=2048 / 输出=64':('center_pp','center_pd'),'B=4 / 输入=2048 / 输出=64':('batch4_pp','batch4_pd'),
 'B=8 / 输入=4096 / 输出=64':('prompt4096_pp','prompt4096_pd'),'B=8 / 输入=2048 / 输出=128':('center_pp','output128_pd')}
table=load(OUT/'table2.csv');assert len(table)==20*len(audit['core_rounds_by_gpu'])
assert len({(v['gpu'],v['configuration'],v['method']) for v in table})==len(table)
for row in table:
 g=int(row['gpu']);ppw,pdw=panels[row['configuration']];pp=scene_idx['core_speed',g,ppw,row['method']];pd=scene_idx['core_speed',g,pdw,row['method']]
 q=qidx[pd['batch'],row['method']]
 for k in ['arc_c','arc_e','winogrande','mmlu','prefill_avg','cnn_dm','dialogsum','iwslt','pd_avg']:assert float(row[k])==float(q[k])
 for k,original in [('pp_ttft',pp['ttft_speedup']),*[(f'pd_{k}',pd[k+'_speedup']) for k in ['ttft','tpot','e2e','cycle']]]:
  assert (row[k]==original=='') or (row[k]!='' and original!='' and float(row[k])==float(original))
 for prefix,value in [('pp',pp),('pd',pd)]:
  for k in ['delta_nll','two_se']:assert float(row[prefix+'_'+k])==float(value[k])
 assert row['pp_status']==pp['status'] and row['pd_status']==pd['status'] and row['pp_rounds']==pp['completed_rounds'] and row['pd_rounds']==pd['completed_rounds']
assert load(OUT/'long_table.csv')==[r for r in scene_rows if r['phase']=='long_speed']
assert load(OUT/'table2_primary.csv')==[r for r in table if int(r['gpu'])==audit['core_primary_gpu']]
assert load(OUT/'long_primary.csv')==[r for r in scene_rows if r['phase']=='long_speed' and int(r['gpu'])==audit['long_primary_gpu']]
assert len(load(OUT/'table2_primary.csv'))==audit['primary_table2_rows'] and len(load(OUT/'long_primary.csv'))==audit['primary_long_rows']
for row in load(OUT/'long_nll.csv'):
 g=int(row['gpu']);p=row['policy'];w=row['workload'];v=quality_data[g,p,w]
 assert math.isclose(float(row['mean_nll']),statistics.mean(x['avg_nll'] for x in v['blocks']),abs_tol=1e-12)
 if row['delta_nll']:
  b=quality_data[g,'bf16',w];d=[x['avg_nll']-y['avg_nll'] for x,y in zip(v['blocks'],b['blocks'])]
  assert math.isclose(float(row['delta_nll']),statistics.mean(d),abs_tol=1e-12)
  assert math.isclose(float(row['two_se']),2*statistics.stdev(d)/8,abs_tol=1e-12)
assert timed==audit['timed_samples'] and warmup==audit['warmup_samples']
group_counts={}
for m in records:
 if m['phase']!='long_quality':
  v=group_counts.setdefault(str(m['gpu']),{});v[m['group']]=v.get(m['group'],0)+1
assert group_counts==audit['group_rounds_by_gpu']
coverage_gpus={}
for group,spec in D['cohort_groups'].items():
 candidates=[]
 for gpu in {m['gpu'] for m in records if m['phase']==spec['phase']}:
  rows=[m for m in records if m['phase']==spec['phase'] and m['group']==group and m['gpu']==gpu]
  if len(rows)==3 and {m['round'] for m in rows}=={0,1,2}:candidates.append((max(m['finished_unix'] for m in rows),gpu))
 if candidates:coverage_gpus[group]=min(candidates)[1]
assert coverage_gpus==audit['group_coverage_gpus']
for phase,field in [('core_speed','core_coverage_complete'),('long_speed','long_coverage_complete')]:
 assert audit[field]==all(g in coverage_gpus for g,spec in D['cohort_groups'].items() if spec['phase']==phase)
for m in records:
 if 'dispatch_request' not in m:continue
 request=RUN/m['dispatch_request'];assert str(request.relative_to(BASE)) in audit['source_sha256']
 assert read(request)['time_unix']<=m['started_unix']
 prior=[v for v in records if v['phase']==m['phase'] and v['finished_unix']<m['started_unix']]
 complete_before={g for g,spec in D['cohort_groups'].items() if spec['phase']==m['phase'] and any(
  {v['round'] for v in prior if v['gpu']==gpu and v['group']==g}=={0,1,2} for gpu in W['preferred_cards'])}
 assert m['group']==next(g for g,spec in D['cohort_groups'].items() if spec['phase']==m['phase'] and g not in complete_before)
case_coverage={w:coverage_gpus[g] for g,spec in D['cohort_groups'].items() if g in coverage_gpus for w in spec['workloads']}
panel_workloads={r['configuration']:(('batch4_pp','batch4_pd') if 'B=4' in r['configuration'] else ('prompt4096_pp','prompt4096_pd') if '输入=4096' in r['configuration'] else ('center_pp','output128_pd') if '输出=128' in r['configuration'] else ('center_pp','center_pd')) for r in load(OUT/'table2.csv')}
table_rows=load(OUT/'table2.csv');table_idx={(int(r['gpu']),r['configuration'],r['method']):r for r in table_rows};expected_coverage=[];covered_keys=set()
for base in table_rows:
 key=(base['configuration'],base['method'])
 if key in covered_keys:continue
 covered_keys.add(key);ppw,pdw=panel_workloads[base['configuration']];pg=case_coverage.get(ppw);dg=case_coverage.get(pdw)
 row={k:v for k,v in base.items() if k!='gpu'};row.update(pp_gpu=str(pg) if pg is not None else '',pd_gpu=str(dg) if dg is not None else '')
 for gpu,fields in [(pg,['pp_ttft','pp_rounds','pp_status']),(dg,['pd_ttft','pd_tpot','pd_e2e','pd_cycle','pd_rounds','pd_status'])]:
  for field in fields:row[field]=table_idx[gpu,*key][field] if gpu is not None else '0' if field.endswith('_rounds') else '待测' if field.endswith('_status') else ''
 expected_coverage.append(row)
assert load(OUT/'table2_coverage.csv')==expected_coverage
assert load(OUT/'long_coverage.csv')==[r for r in scene_rows if r['phase']=='long_speed' and int(r['gpu'])==case_coverage.get(r['workload'])]
assert len(expected_coverage)==audit['coverage_table2_rows'] and len(load(OUT/'long_coverage.csv'))==audit['coverage_long_rows']
for phase,field in [('core_speed','core_primary_gpu'),('long_speed','long_primary_gpu'),('long_quality','long_quality_gpu')]:
 candidates=[]
 for g in {m['gpu'] for m in records if m['phase']==phase}:
  subset=[m for m in records if m['phase']==phase and m['gpu']==g]
  if phase=='long_quality':
   policies=[m['jobs'][0]['policy'] for m in subset];assert len(set(policies))==len(policies)
   complete=set(policies)=={v['policy'] for v in D['rows']['prefill_decode']}
  else:
   complete=all(sum(m['group']==name for m in subset)==3 for name,spec in D['cohort_groups'].items() if spec['phase']==phase)
  if complete:candidates.append((max(m['finished_unix'] for m in subset),g))
 assert audit[field]==(min(candidates)[1] if candidates else None)
if args.phase:
 field={'core_speed':'core_primary_gpu','long_speed':'long_primary_gpu','long_quality':'long_quality_gpu'}[args.phase]
 assert audit[field] is not None
if args.coverage_phase:assert audit['core_coverage_complete' if args.coverage_phase=='core_speed' else 'long_coverage_complete']
if args.final:
 import time
 current=read(RUN/'results/worker.json')
 assert current['status']=='complete' or (current['status']=='window_ended' and time.time()>=audit['deadline_unix'])
 if current['status']=='complete' and current.get('completion_scope')=='per_group_same_gpu_three_rounds':assert audit['core_coverage_complete']
result={'verified':True,'phase':args.phase,'final':args.final,'accepted_records':len(records),'accepted_jobs':jobs,
 'timed_samples':timed,'warmup_samples':warmup,'quality_blocks_recomputed':quality_blocks,
 'idle_admissions_checked':len(seen_admissions),'context_calibrations_checked':len(seen_calibrations),
 'resource_evidence_verified':True,'per_device_normalization_verified':True,'original_sources_unchanged':True,
 'paired_comparisons_verified':True,'table2_joins_verified':True,'verification_code_sha256':sha(Path(__file__).resolve()),
 'group_coverage_and_primary_selection_verified':True,
 'core_nll_source_and_primary_exports_verified':True,
 'coverage_phase':args.coverage_phase,'per_group_same_gpu_coverage_verified':True,
 'group_coverage_gpus':coverage_gpus,'core_coverage_complete':audit['core_coverage_complete'],'long_coverage_complete':audit['long_coverage_complete'],
 'core_primary_gpu':audit['core_primary_gpu'],'long_primary_gpu':audit['long_primary_gpu'],'long_quality_gpu':audit['long_quality_gpu'],
 'source_sha256':audit['source_sha256']}
name=args.coverage_phase+'_coverage_delivery_audit.json' if args.coverage_phase else args.phase+'_delivery_audit.json' if args.phase else 'delivery_audit.json'
(OUT/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
