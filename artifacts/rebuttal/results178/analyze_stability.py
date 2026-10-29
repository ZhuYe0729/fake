"""Describe every raw repeat and the frozen DASO points' ordering, without exclusions."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/049'
OUT=BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];sources={};known_pids=set()
for p in sorted((RUN/'results/measurements').glob('speed_t1_*.json')):
 d=json.loads(p.read_text());sources[str(p.relative_to(BASE))]=sha(p);known_pids.add(d['placement']['pid'])
 for c in d['cases']:
  samples=[s for s in c['samples'] if not s['warmup']];values=[s['e2e_ms'] for s in samples]
  assert len(values)==10
  for s in samples:
   assert abs(s['cycle_ms']-s['e2e_ms']-s['restore_ms'])<1e-8
   if c['case']['output']>1:assert abs(s['e2e_ms']-s['ttft_ms']-(c['case']['output']-1)*s['tpot_ms'])<1e-8
  rows.append({'source':p.stem,'policy':d['policy'],'round':d['round'],'workload':c['case']['label'],
   'e2e_median_ms':statistics.median(values),'timed_min_ms':min(values),'timed_max_ms':max(values),
   'timed_range_fraction':(max(values)-min(values))/statistics.median(values),
   'sample_count':len(values),'gpu':d['gpu']})
if rows:
 with (OUT/'sample_statistics.csv').open('w') as handle:
  writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
comparison=[]
cross_method=[]
for w in ['center_pd','batch4_pd','prompt4096_pd','output128_pd']:
 ratios=[]
 for ri in range(3):
  a=next((r for r in rows if r['workload']==w and r['round']==ri and r['policy']=='pd_daso_bal'),None)
  b=next((r for r in rows if r['workload']==w and r['round']==ri and r['policy']=='pd_daso_fast'),None)
  if a and b:ratios.append(a['e2e_median_ms']/b['e2e_median_ms'])
 comparison.append({'workload':w,'speed_oriented_vs_balanced_e2e_ratios':ratios,'completed_paired_rounds':len(ratios),
  'speed_oriented_faster_rounds':sum(v>1 for v in ratios),'ordering_reversed_across_rounds':any(v>1 for v in ratios) and any(v<1 for v in ratios)})
 entry={'workload':w}
 for pid,key in [('pd_daso_bal','balanced'),('pd_daso_fast','speed')]:
  values=[]
  for ri in range(3):
   d=next((r for r in rows if r['workload']==w and r['round']==ri and r['policy']==pid),None)
   h=next((r for r in rows if r['workload']==w and r['round']==ri and r['policy']=='pd_hawq_fast'),None)
   if d and h:values.append(h['e2e_median_ms']/d['e2e_median_ms'])
  entry[key+'_e2e_ratios_vs_hawq_speed']=values
  entry[key+'_faster_all_three_rounds']=len(values)==3 and all(v>1 for v in values)
 cross_method.append(entry)
monitors=list((RUN/'logs').glob('device_monitor_*.jsonl'));formal_samples=0;foreign=[];unknown=[];known_exit=[]
for p in monitors:
 sources[str(p.relative_to(BASE))]=sha(p)
 for line in p.read_text().splitlines():
  d=json.loads(line)
  if d['stage'].get('stage')!='formal_speed':continue
  index=d['stage']['gpu'];v=next(v for v in d['devices'] if v['index']==index);formal_samples+=1
  if any(proc['pid'] not in known_pids and proc['belongs_to_run049'] is False for proc in v['processes']):foreign.append(d['time_unix'])
  if any(proc['pid'] not in known_pids and proc['belongs_to_run049'] is None for proc in v['processes']):unknown.append(d['time_unix'])
  if any(proc['pid'] in known_pids and proc['belongs_to_run049'] is not True for proc in v['processes']):known_exit.append(d['time_unix'])
result={'complete':len(rows)==117,'formal_processes':len({r['source'] for r in rows}),'formal_cases':len(rows),
 'timed_samples':sum(r['sample_count'] for r in rows),'all_iterations_retained':True,
 'within_case_ranges_over_5percent':[{'source':r['source'],'workload':r['workload'],'range_fraction':r['timed_range_fraction']} for r in rows if r['timed_range_fraction']>.05],
 'daso_pair_comparisons':comparison,'sampled_formal_gpu_process_checks':formal_samples,
 'cross_method_comparisons':cross_method,
 'cases_with_one_fixed_daso_point_faster_than_hawq_speed_in_all_three_rounds':sum(c['balanced_faster_all_three_rounds'] or c['speed_faster_all_three_rounds'] for c in cross_method),
 'foreign_gpu_process_sample_times':foreign,'unknown_gpu_process_sample_times':unknown,
 'known_own_pid_exit_sample_times':known_exit,
 'pid_classification':'Cross-check sampled PIDs with raw formal-process metadata; a disappearing cmdline at process exit does not make a known measured process foreign. Raw logs remain unchanged.',
 'process_monitor_scope':'Read-only samples every two seconds, covering the measured card. Sampling cannot exclude sub-interval interference.',
 'source_sha256':sources}
(OUT/'stability.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','within_case_ranges_over_5percent']},ensure_ascii=False,indent=2))
