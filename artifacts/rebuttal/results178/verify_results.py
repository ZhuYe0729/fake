"""Audit the new dual-scenario tables against raw NLL and individual timings."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from html.parser import HTMLParser

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/049'
OUT=BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
 design=read(RUN/'configs/design.json');published=read(OUT/'audit.json')
 if not args.allow_partial:assert published['complete']
 for path,digest in read(RUN/'configs/protected.json').items():assert sha(Path(path))==digest,path
 for name,digest in read(RUN/'configs/protocol_sha256.json').items():assert sha(RUN/'code'/name)==digest,name
 for path,digest in design['source_sha256'].items():assert sha(Path(path))==digest,path
 for relative,digest in published['source_sha256'].items():assert sha(BASE/relative)==digest,relative
 canonical={r['method']:r for r in read(BASE/'responses/reviewer2/qwen3_main_table/provenance.json')['rows']}
 for pid,spec in design['policies'].items():
  mapping=read(RUN/f'configs/policies/{pid}.json')['method_map']
  actual=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
  assert actual==spec['map_sha256']
  if pid!='bf16':
   name=spec['display'].replace('（均衡）',' (Balanced)').replace('（速度优先）',' (Max-speed)')
   references=['DASO (Max-speed)','HAWQ-V3 (Max-speed)'] if pid=='pp_shared_fast' else [name]
   assert all(canonical[n][spec['scenario']]['map_sha256']==actual for n in references)
 rows=list(csv.DictReader((OUT/'comparison.csv').open()))
 cases=[c for c in design['workloads'] if c['tier']<=published['tier']]
 assert len(rows)==len(cases)*5==published['display_rows']
 assert read(RUN/'configs/center_speed_inputs.json')==read(BASE/'runs/048/configs/speed_inputs.json')
 measured=0;timed=0;sample_count=0;measurement_files=[]
 for r in rows:
  case=next(c for c in cases if c['label']==r['workload']);pid=r['policy'];w=r['workload']
  assert any(s['method']==r['method'] and s['policy']==pid for s in design['rows'][r['scenario']])
  for key in ['batch','prompt','output']:assert int(r[key])==case[key]
  if r['delta_nll']!='':
   raw=read(RUN/f'results/quality/{pid}/{w}.json');bf=read(RUN/f'results/quality/bf16/{w}.json')
   assert raw['case']==bf['case']==case and len(raw['blocks'])==len(bf['blocks'])==64
   count=case['prompt']-1 if case['scenario']=='prefill_only' else case['output']
   assert all(b['block']==i and b['token_count']==count and math.isfinite(b['avg_nll']) for i,b in enumerate(raw['blocks']))
   differences=[a['avg_nll']-b['avg_nll'] for a,b in zip(raw['blocks'],bf['blocks'])]
   assert math.isclose(float(r['delta_nll']),statistics.mean(differences),abs_tol=1e-12)
   assert math.isclose(float(r['two_se']),statistics.stdev(differences)/4,abs_tol=1e-12)
   measured+=1
  round_values=[]
  for ri in range(3):
   paths=[RUN/f"results/measurements/speed_t{case['tier']}_r{ri}_bf16_bracket_{end}.json" for end in ['start','end']]
   own=RUN/f"results/measurements/speed_t{case['tier']}_r{ri}_{pid}.json"
   if not all(p.exists() for p in paths) or (pid!='bf16' and not own.exists()):continue
   anchors=[next(c for c in read(p)['cases'] if c['case']['label']==w)['medians'] for p in paths]
   reference={k:(anchors[0][k]+anchors[1][k])/2 for k in anchors[0]}
   values=reference if pid=='bf16' else next(c for c in read(own)['cases'] if c['case']['label']==w)['medians']
   round_values.append({**values,**{k.replace('_ms','_speedup'):reference[k]/v for k,v in values.items() if k!='restore_ms'}})
  assert int(r['completed_speed_rounds'])==len(round_values)
  for key in ['ttft_ms','tpot_ms','e2e_ms','restore_ms','cycle_ms','ttft_speedup','tpot_speedup','e2e_speedup','cycle_speedup']:
   values=[v[key] for v in round_values if key in v]
   if values:assert math.isclose(float(r[key]),statistics.median(values),abs_tol=1e-12),(w,pid,key)
   else:assert r[key]==''
  timed+=int(r['completed_speed_rounds'])==3
 for p in sorted((RUN/'results/measurements').glob('speed_*.json')):
  d=read(p);assert d['complete'] and d['engine']==design['engine']
  assert all(v['exact_equal'] for v in d['weight_audit'].values())
  assert int(d['gpu'])==read(RUN/'configs/formal_device.json')['physical_gpu']
  for c in d['cases']:
   assert c['case'] in cases
   samples=[s for s in c['samples'] if not s['warmup']]
   assert len(samples)==10 and len(c['samples'])==15
   for s in c['samples']:
    assert len(s['output_token_ids'])==c['case']['batch']
    assert all(len(ids)==c['case']['output'] for ids in s['output_token_ids'])
    assert all(math.isfinite(s[k]) and s[k]>0 for k in ['ttft_ms','e2e_ms','cycle_ms'])
   for key,value in c['medians'].items():assert math.isclose(value,statistics.median(s[key] for s in samples),abs_tol=1e-12)
   sample_count+=len(samples)
  measurement_files.append(p)
 assert measured==published['quality_rows'] and timed==published['three_round_speed_rows']
 text=(OUT/'README.md').read_text();tables=re.findall(r'(?m)^\|.*(?:\n\|.*)+',text)
 assert len(tables)==len(cases) and sum(len(t.splitlines())-2 for t in tables)==len(rows)
 for target in re.findall(r'\]\(([^)]+)\)',text):assert (OUT/target.split('#')[0]).exists(),target
 class Tables(HTMLParser):
  def __init__(self):super().__init__();self.body=False;self.inside=False;self.rows=[]
  def handle_starttag(self,tag,attrs):
   if tag=='tbody':self.body=True
   if tag=='tr' and self.body:self.current=[]
   if tag=='td' and self.body:self.inside=True;self.value=''
  def handle_data(self,data):
   if self.inside:self.value+=data
  def handle_endtag(self,tag):
   if tag=='td' and self.inside:self.current.append(self.value);self.inside=False
   if tag=='tr' and self.body:self.rows.append(self.current)
   if tag=='tbody':self.body=False
 html=Tables();html.feed((OUT/'table.html').read_text())
 ordered=[r for scene in ['prefill_only','prefill_decode'] for r in rows if r['scenario']==scene]
 def fmt(v,n=3):return '待测' if v=='' else f'{float(v):.{n}f}'
 for cells,r in zip(html.rows,ordered):
  pp=r['scenario']=='prefill_only'
  expected=[f"{r['batch']} / {r['prompt']} / "+('—' if pp else r['output']),r['method'],
   '待测' if r['delta_nll']=='' else fmt(r['delta_nll'],5)+' ± '+fmt(r['two_se'],5),fmt(r['ttft_speedup'])]
  if not pp:expected.extend(fmt(r[k]) for k in ['tpot_speedup','e2e_speedup','cycle_speedup'])
  assert cells==[*expected,r['timing_status']]
 assert len(html.rows)==len(rows)
 if published['complete']:
  assert len(measurement_files)==27*published['tier']
  assert measured==timed==len(rows)
 result={'verified':True,'complete':published['complete'],'display_rows':len(rows),'measured_nll_rows':measured,
  'three_round_speed_rows':timed,'formal_processes_checked':len(measurement_files),'formal_timed_case_samples_checked':sample_count,
  'all_fixed_policies_match_primary_table':True,'both_scenarios':True,'old_results_unchanged':True,
  'center_inputs_equal_to_048':True,'html_cells_match_csv':True,'normalized_speed_cells_recomputed':True,
  'source_files_checked':len(published['source_sha256']),'errors':[]}
 (BASE/'results178/audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
