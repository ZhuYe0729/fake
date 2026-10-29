"""Reporter regression fixture, exclusively in /tmp; never new measurements."""
from pathlib import Path
import copy
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

BASE=Path(__file__).resolve().parents[1];OLD=BASE/'runs/052';REAL=BASE/'runs/053'
ROOT=Path(tempfile.mkdtemp(prefix='qwen053_report_fixture_'))/'rebuttal';RUN=ROOT/'runs/053'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def link(a,b):b.parent.mkdir(parents=True,exist_ok=True);b.symlink_to(a.resolve(),target_is_directory=a.is_dir())
for name in ['code','configs','reference','data']:link(REAL/name,RUN/name)
link(OLD,ROOT/'runs/052')
for name in ['qwen_original_quality_run050','qwen_nearby_dual_scenario_run049']:
 link(BASE/'responses/reviewer2'/name,ROOT/'responses/reviewer2'/name)
for name in ['report053.py','verify053.py']:
 p=ROOT/'results181'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/'results181'/name,p)
write(ROOT/'FIXTURE_ONLY.json',{'fixture_only':True,'new_measurements':False,'source':'052 actual accepted first round; metadata adapted only for reporter regression'})
write(RUN/'results/worker.json',{'status':'running','deadline_unix':1793236373.0})
d=read(REAL/'configs/design.json');protocol=read(REAL/'configs/protocol_sha256.json');version=sha(REAL/'configs/protocol_sha256.json')[:16]
link(REAL/'code',RUN/'code_versions'/version)
source=read(OLD/'results/accepted/core_speed_g3_r0.json');lease=RUN/'results/leases/fixture'
write(lease/'admission.json',read(OLD/source['admission']))
write(lease/'calibration.json',read(OLD/source['jobs'][0]['calibration']))
markers=[]
for group in [k for k,v in d['cohort_groups'].items() if v['phase']=='core_speed']:
 m=copy.deepcopy(source);m['group']=group;m['jobs']=[];m['admission']='results/leases/fixture/admission.json'
 for item in source['jobs']:
  if item['policy']!='bf16' and d['policies'][item['policy']]['scenario']!=d['cohort_groups'][group]['scenario']:continue
  j=copy.deepcopy(item);label=item['full_label']+'_fixture_'+group;j.update(full_label=label,group=group,code_version=version,code_sha256=protocol,calibration='results/leases/fixture/calibration.json')
  argv=j['command']
  for key,value in [('--checkpoint',str(REAL/'data/work/checkpoints'/j['policy'])),('--label',label)]:argv[argv.index(key)+1]=value
  argv=[str(REAL/'code/experiment.py') if v.endswith('/experiment.py') else v for v in argv];argv.extend(['--group',group]);j['command']=argv
  value=read(OLD/j['result']);value['cases']=[c for c in value['cases'] if c['case']['label'] in d['cohort_groups'][group]['workloads']]
  for c in value['cases']:c['case']=next(spec for spec in d['workloads'] if spec['label']==c['case']['label'])
  j['result']=f'results/measurements/{label}.json';write(RUN/j['result'],value);j['result_sha256']=sha(RUN/j['result'])
  old_log=OLD/item['log'];j['log']=f'logs/{label}.log';log=RUN/j['log'];log.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old_log,log)
  monitor=[];previous=None
  for line in old_log.with_suffix('.monitor.jsonl').read_text().splitlines():
   v=json.loads(line);own=v['own_used_bytes'];v['allocation_stable']=own>0 and previous is not None and abs(own-previous)<=64*1024**2;previous=own;monitor.append(json.dumps(v))
  log.with_suffix('.monitor.jsonl').write_text('\n'.join(monitor)+'\n');write(log.with_suffix('.command.json'),j);write(log.with_suffix('.completion.json'),j['completion']);m['jobs'].append(j)
 markers.append((group,m))
def run(name,ok=True):
 p=subprocess.run([sys.executable,str(ROOT/'results181'/name)],capture_output=True,text=True)
 assert (p.returncode==0)==ok,(name,p.stdout,p.stderr)
 return p
passed=[]
for group,marker in markers:
 write(RUN/f'results/accepted/core_speed_{group}_g3_r0.json',marker)
 run('report053.py');run('verify053.py');passed.append(group)
out=ROOT/'responses/reviewer2/qwen_night_cohorts_run053'
audit=read(out/'audit.json');assert audit['formal_case_rounds']==39 and audit['timed_samples']==390
for name,column in [('table2.csv','pd_ttft'),('table2.csv','pd_delta_nll'),('comparison.csv','delta_nll')]:
 table=out/name;rows=list(csv.DictReader(table.open()));rows[0][column]=str(float(rows[0][column])+.1)
 with table.open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 run('verify053.py',ok=False);run('report053.py');run('verify053.py')
summary={'fixture_only':True,'new_measurements':False,'temporary_root':str(ROOT),'partial_prefill_passed':True,'scenario_merge_passed':True,'corrupt_table_join_rejected':True,'corrupt_nll_join_rejected':True,'corrupt_nll_source_rejected':True,'source_case_rounds':39,'source_timed_samples':390}
write(BASE/'results181/reference/cohort_replay_validation.json',summary)
print(json.dumps(summary,indent=2))
