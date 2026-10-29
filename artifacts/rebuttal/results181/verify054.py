"""Recompute long NLL CSV directly from captured token log probabilities."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import time

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/054';OUT=BASE/'responses/reviewer2/qwen_long_nll_run054'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
design=read(RUN/'configs/design.json');audit=read(OUT/'audit.json');actual={};gpu={}
for filename,digest in audit['source_sha256'].items():assert sha(Path(filename))==digest
for path in (RUN/'results/completed').glob('*.json'):
 marker=read(path);role=marker['role'];gpu[role]=marker['gpu'];assert marker['completion']['returncode']==0
 value=read(RUN/marker['result']);assert value['gpu_uuid']==marker['uuid'] and value['engine']==design['engine']
 assert value['policy_map_sha256']==design['policies'][marker['policy']]['map_sha256']
 for case in design['workloads']:
  losses=[]
  for i in range(64):
   capture=RUN/f"results/captures/{marker['label']}/{case['label']}/{i:04d}.jsonl"
   tokens=[json.loads(line) for line in capture.read_text().splitlines()]
   assert len(tokens)==case['output'] and all(math.isfinite(v['logprob']) for v in tokens)
   assert [v['step'] for v in tokens]==list(range(case['output']))
   losses.append(-statistics.fmean(v['logprob'] for v in tokens))
  actual[(role,case['label'])]=losses
rows=list(csv.DictReader((OUT/'comparison.csv').open()));expected={(p,c['label']) for p in design['batch_policies'] for c in design['workloads'] if (p,c['label']) in actual}
assert {(r['policy'],r['workload']) for r in rows}==expected and len(rows)==len(expected)
for row in rows:
 values=actual[(row['policy'],row['workload'])]
 assert int(row['gpu'])==gpu[row['policy']] and int(row['windows'])==64
 assert math.isclose(float(row['mean_nll']),statistics.fmean(values),rel_tol=0,abs_tol=1e-12)
 baseline=actual.get(('bf16',row['workload']))
 if baseline is not None:
  differences=[a-b for a,b in zip(values,baseline)]
  assert int(row['bf16_gpu'])==gpu['bf16']
  assert math.isclose(float(row['delta_nll']),statistics.fmean(differences),rel_tol=0,abs_tol=1e-12)
  assert math.isclose(float(row['delta_2se']),statistics.stdev(differences)/4,rel_tol=0,abs_tol=1e-12)
 else:assert row['delta_nll']==row['delta_2se']==''
full=len(expected)==10;assert audit['main_complete']==full and audit['main_points']==len(expected)
for case,check in audit['baseline_repetition'].items():
 maximum=max(abs(a-b) for a,b in zip(actual[('bf16',case)],actual[('bf16_repeat',case)]))
 assert abs(check['maximum_window_abs_difference']-maximum)<1e-12
 assert check['within_tolerance']==(maximum<=design['baseline_replication_tolerance'])
result={'verified':True,'main_complete':full,'main_points':len(expected),'all_window_records':sum(map(len,actual.values())),
 'time_unix':time.time(),'report_audit_sha256':sha(OUT/'audit.json'),'verifier_sha256':sha(Path(__file__)),
 'comparison_sha256':sha(OUT/'comparison.csv'),'baseline_repetition':audit['baseline_repetition']}
(OUT/'delivery_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result))
