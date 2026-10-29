"""Accuracy-only report; never interpret parallel quality wall times as speed."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/054';SOURCE=BASE/'runs/053'
OUT=BASE/'responses/reviewer2/qwen_long_nll_run054';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
design=read(RUN/'configs/design.json');state=read(RUN/'results/worker.json');sources={}
def track(p):
 sources[str(p.resolve())]=sha(p);return p
track(RUN/'configs/design.json');track(RUN/'configs/protocol_sha256.json')
manifest=read(RUN/'configs/protocol_sha256.json');version=sha(RUN/'configs/protocol_sha256.json')[:16]
for name,digest in manifest.items():
 for root in [RUN/'code',RUN/'code_versions'/version,SOURCE/'code']:
  assert sha(track(root/name))==digest
for path,digest in design['source_sha256'].items():assert sha(track(Path(path)))==digest
for path,digest in read(RUN/'reference/input_sha256.json').items():assert sha(track(Path(path)))==digest
track(RUN/'reference/checkpoint_reuse.json')
for spec in read(RUN/'reference/checkpoint_reuse.json').values():
 a=Path(spec['source']).stat();b=Path(spec['target']).stat()
 assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)==(spec['device'],spec['inode'],spec['bytes'])
 assert sha(track(Path(spec['target']).parent/'config.json'))==spec['config_sha256']
windows={};jobs={}
for marker_path in sorted((RUN/'results/completed').glob('*.json')):
 marker=read(track(marker_path));role=marker['role'];policy=marker['policy'];jobs[role]=marker
 assert marker['completion']['returncode']==0 and marker['completion']['accuracy_only']
 result=read(track(RUN/marker['result']));assert sha(RUN/marker['result'])==marker['result_sha256']
 assert result['complete'] and result['mode']=='quality' and result['tier']==2
 assert result['policy']==policy and result['gpu_uuid']==marker['uuid'] and int(result['gpu'])==marker['gpu']
 assert result['engine']==design['engine'] and result['policy_map_sha256']==design['policies'][policy]['map_sha256']
 assert all(v['exact_equal'] for v in result['weight_audit'].values())
 assert marker['code_sha256']==manifest and marker['code_version']==version
 admission=read(track(RUN/marker['admission']));assert admission['selected']['gpu']==marker['gpu']
 samples=admission['samples'];assert samples[-1]['time']-samples[0]['time']>=90
 assert all(v['used_bytes']<128*1024**2 and v['utilization_percent']<=5 and not v['visible_processes'] for v in samples)
 assert all(b['time']-a['time']<=45 for a,b in zip(samples,samples[1:]))
 log=RUN/marker['log']
 for p in [log,log.with_suffix('.command.json'),log.with_suffix('.completion.json'),log.with_suffix('.monitor.jsonl')]:
  if p.exists():track(p)
 assert {v['workload'] for v in result['cases']}=={v['label'] for v in design['workloads']}
 for item in result['cases']:
  raw=read(track(RUN/item['quality_result']));case=next(c for c in design['workloads'] if c['label']==item['workload'])
  assert raw['complete'] and raw['case']==case and raw['policy']==policy and len(raw['blocks'])==64
  losses=[]
  for i,block in enumerate(raw['blocks']):
   assert block['block']==i and block['token_count']==case['output']
   capture=RUN/f"results/captures/{marker['label']}/{case['label']}/{i:04d}.jsonl"
   token_rows=[json.loads(line) for line in track(capture).read_text().splitlines()]
   assert len(token_rows)==case['output']
   assert [v['step'] for v in token_rows]==list(range(case['output']))
   assert all(math.isfinite(v['logprob']) for v in token_rows)
   values=-sum(v['logprob'] for v in token_rows)/len(token_rows)
   assert math.isclose(values,block['avg_nll'],rel_tol=0,abs_tol=1e-12)
   losses.append(values)
  windows[(role,case['label'])]=losses
dispatch=RUN/'results/dispatch.json'
if dispatch.exists():
 record=read(track(dispatch));assert record['code_version']==version
 assert sha(track(Path(record['controller'])))==record['controller_sha256']
 speed_audit=BASE/'responses/reviewer2/qwen_night_cohorts_run053/long_speed_coverage_delivery_audit.json'
 assert read(track(speed_audit))['verified']
 repeat_checks={}
else:repeat_checks={}
rows=[];paired=[]
for case in design['workloads']:
 baseline=windows.get(('bf16',case['label']))
 repeat=windows.get(('bf16_repeat',case['label']))
 if baseline is not None and repeat is not None:
  maximum=max(abs(a-b) for a,b in zip(baseline,repeat))
  repeat_checks[case['label']]={'maximum_window_abs_difference':maximum,'tolerance':design['baseline_replication_tolerance'],'within_tolerance':maximum<=design['baseline_replication_tolerance']}
 for policy in design['batch_policies']:
  values=windows.get((policy,case['label']))
  if values is None:continue
  differences=[a-b for a,b in zip(values,baseline)] if baseline is not None else None
  row={'workload':case['label'],'method':design['policies'][policy]['display'],'policy':policy,'gpu':jobs[policy]['gpu'],
   'bf16_gpu':jobs['bf16']['gpu'] if baseline is not None else '', 'windows':64,'output':case['output'],'mean_nll':statistics.mean(values),
   'delta_nll':statistics.mean(differences) if differences is not None else '',
   'delta_2se':2*statistics.stdev(differences)/8 if differences is not None else ''}
  rows.append(row)
  for i,value in enumerate(values):paired.append({'workload':case['label'],'policy':policy,'block':i,'nll':value,'bf16_nll':baseline[i] if baseline is not None else ''})
for name,data,fields in [('comparison.csv',rows,['workload','method','policy','gpu','bf16_gpu','windows','output','mean_nll','delta_nll','delta_2se']),('windows.csv',paired,['workload','policy','block','nll','bf16_nll'])]:
 with (OUT/name).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
full=all((p,c['label']) in windows for p in design['batch_policies'] for c in design['workloads'])
audit={'state':state,'main_complete':full,'main_jobs':len(set(jobs)&set(design['batch_policies'])),'main_points':len(rows),
 'all_window_records':sum(map(len,windows.values())),'baseline_repetition':repeat_checks,'source_sha256':sources,'report_sha256':sha(Path(__file__))}
write(OUT/'audit.json',audit)
lines=['# Qwen 长Decode NLL（独立并行精度补测）','',f"状态：{state['status']}；五策略完整：{full}；已完成主表点：{len(rows)}/10。",'',
 'B=8、输入2048、输出256/512，各64个既有WikiText窗口；与053固定引擎、权重、策略和teacher forcing实现一致。只测NLL，所有并行作业耗时均不作为速度。',
 '每点卡号及共同BF16参考卡明示。ΔNLL是同窗配对均值，±为2SE；这些是已研究窗口，不是新的盲测，也不构成新质量预算保证。',
 '若第六张卡可用，额外BF16用于检查跨卡数值重复，容差在派发前固定为每窗1e-4。它不增加独立样本量；完整结果和检查失败均保留。','',
 '| 输出 | 方法 | GPU / BF参考GPU | NLL | ΔNLL ± 2SE |','|---|---|---|---:|---:|']
for row in rows:
 delta=f"{row['delta_nll']:+.5f} ± {row['delta_2se']:.5f}" if row['delta_nll']!='' else '待BF参考'
 lines.append(f"| {row['output']} | {row['method']} | {row['gpu']} / {row['bf16_gpu']} | {row['mean_nll']:.5f} | {delta} |")
lines+=['',f'跨卡BF16检查：`{json.dumps(repeat_checks,ensure_ascii=False)}`。','',
 '[CSV](comparison.csv) · [逐窗数据](windows.csv) · [来源审计](audit.json) · [独立核验](delivery_audit.json)','']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({'main_complete':full,'main_points':len(rows),'repeat':repeat_checks}))
