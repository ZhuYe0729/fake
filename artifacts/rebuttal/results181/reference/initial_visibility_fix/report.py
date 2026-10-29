"""Render only complete, resource-checked rounds, independently for every card."""
from pathlib import Path
import csv
import hashlib
import html
import json
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/052'
OUT=BASE/'responses/reviewer2/qwen_night_idle_run052';OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
load=lambda p:list(csv.DictReader(p.open()))
D=read(RUN/'configs/design.json');state=read(RUN/'results/worker.json');sources={}
def source(p):sources[str(p.relative_to(BASE))]=sha(p);return p
def save(name,rows,empty_fields=None):
 fields=list(dict.fromkeys(k for r in rows for k in r)) or empty_fields or ['phase','gpu','workload','method']
 with (OUT/name).open('w') as handle:
  writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
for path,digest in D['source_sha256'].items():assert sha(Path(path))==digest,path
quality=load(source(BASE/'responses/reviewer2/qwen_original_quality_run050/quality_table.csv'))
old=load(source(BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049/comparison.csv'))
qidx={(int(r['batch']),r['method']):r for r in quality};nidx={(r['workload'],r['method']):r for r in old}
accepted=[read(source(p)) for p in (RUN/'results/accepted').glob('*.json')]
def primary(phase,target):
 groups={}
 for m in accepted:
  if m['phase']==phase:groups.setdefault(m['gpu'],[]).append(m)
 winners=[(max(v['finished_unix'] for v in rows),gpu) for gpu,rows in groups.items() if len(rows)==target]
 return min(winners)[1] if winners else None
main_gpu=primary('core_speed',3);long_gpu=primary('long_speed',3);quality_gpu=primary('long_quality',5)
records={};formal_cases=0;quality_cells={}
for marker in accepted:
 if marker['phase']!='long_quality':assert marker['accepted']
 source(RUN/marker['admission'])
 for job in marker['jobs']:
  source(RUN/job['calibration'])
  log=RUN/job['log']
  for path in [log,log.with_suffix('.command.json'),log.with_suffix('.completion.json'),log.with_suffix('.monitor.jsonl')]:source(path)
  file=RUN/job['result'];assert sha(source(file))==job['result_sha256'];data=read(file)
  assert data['complete'] and int(data['gpu'])==job['gpu'] and data['engine']==D['engine']
  assert data['policy_map_sha256']==D['policies'][job['policy']]['map_sha256'] and all(x['exact_equal'] for x in data['weight_audit'].values())
  assert job['completion']['returncode']==0 and job['completion']['resource_interruption'] is None
  if marker['phase']=='long_quality':
   for c in data['cases']:
    p=RUN/c['quality_result'];v=read(source(p));assert v['complete'] and len(v['blocks'])==64
    assert all(x['block']==i and x['token_count']==v['case']['output'] for i,x in enumerate(v['blocks']))
    quality_cells[job['gpu'],job['policy'],c['workload']]=v
  else:
   records[marker['phase'],job['gpu'],job['round'],job['label']]={c['case']['label']:c for c in data['cases']}
   formal_cases+=len(data['cases'])

long_nll=[]
for (gpu,pid,w),data in quality_cells.items():
 baseline=quality_cells.get((gpu,'bf16',w));values=[v['avg_nll'] for v in data['blocks']]
 differences=[a['avg_nll']-b['avg_nll'] for a,b in zip(data['blocks'],baseline['blocks'])] if baseline else []
 long_nll.append({'gpu':gpu,'policy':pid,'workload':w,'windows':64,'mean_nll':statistics.mean(values),
  'delta_nll':statistics.mean(differences) if differences else '',
  'two_se':2*statistics.stdev(differences)/8 if differences else ''})
save('long_nll.csv',long_nll)
nlidx={(r['gpu'],r['policy'],r['workload']):r for r in long_nll}
all_rows=[];round_rows=[];stats_by_key={}
for phase,tier in [('core_speed',1),('long_speed',2)]:
 cards=sorted({m['gpu'] for m in accepted if m['phase']==phase})
 for gpu in cards:
  for case in [c for c in D['workloads'] if c['tier']==tier]:
   w=case['label']
   for spec in D['rows'][case['scenario']]:
    pid=spec['policy'];stats=[]
    for ri in range(3):
     key=lambda label:(phase,gpu,ri,label)
     ids=[key('bf16_bracket_'+v) for v in ['start','end']]
     if not all(k in records for k in ids) or (pid!='bf16' and key(pid) not in records):continue
     anchors=[records[k][w] for k in ids];keys=list(anchors[0]['medians'])
     reference={k:statistics.mean(c['medians'][k] for c in anchors) for k in keys}
     values=reference if pid=='bf16' else records[key(pid)][w]['medians']
     item={'phase':phase,'gpu':gpu,'workload':w,'method':spec['method'],'policy':pid,'round':ri,**values,
      **{k.replace('_ms','_speedup'):reference[k]/values[k] for k in keys if k!='restore_ms'},
      'bf16_anchor_drift':abs(anchors[0]['medians']['e2e_ms']-anchors[1]['medians']['e2e_ms'])/reference['e2e_ms']}
     stats.append(item);round_rows.append(item)
    q=qidx[case['batch'],spec['method']];prior=nidx.get((w,spec['method']))
    fresh=nlidx.get((quality_gpu,pid,w)) if quality_gpu is not None else None
    row={'phase':phase,'gpu':gpu,'workload':w,'scenario':case['scenario'],'batch':case['batch'],'prompt':case['prompt'],'output':case['output'],
     'method':spec['method'],'policy':pid,'original_task_avg':float(q['prefill_avg' if case['scenario']=='prefill_only' else 'pd_avg']),
     'delta_nll':float(prior['delta_nll']) if prior else fresh['delta_nll'] if fresh else '',
     'two_se':float(prior['two_se']) if prior else fresh['two_se'] if fresh else '',
     'nll_source':'049 same-load 64 windows' if prior else f'052 GPU{quality_gpu}, 64 windows' if fresh else 'not yet measured',
     'completed_rounds':len(stats)}
    for field in ['ttft_ms','tpot_ms','e2e_ms','restore_ms','cycle_ms','ttft_speedup','tpot_speedup','e2e_speedup','cycle_speedup']:
     values=[s[field] for s in stats if field in s];row[field]=statistics.median(values) if values else ''
    values=[s['e2e_speedup'] for s in stats]
    row.update(e2e_min=min(values) if values else '',e2e_max=max(values) if values else '',
     relative_round_range=(max(values)-min(values))/statistics.median(values) if values else '',
     max_bf16_anchor_drift=max(s['bf16_anchor_drift'] for s in stats) if stats else '')
    row['status']='待测' if not stats else f'{len(stats)}/3轮' if len(stats)<3 else '需复核' if row['relative_round_range']>.05 or row['max_bf16_anchor_drift']>.05 else '三轮完成'
    all_rows.append(row);stats_by_key[phase,gpu,w,spec['method']]=stats
save('comparison.csv',all_rows);save('speed_rounds.csv',round_rows)
idx={(r['phase'],r['gpu'],r['workload'],r['method']):r for r in all_rows}
panels=[('B=8 / 输入=2048 / 输出=64','center_pp','center_pd'),('B=4 / 输入=2048 / 输出=64','batch4_pp','batch4_pd'),
 ('B=8 / 输入=4096 / 输出=64','prompt4096_pp','prompt4096_pd'),('B=8 / 输入=2048 / 输出=128','center_pp','output128_pd')]
combined=[];long_table=[];paired=[]
for gpu in sorted({r['gpu'] for r in all_rows if r['phase']=='core_speed'}):
 for title,ppw,pdw in panels:
  for spec in D['rows']['prefill_only']:
   method=spec['method'];pp=idx['core_speed',gpu,ppw,method];pd=idx['core_speed',gpu,pdw,method];q=qidx[pd['batch'],method]
   combined.append({'gpu':gpu,'configuration':title,'method':method,**{k:float(q[k]) for k in ['arc_c','arc_e','winogrande','mmlu','prefill_avg','cnn_dm','dialogsum','iwslt','pd_avg']},
    'pp_ttft':pp['ttft_speedup'],'pd_ttft':pd['ttft_speedup'],'pd_tpot':pd['tpot_speedup'],'pd_e2e':pd['e2e_speedup'],
    'pd_cycle':pd['cycle_speedup'],'pp_rounds':pp['completed_rounds'],'pd_rounds':pd['completed_rounds'],'pd_status':pd['status']})
for row in all_rows:
 if row['phase']=='long_speed':long_table.append(row)
 if row['scenario']!='prefill_decode' or not row['method'].startswith('DASO'):continue
 phase=row['phase'];gpu=row['gpu'];w=row['workload'];h=idx[phase,gpu,w,'HAWQ-V3（速度优先）']
 hr={v['round']:v for v in stats_by_key[phase,gpu,w,h['method']]};ds=stats_by_key[phase,gpu,w,row['method']]
 ratios=[v['e2e_speedup']/hr[v['round']]['e2e_speedup'] for v in ds if v['round'] in hr]
 cycles=[v['cycle_speedup']/hr[v['round']]['cycle_speedup'] for v in ds if v['round'] in hr]
 paired.append({'phase':phase,'gpu':gpu,'workload':w,'method':row['method'],'baseline':h['method'],'paired_rounds':len(ratios),
  'e2e_gain_percent':100*(row['e2e_speedup']/h['e2e_speedup']-1),'cycle_gain_percent':100*(row['cycle_speedup']/h['cycle_speedup']-1),
  'median_paired_e2e_gain_percent':100*(statistics.median(ratios)-1),'median_paired_cycle_gain_percent':100*(statistics.median(cycles)-1),
  'faster_rounds':sum(x>1 for x in ratios),'min_paired_gain_percent':100*(min(ratios)-1),'max_paired_gain_percent':100*(max(ratios)-1),'status':row['status']})
save('table2.csv',combined);save('long_table.csv',long_table);save('paired_comparisons.csv',paired)
cols=[('method','方法'),('prefill_avg','Prefill Avg ↑'),('pp_ttft','Prefill TTFT× ↑'),('pd_avg','P+D Avg ↑'),
 ('pd_ttft','P+D TTFT× ↑'),('pd_tpot','TPOT× ↑'),('pd_e2e','E2E× ↑'),('pd_cycle','cycle× ↑')]
def fmt(v,n=3):return '待测' if v=='' or v is None else f'{float(v):.{n}f}'
def cells(row):return [row[k] if k=='method' else fmt(row[k],2 if k.endswith('avg') else 3)+('†' if k=='pd_e2e' and row['pd_status']=='需复核' else '') for k,_ in cols]
core_counts={g:sum(m['phase']=='core_speed' and m['gpu']==g for m in accepted) for g in sorted({m['gpu'] for m in accepted if m['phase']=='core_speed'})}
long_counts={g:sum(m['phase']=='long_speed' and m['gpu']==g for m in accepted) for g in sorted({m['gpu'] for m in accepted if m['phase']=='long_speed'})}
lines=['# Qwen九小时持续空闲复测（R2 / R3）','',
 f"**调度状态：{state['status']}。核心主卡：{'待完整三轮' if main_gpu is None else 'GPU'+str(main_gpu)}；长Decode：{'待完整三轮' if long_gpu is None else 'GPU'+str(long_gpu)}。** 窗口截至2026-10-29 01:12:53 UTC。",'',
 '每30秒观察8卡，应用显存<128MiB、利用率≤5%、无可见进程并连续5分钟才开测。驱动保留显存单列。入场后校准NVML归属差额；可见外部进程或持续显存归属异常则只停止自己的进程，保存整个尝试并重新等空闲。未归属增长本身不能确证外部任务。051未完成首轮不混用。','',
 '下面速度仅来自052完整、资源检查通过的匹配轮次，三轮未齐不作最终排序。各卡自己BF16首尾归一化，不混卡。†为轮间E2E范围或BF16漂移超过5%，只标记不删轮。精度沿用050原任务（上限256），Avg按map/batch共用；固定S/O NLL与Avg分列。','',
 '[分组HTML](table.html) · [核心Table2 CSV](table2.csv) · [全部场景/NLL](comparison.csv) · [逐轮速度](speed_rounds.csv) · [DASO对HAWQ配对](paired_comparisons.csv) · [长Decode NLL](long_nll.csv)','',
 '[实时调度](../../../runs/052/results/worker.json) · [连续空闲状态](../../../runs/052/results/idle_status.json) · [完整协议](../../../runs/052/README.md) · [原任务精度](../qwen_original_quality_run050/README.md)','']
body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen九小时空闲后测速</title><style>body{font:15px system-ui;max-width:1450px;margin:24px auto;padding:0 16px}p{line-height:1.7}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #dce3e9;padding:9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.daso{background:#eaf5fc}.hawq{background:#fff9ed}</style><h1>Qwen：持续空闲后正式测速</h1>',
 '<p>只展示完整匹配轮次，按卡分别归一化；三轮未齐不作最终排序。精度来自原任务，S/O为测速规格。†为超过5%波动门槛。<a href="README.md">协议与状态</a>。</p>']
for gpu in sorted(core_counts,key=lambda g:(g!=main_gpu,g)):
 label=f"GPU{gpu} {'主结果' if gpu==main_gpu else '独立记录'}：{core_counts[gpu]}/3轮"
 lines+=['## '+label,''];body.append('<h2>'+label+'</h2>')
 for title,_,_ in panels:
  lines+=['### '+title,'','| '+' | '.join(v for _,v in cols)+' |','| '+' | '.join(['---']+['---:']*7)+' |']
  body.append('<h3>'+html.escape(title)+'</h3><div class="scroll"><table><thead><tr><th rowspan="2">方法</th><th colspan="2">Prefill-only</th><th colspan="5">Prefill+Decode</th></tr><tr>'+''.join('<th>'+v+'</th>' for _,v in cols[1:])+'</tr></thead><tbody>')
  for row in [r for r in combined if r['gpu']==gpu and r['configuration']==title]:
   values=cells(row);lines.append('| '+' | '.join(values)+' |');cls='daso' if row['method'].startswith('DASO') else 'hawq' if row['method'].startswith('HAWQ') else ''
   body.append('<tr class="'+cls+'">'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
  lines.append('');body.append('</tbody></table></div>')
if not core_counts:lines+=['## 核心速度','','尚无完整、资源检查通过的匹配轮次；当前进度见实时调度，不填旧暂估。','']
lines+=['## 长Decode：输入2048、Batch=8','']
if not long_counts:lines+=['核心完成且余量充足后测输出256/512；尚无完整速度轮次。','']
for gpu in sorted(long_counts,key=lambda g:(g!=long_gpu,g)):
 for w in ['decode256_pd','decode512_pd']:
  heading=f"GPU{gpu} / 输出{w.removeprefix('decode').removesuffix('_pd')}：{long_counts[gpu]}/3轮"
  lines+=['### '+heading,'','| 方法 | 原任务Avg | TTFT× | TPOT× | E2E× | cycle× | 新ΔNLL ±2SE |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
  body.append('<h2>'+heading+'</h2><div class="scroll"><table><tr><th>方法</th><th>原任务Avg</th><th>TTFT×</th><th>TPOT×</th><th>E2E×</th><th>cycle×</th><th>新ΔNLL ±2SE</th></tr>')
  for row in [r for r in long_table if r['gpu']==gpu and r['workload']==w]:
   values=[row['method'],fmt(row['original_task_avg'],2),*[fmt(row[k]) for k in ['ttft_speedup','tpot_speedup','e2e_speedup','cycle_speedup']],fmt(row['delta_nll'],5)+' ± '+fmt(row['two_se'],5) if row['delta_nll']!='' else '待测']
   lines.append('| '+' | '.join(values)+' |');body.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
  lines.append('');body.append('</table></div>')
if quality_gpu is not None:lines +=[f'长Decode新NLL来自GPU{quality_gpu}的64窗完整测量，同卡BF16逐窗配对；任务Avg仍为050原任务分数。','']
lines+=['## 来源与解释范围','',
 '- 核心策略、数值内核、输入、引擎均沿用049；原精度不重跑。长Decode固定输出256/512用于测速，不改原任务的256生成上限。新长NLL仅在余量足够时测，未完成保持待测。',
 '- 每case五预热、十正式计时；绝对耗时和速度比各自取中位数，速度比先按每轮BF16首尾均值归一化。cycle包含恢复下一批Prefill的开销，E2E优势不等于cycle或持续吞吐优势。',
 '- 配对CSV同时保留表中中位加速比之比（gain_percent）和同轮HAWQ耗时/DASO耗时之比的中位数（median_paired_*），另列逐轮范围和获胜轮数；二者统计口径有区别。',
 '- 以最早完成完整三轮的卡为主结果，选择规则不依赖DASO/HAWQ排名；其他设备分列。没有共同实际质量预算或所有场景全局最优的声明。',
 '- [来源与资源检查审计](audit.json) · [冻结设计](../../../runs/052/configs/design.json) · [旧暂估](../qwen_original_quality_run050/PREVIEW.md)','']
(OUT/'README.md').write_text('\n'.join(lines));(OUT/'table.html').write_text(''.join(body)+'</html>')
attempts=[read(p) for p in (RUN/'results/round_attempts').glob('*/round.json')]
for p in (RUN/'results/round_attempts').glob('*/round.json'):source(p)
source(RUN/'configs/design.json');source(RUN/'configs/watch.json');source(RUN/'configs/protocol_sha256.json');source(Path(__file__).resolve())
audit={'verified':True,'complete':state['status']=='complete','controller_status':state['status'],
 'core_primary_gpu':main_gpu,'long_primary_gpu':long_gpu,'long_quality_gpu':quality_gpu,
 'core_rounds_by_gpu':core_counts,'long_rounds_by_gpu':long_counts,'accepted_records':len(accepted),'formal_case_rounds':formal_cases,
 'timed_samples':formal_cases*10,'warmup_samples':formal_cases*5,'scene_rows':len(all_rows),'table2_rows':len(combined),
 'long_quality_cells':len(long_nll),'interrupted_speed_round_attempts':sum(not m['accepted'] for m in attempts),
 'source_sha256':sources,'memory_accounting':'NVMLv2 total - free - reserved','deadline_unix':D['watch']['deadline_unix']}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},ensure_ascii=False))
