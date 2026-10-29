"""Publish per-device new timing; reuse separately measured, frozen quality."""
from pathlib import Path
import csv
import hashlib
import html
import json
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/051'
OUT=BASE/'responses/reviewer2/qwen_quiet_speed_run051';OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
load=lambda p:list(csv.DictReader(p.open()))
DESIGN=read(RUN/'configs/design.json')
sources={}
def source(p):sources[str(p.relative_to(BASE))]=sha(p);return p
def save(name,rows):
 if not rows:
  if name=='speed_rounds.csv':(OUT/name).write_text('gpu,workload,method,policy,round\n')
  return
 with (OUT/name).open('w') as handle:
  writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
for path,digest in DESIGN['source_sha256'].items():assert sha(Path(path))==digest,path
quality=load(source(BASE/'responses/reviewer2/qwen_original_quality_run050/quality_table.csv'))
nll=load(source(BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049/comparison.csv'))
qidx={(int(r['batch']),r['method']):r for r in quality};nidx={(r['workload'],r['method']):r for r in nll}
markers={p.stem:read(source(p)) for p in (RUN/'results/completed').glob('*.json')}
records={};formal_cases=0
for ident,m in markers.items():
 p=RUN/m['result'];assert sha(source(p))==m['result_sha256'];data=read(p)
 assert data['complete'] and data['mode']=='speed' and not m['completion']['foreign_pids'] and m['completion']['returncode']==0
 assert int(data['gpu'])==m['gpu'] and data['gpu_uuid']==m['gpu_uuid'] and data['round']==m['round']
 assert data['engine']==DESIGN['engine'] and data['policy_map_sha256']==DESIGN['policies'][m['policy']]['map_sha256']
 assert all(v['exact_equal'] for v in data['weight_audit'].values())
 records[ident]={c['case']['label']:c for c in data['cases']};formal_cases+=len(data['cases'])

all_rows=[];round_rows=[];stats_by_key={}
for gpu in DESIGN['cards']:
 for case in DESIGN['workloads']:
  w=case['label']
  for spec in DESIGN['rows'][case['scenario']]:
   pid=spec['policy'];stats=[]
   for ri in range(3):
    anchor_ids=[f'g{gpu}_r{ri}_bf16_bracket_{part}' for part in ['start','end']]
    own_id=f'g{gpu}_r{ri}_{pid}'
    if not all(k in records for k in anchor_ids) or (pid!='bf16' and own_id not in records):continue
    anchors=[records[k][w] for k in anchor_ids];keys=list(anchors[0]['medians'])
    reference={k:statistics.mean(c['medians'][k] for c in anchors) for k in keys}
    values=reference if pid=='bf16' else records[own_id][w]['medians']
    item={'gpu':gpu,'workload':w,'method':spec['method'],'policy':pid,'round':ri,**values,
     **{k.replace('_ms','_speedup'):reference[k]/values[k] for k in keys if k!='restore_ms'},
     'bf16_anchor_drift':abs(anchors[0]['medians']['e2e_ms']-anchors[1]['medians']['e2e_ms'])/reference['e2e_ms']}
    stats.append(item);round_rows.append(item)
   old=nidx[w,spec['method']];q=qidx[case['batch'],spec['method']]
   row={'gpu':gpu,'workload':w,'scenario':case['scenario'],'batch':case['batch'],'prompt':case['prompt'],'output':case['output'],
    'method':spec['method'],'policy':pid,'original_task_avg':float(q['prefill_avg' if case['scenario']=='prefill_only' else 'pd_avg']),
    'prior_delta_nll':float(old['delta_nll']),'prior_two_se':float(old['two_se']),'completed_rounds':len(stats)}
   for key in ['ttft_ms','tpot_ms','e2e_ms','restore_ms','cycle_ms','ttft_speedup','tpot_speedup','e2e_speedup','cycle_speedup']:
    values=[s[key] for s in stats if key in s];row[key]=statistics.median(values) if values else ''
   values=[s['e2e_speedup'] for s in stats]
   row.update(e2e_min=min(values) if values else '',e2e_max=max(values) if values else '',
    relative_round_range=(max(values)-min(values))/statistics.median(values) if values else '',
    max_bf16_anchor_drift=max(s['bf16_anchor_drift'] for s in stats) if stats else '')
   row['status']='待测' if not stats else f'{len(stats)}/3轮' if len(stats)<3 else '需复核' if row['relative_round_range']>.05 or row['max_bf16_anchor_drift']>.05 else '三轮完成'
   all_rows.append(row);stats_by_key[gpu,w,spec['method']]=stats
save('comparison.csv',all_rows);save('speed_rounds.csv',round_rows)
idx={(r['gpu'],r['workload'],r['method']):r for r in all_rows}
panels=[('B=8 / 输入=2048 / 输出=64','center_pp','center_pd'),('B=4 / 输入=2048 / 输出=64','batch4_pp','batch4_pd'),
 ('B=8 / 输入=4096 / 输出=64','prompt4096_pp','prompt4096_pd'),('B=8 / 输入=2048 / 输出=128','center_pp','output128_pd')]
combined=[]
for gpu in DESIGN['cards']:
 for title,ppw,pdw in panels:
  for spec in DESIGN['rows']['prefill_only']:
   method=spec['method'];pp=idx[gpu,ppw,method];pd=idx[gpu,pdw,method];q=qidx[pd['batch'],method]
   combined.append({'gpu':gpu,'configuration':title,'batch':pd['batch'],'prompt':pd['prompt'],'output':pd['output'],'method':method,
    **{k:float(q[k]) for k in ['arc_c','arc_e','winogrande','mmlu','prefill_avg','cnn_dm','dialogsum','iwslt','pd_avg']},
    'pp_ttft':pp['ttft_speedup'],'pd_ttft':pd['ttft_speedup'],'pd_tpot':pd['tpot_speedup'],'pd_e2e':pd['e2e_speedup'],
    'pd_cycle':pd['cycle_speedup'],'pp_rounds':pp['completed_rounds'],'pd_rounds':pd['completed_rounds'],
    'pp_status':pp['status'],'pd_status':pd['status']})
save('table2.csv',combined)
paired=[]
for gpu in DESIGN['cards']:
 for _,_,w in panels:
  h=idx[gpu,w,'HAWQ-V3（速度优先）'];hr={s['round']:s for s in stats_by_key[gpu,w,h['method']]}
  for method in ['DASO（均衡）','DASO（速度优先）']:
   d=idx[gpu,w,method];ds=[s for s in stats_by_key[gpu,w,method] if s['round'] in hr]
   values=[s['e2e_speedup']/hr[s['round']]['e2e_speedup'] for s in ds]
   paired.append({'gpu':gpu,'workload':w,'method':method,'baseline':h['method'],'paired_rounds':len(ds),
    'e2e_gain_percent':100*(d['e2e_speedup']/h['e2e_speedup']-1) if ds else '',
    'cycle_gain_percent':100*(d['cycle_speedup']/h['cycle_speedup']-1) if ds else '',
    'faster_rounds':sum(v>1 for v in values),'min_paired_e2e_gain_percent':100*(min(values)-1) if values else '',
    'max_paired_e2e_gain_percent':100*(max(values)-1) if values else '', 'status':d['status']})
save('paired_comparisons.csv',paired)
complete=len(markers)==54 and all(r['completed_rounds']==3 for r in all_rows)
columns=[('method','方法'),('prefill_avg','Prefill Avg ↑'),('pp_ttft','Prefill TTFT× ↑'),('pd_avg','P+D Avg ↑'),
 ('pd_ttft','P+D TTFT× ↑'),('pd_tpot','TPOT× ↑'),('pd_e2e','E2E× ↑'),('pd_cycle','cycle× ↑')]
def fmt(row,key):
 if key=='method':return row[key]
 if row[key]=='':return '待测'
 return format(float(row[key]),'.2f' if key.endswith('avg') else '.3f')+('†' if key=='pd_e2e' and row['pd_status']=='需复核' else '')
lines=['# Qwen正式速度复测＋实测精度（R2 / R3）','',
 f"**状态：{'全部完成' if complete else '进行中'}，{len(markers)}/54个正式进程完成。** GPU3为主结果，GPU4独立复核，分卡归一化，不合并或挑选更有利的卡。",'',
 '速度来自051新测，空格不沿用旧暂估。每卡3轮，每case 5预热/10计时；轮数未齐时只展示当前结果，不作最终排序。†为E2E轮间范围或BF16首尾漂移超过5%，全部轮次保留。','',
 '精度沿用050完整原任务（最大生成256）及049固定S/O NLL，本轮不重跑精度。Avg按map/batch共用，测速S/O不构成新的任务数据集。PP两方法速度端点为同map，一次测量两行展示。','',
 '[分组HTML](table.html) · [40行Table2 CSV](table2.csv) · [70行场景CSV与NLL](comparison.csv) · [逐轮速度](speed_rounds.csv) · [逐轮配对](paired_comparisons.csv) · [实时状态](../../../runs/051/results/worker.json)','']
body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen正式速度复测</title><style>body{font:15px system-ui;max-width:1450px;margin:25px auto;padding:0 16px}p{line-height:1.7}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #dce3e9;padding:9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.daso{background:#eaf5fc}.hawq{background:#fff9ed}</style><h1>Qwen：新速度＋原协议实测精度</h1>',
 f'<p>进度{len(markers)}/54。GPU3主结果、GPU4独立复核；每卡3轮，按各自BF16归一化。精度沿用050原任务，S/O为测速规格。未完成速度不填旧值；†表示需复核。</p><p><a href="README.md">协议与比较</a> · <a href="table2.csv">CSV</a></p>']
for gpu in DESIGN['cards']:
 count=sum(f'g{gpu}_r{ri}_bf16_bracket_end' in markers for ri in range(3))
 if gpu!=DESIGN['primary_gpu']:lines+=['<details>','<summary>GPU4独立复核表（点击展开）</summary>','']
 lines +=[f"## GPU{gpu} {'主结果' if gpu==3 else '独立复核'}：完整{count}/3轮",'']
 body.append(f'<h2>GPU{gpu}：完整{count}/3轮</h2>')
 for title,_,_ in panels:
  lines +=['### '+title,'','| '+' | '.join(label for _,label in columns)+' |','| '+' | '.join(['---']+['---:']*7)+' |']
  body.append('<h3>'+html.escape(title)+'</h3><div class="scroll"><table><thead><tr><th rowspan="2">方法</th><th colspan="2">Prefill-only</th><th colspan="5">Prefill+Decode</th></tr><tr>'+''.join('<th>'+label+'</th>' for _,label in columns[1:])+'</tr></thead><tbody>')
  for row in [r for r in combined if r['gpu']==gpu and r['configuration']==title]:
   values=[fmt(row,k) for k,_ in columns];lines.append('| '+' | '.join(values)+' |')
   cls='daso' if row['method'].startswith('DASO') else 'hawq' if row['method'].startswith('HAWQ') else ''
   body.append('<tr class="'+cls+'">'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
  lines.append('');body.append('</tbody></table></div>')
 if gpu!=DESIGN['primary_gpu']:lines+=['</details>','']
lines +=['## 计时与来源','',
 '- 正式进程串行，GPU3先三轮、GPU4再三轮；各卡固定CPU组与NUMA，Torch/NVML UUID一致。卡上外部进程、资源失败的尝试另外保留，不与无外部干扰的完成标记混用；任何计时数值都不触发删轮。',
 '- 每轮BF16首尾均值作参照，再取三个速度比的中位数。绝对耗时中位数与速度比中位数分别计算，不能用汇总绝对耗时之比替代逐轮归一化。恢复耗时与完整cycle单独保留。',
 '- 同049的Graph B4/B8/B16、上下文4232、token预算32768、KV6GiB。输入、数值内核、checkpoint/map均冻结；正式计时不包含初始化、Graph捕获、编译或权重审计。',
 '- 050四项MC和三项生成任务使用原prompt/样本，按B8/B4测量；固定S/O ΔNLL另用049原64窗实测值。GPU4只复核速度，不声称每张卡重新测了精度。',
 '- HAWQ沿用主表PTQ适配；均衡/速度优先是既有工作点名称，非每负载全局最优或共同实际质量约束。Prefill DASO均衡点为全SQ-INT8；所有预定点均展示。',
 '- [冻结设计](../../../runs/051/configs/design.json) · [运行说明](../../../runs/051/README.md) · [原精度](../qwen_original_quality_run050/README.md) · [旧暂估](../qwen_original_quality_run050/PREVIEW.md) · [来源审计](audit.json)','']
(OUT/'README.md').write_text('\n'.join(lines));(OUT/'table.html').write_text(''.join(body)+'</html>')
source(Path(__file__).resolve());source(RUN/'configs/design.json');source(RUN/'configs/protocol_sha256.json')
audit={'verified':True,'complete':complete,'completed_processes':len(markers),'planned_processes':54,
 'formal_case_rounds':formal_cases,'timed_samples':formal_cases*10,'warmup_samples':formal_cases*5,
 'display_scene_rows':70,'combined_table_rows':40,'source_sha256':sources,'primary_gpu':3,'validation_gpu':4,
 'new_quality_measurements':0,'per_device_summary':{str(g):{'complete_rows':sum(r['completed_rounds']==3 for r in all_rows if r['gpu']==g),
 'unstable_rows':[{'workload':r['workload'],'method':r['method']} for r in all_rows if r['gpu']==g and r['status']=='需复核']} for g in DESIGN['cards']}}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ['source_sha256','per_device_summary']},ensure_ascii=False))
