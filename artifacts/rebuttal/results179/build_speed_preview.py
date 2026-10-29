"""Join measured task quality with explicitly provisional, previously measured speed."""
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import statistics

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'responses/reviewer2/qwen_original_quality_run050'
OLD=BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049'
read=lambda p:json.loads(p.read_text())
load=lambda p:list(csv.DictReader(p.open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
quality=load(OUT/'quality_table.csv');speed=load(OLD/'comparison.csv');rounds=load(OLD/'speed_rounds.csv')
d49=read(BASE/'runs/049/configs/design.json');d50=read(BASE/'runs/050/configs/design.json')
assert read(OUT/'delivery_audit.json')['final'] and read(OUT/'audit.json')['complete']
assert len(quality)==10 and len(speed)==35 and len(rounds)==105
assert d49['policies']==d50['policies'] and d49['rows']==d50['rows']
qidx={(int(r['batch']),r['method']):r for r in quality}
sidx={(r['workload'],r['method']):r for r in speed}
for row in speed:
 values=[r for r in rounds if r['workload']==row['workload'] and r['method']==row['method']]
 assert len(values)==3 and {int(v['round']) for v in values}=={0,1,2}
 for field in ['ttft_speedup','tpot_speedup','e2e_speedup','cycle_speedup']:
  if row[field]:assert math.isclose(float(row[field]),statistics.median(float(v[field]) for v in values),abs_tol=1e-12)

panels=[('B=8 / 输入=2048 / 输出=64','center_pp','center_pd'),
 ('B=4 / 输入=2048 / 输出=64','batch4_pp','batch4_pd'),
 ('B=8 / 输入=4096 / 输出=64','prompt4096_pp','prompt4096_pd'),
 ('B=8 / 输入=2048 / 输出=128','center_pp','output128_pd')]
rows=[];used=set()
for title,ppid,pdid in panels:
 for method in [r['method'] for r in d50['rows']['prefill_only']]:
  pp=sidx[ppid,method];pd=sidx[pdid,method];q=qidx[int(pd['batch']),method]
  assert pp['batch']==pd['batch'] and pp['prompt']==pd['prompt']
  assert pp['policy']==next(r['policy'] for r in d50['rows']['prefill_only'] if r['method']==method)
  assert pd['policy']==next(r['policy'] for r in d50['rows']['prefill_decode'] if r['method']==method)
  used.update([(ppid,method),(pdid,method)])
  row={'configuration':title,'batch':int(pd['batch']),'prompt':int(pd['prompt']),'output':int(pd['output']),'method':method,
   **{k:float(q[k]) for k in ['arc_c','arc_e','winogrande','mmlu','prefill_avg','cnn_dm','dialogsum','iwslt','pd_avg']},
   'pp_policy':pp['policy'],'pd_policy':pd['policy'],'pp_speed_source':ppid,'pd_speed_source':pdid,
   'pp_ttft_estimate':float(pp['ttft_speedup']),'pd_ttft_estimate':float(pd['ttft_speedup']),
   'pd_tpot_estimate':float(pd['tpot_speedup']),'pd_e2e_estimate':float(pd['e2e_speedup']),
   'pd_cycle_estimate':float(pd['cycle_speedup']),'pd_timing_status':pd['timing_status'],
   'pp_delta_nll':float(pp['delta_nll']),'pd_delta_nll':float(pd['delta_nll']),
   'speed_is_provisional':True,'quality_scope':'Original tasks, max_new_tokens=256; shared by policy/batch, not fixed S/O',
   'speed_scope':'049 three-round medians at fixed S/O, pending quiet-window remeasurement'}
  rows.append(row)
assert len(rows)==20 and used==set(sidx)
with (OUT/'preview.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

comparisons=[]
gain=lambda a,b:100*(float(a)/float(b)-1)
for _,_,workload in panels:
 h=sidx[workload,'HAWQ-V3（速度优先）']
 hr={int(r['round']):float(r['e2e_speedup']) for r in rounds if r['workload']==workload and r['method']==h['method']}
 for method in ['DASO（均衡）','DASO（速度优先）']:
  d=sidx[workload,method];paired=[gain(r['e2e_speedup'],hr[int(r['round'])]) for r in rounds if r['workload']==workload and r['method']==method]
  comparisons.append({'workload':workload,'method':method,'baseline':h['method'],
   'e2e_gain_percent':gain(d['e2e_speedup'],h['e2e_speedup']),
   'cycle_gain_percent':gain(d['cycle_speedup'],h['cycle_speedup']),
   'paired_round_e2e_gain_min_percent':min(paired),'paired_round_e2e_gain_max_percent':max(paired),
   'paired_rounds_faster':sum(x>0 for x in paired),'paired_rounds':3,'timing_status':d['timing_status']})
with (OUT/'preview_comparisons.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)

columns=[('method','方法'),('prefill_avg','Prefill Avg ↑'),('pp_ttft_estimate','暂估Prefill TTFT× ↑'),
 ('pd_avg','P+D Avg ↑'),('pd_ttft_estimate','暂估P+D TTFT× ↑'),('pd_tpot_estimate','暂估TPOT× ↑'),
 ('pd_e2e_estimate','暂估E2E× ↑'),('pd_cycle_estimate','暂估cycle× ↑')]
def display(row,key):
 if key=='method':return row[key]
 if key.endswith('_estimate'):return f"≈{row[key]:.3f}"+('†' if key=='pd_e2e_estimate' and row['pd_timing_status']=='需复核' else '')
 return f'{row[key]:.2f}'

lines=['# Qwen结果总览：实测精度＋暂估速度（R2 / R3）','',
 '**精度已实测完成；速度为暂估。** 速度直接采用049同配置、同负载的三轮实测中位数，尚未在新的空闲时段重测；不是从精度作业耗时推算。所有速度相对同负载BF16，越高越快。','',
 'Avg沿用原完整任务、原prompt、生成上限256，按batch=8/4分别实测。同一batch/map的Avg在多个S/O测速行共用，不代表固定S/O下重新测试了下游任务。†保留旧测速超过5%波动门槛的标记。','',
 '[分组HTML](preview.html) · [完整CSV与NLL](preview.csv) · [逐项实测精度](README.md) · [历史测速详情](../qwen_nearby_dual_scenario_run049/README.md)','',
 '## 当前能否超过HAWQ','',
 '- Prefill均衡点：DASO两个batch的Avg均更高；已有速度暂估比HAWQ均衡点快约37.6%–47.0%。两方法的Prefill速度优先点为同一map，精度与速度打平。',
 '- P+D速度点：DASO的原任务Avg在B8/B4分别高0.20/0.18分；输入2048时，请求E2E点估计快约5.2%–9.8%。输入4096时基本持平，不能宣称全面领先。',
 '- 完整cycle包含下一批Prefill前的恢复：前三种负载DASO速度点暂估慢约1.0%–3.1%，输出128时快约4.3%。请求响应速度优势不能直接写成持续吞吐优势。',
 '- HAWQ均衡点的生成Avg仍更高；固定S/O NLL通常也是HAWQ速度点更低。更高原任务Avg不等于所有质量指标更优。','']
body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen实测精度与暂估速度</title><style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px}p{line-height:1.7}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #d8e0e8;padding:9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.daso{background:#eaf5fc}.hawq{background:#fff9ed}.notice{padding:12px;background:#fff5d9}</style><h1>Qwen：实测精度＋暂估速度</h1><p class="notice">精度来自050原协议完整任务；所有≈速度来自049已有三轮中位数，等待夜间复测。Avg按batch/map共用，并非固定S/O任务精度。†表示旧测速需复核。cycle包含下一批恢复。</p><p><a href="PREVIEW.md">结论、边界与来源</a> · <a href="preview.csv">CSV</a> · <a href="README.md">完整精度分项</a></p>']
for title,_,_ in panels:
 subset=[r for r in rows if r['configuration']==title]
 lines +=[f'## {title}','','| '+' | '.join(label for _,label in columns)+' |','| '+' | '.join(['---']+['---:']*(len(columns)-1))+' |']
 body.append('<h2>'+html.escape(title)+'</h2><div class="scroll"><table><thead><tr><th rowspan="2">方法</th><th colspan="2">Prefill-only</th><th colspan="5">Prefill+Decode</th></tr><tr>'+''.join('<th>'+label+'</th>' for _,label in columns[1:])+'</tr></thead><tbody>')
 for row in subset:
  values=[display(row,key) for key,_ in columns];lines.append('| '+' | '.join(values)+' |')
  cls='daso' if row['method'].startswith('DASO') else 'hawq' if row['method'].startswith('HAWQ') else ''
  body.append('<tr class="'+cls+'">'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
 lines.append('');body.append('</tbody></table></div>')

lines +=['## P+D：相对HAWQ速度点的暂估差异','',
 '正值表示DASO更快。百分比取两个方法各自三轮速度比中位数之比；“三轮快于HAWQ”另外逐轮配对统计，不是置信区间或未来保证。','',
 '| B / 输入 / 输出 | DASO工作点 | E2E速度差异 | 完整cycle速度差异 | 三轮快于HAWQ |',
 '| --- | --- | ---: | ---: | ---: |']
for title,_,workload in panels:
 for c in [c for c in comparisons if c['workload']==workload]:
  mark='†' if c['timing_status']=='需复核' else ''
  e=f"{c['e2e_gain_percent']:+.1f}%" if abs(c['e2e_gain_percent'])>=.05 else '基本持平'
  lines.append(f"| {title} | {c['method']} | ≈{e}{mark} | ≈{c['cycle_gain_percent']:+.1f}% | {c['paired_rounds_faster']}/3 |")
lines +=['','## 使用说明','',
 '- 本页用于当前内部判断，夜间复测前不作为新的正式速度结果；无新GPU计时。所有预定配置和两方法两个点均保留，没有逐负载选取更有利的DASO点。',
 '- 均衡/速度优先是冻结工作点名称，实际质量预算并不相等；不声称分配器在共同实际质量约束下被单独验证更优。Prefill DASO均衡点选中全SQ-INT8，具体后端沿用主表。',
 '- S2048/O128的Prefill数据与S2048/O64是同一份测量；各重复Avg也只对应两种batch的原协议完整任务。固定S/O的64窗ΔNLL保存在CSV，不用Avg替代它。',
 '- 来源：050 quality_table.csv和完整交付审计；049 comparison.csv、speed_rounds.csv及冻结设计。两个campaign的全部policy/map签名已核对一致，旧质量和测速源文件不改写。',
 '- [数值来源与范围审计](preview_audit.json) · [逐轮比较CSV](preview_comparisons.csv)','']
body.append('<p>判断范围与逐轮配对见<a href="PREVIEW.md">Markdown</a>；来源见<a href="preview_audit.json">审计</a>。</p></html>')
(OUT/'PREVIEW.md').write_text('\n'.join(lines));(OUT/'preview.html').write_text(''.join(body))
sources=[OUT/'quality_table.csv',OUT/'audit.json',OUT/'delivery_audit.json',OLD/'comparison.csv',OLD/'speed_rounds.csv',BASE/'runs/049/configs/design.json',BASE/'runs/050/configs/design.json',Path(__file__).resolve()]
audit={'verified':True,'preview_rows':20,'distinct_quality_rows':10,'distinct_source_speed_rows':35,
 'source_speed_rounds':3,'new_speed_measurements':0,'speed_is_provisional':True,'quality_is_fixed_S_O':False,
 'all_policy_map_signatures_equal':True,'source_sha256':{str(p.relative_to(BASE)):sha(p) for p in sources}}
(OUT/'preview_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
