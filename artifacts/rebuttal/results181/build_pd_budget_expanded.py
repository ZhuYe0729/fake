"""Recompute every predeclared run003 workload-adapted P+D budget check."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/003';OUT=BASE/'responses/reviewer1/pd_quality_budget_expanded'
OUT.mkdir(parents=True,exist_ok=True);read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sources={}
def tracked(p):sources[str(p.resolve())]=sha(p);return p
design=read(tracked(RUN/'configs/design.json'));frozen=read(tracked(RUN/'configs/evaluation_points.json'))
assert frozen['complete_design'] and frozen['design_sha256']==sha(RUN/'configs/design.json')
workloads=[v for v in design['workloads'] if v['scenario']=='prefill_decode'];assert len(workloads)==9 and design['quality_budgets']==[.01,.07]
points=[v for v in frozen['points'] if v['family']=='DASO workload adapted' and v['workload'] in {w['id'] for w in workloads}]
assert len(points)==18 and {(p['workload'],p['quality_budget']) for p in points}=={(w['id'],e) for w in workloads for e in design['quality_budgets']}
summary=list(csv.DictReader(tracked(RUN/'results/summary_final.csv').open()));rows=[];maps=set();samples=[]
fit=read(tracked(RUN/'reference/quality_model_prefill_decode.json'))
errors={(v['module'],v['phase'],v['method']):float(v['output_rel_mse']) for v in csv.DictReader(tracked(RUN/'data/local_errors.csv').open())}
kinds=['qkv_proj','o_proj','gate_up_proj','down_proj']
for point in points:
 policy=point['policy_id'];workload=next(w for w in workloads if w['id']==point['workload']);output=workload['output']
 mapping=read(tracked(RUN/f'configs/policies/{policy}.json'))['method_map'];maps.add(hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest())
 predicted=fit['bias']
 for name,pair in mapping.items():
  bucket=int(name.split('.')[2])*4//36;kind=kinds.index(name.rsplit('.',1)[-1])
  for i,phase in enumerate(['prefill','decode']):
   method=pair[phase+'_method']
   if method!='dense_bf16':predicted+=errors[name,phase,method]*fit['coefficients'][i][design['methods'].index(method)][bucket][kind]/fit['feature_scale']
 assert abs(predicted-point['predicted_delta_nll'])<1e-12 and point['predicted_delta_nll']<=point['quality_budget']+1e-12
 raw=read(tracked(RUN/f"results/quality/{policy}/{workload['id']}.json"));baseline=read(tracked(RUN/f"results/quality/uniform_dense_bf16/{workload['id']}.json"))
 assert raw['workload']==baseline['workload']==workload and raw['decode_score_batch']==baseline['decode_score_batch']==workload['batch']
 assert len(raw['blocks'])==len(baseline['blocks'])==100 and raw['phase_events']['enter_decode']>0 and raw['phase_events']['apply_decode']>0
 delta=[]
 for i,(a,b) in enumerate(zip(raw['blocks'],baseline['blocks'],strict=True)):
  assert a['block']==b['block']==i and a['token_count']==b['token_count']==output
  assert math.isfinite(a['avg_nll']) and math.isfinite(b['avg_nll'])
  delta.append(a['avg_nll']-b['avg_nll']);samples.append({'label':point['label'],'workload':workload['id'],'budget':point['quality_budget'],'block':i,'delta_nll':delta[-1]})
 mean=statistics.mean(delta);se=statistics.stdev(delta)/10
 assert abs(mean-(raw['avg_nll']-baseline['avg_nll']))<1e-12
 src=next(r for r in summary if r['label']==point['label']);assert src['policy_id']==policy
 assert abs(float(src['measured_delta_nll'])-mean)<1e-12 and abs(float(src['predicted_delta_nll'])-point['predicted_delta_nll'])<1e-12
 rows.append({'label':point['label'],'policy':policy,'workload':workload['id'],'batch':workload['batch'],'input':workload['input'],'output':output,
  'budget':point['quality_budget'],'predicted_delta_nll':point['predicted_delta_nll'],'measured_mean_delta_nll':mean,'paired_se':se,
  'test_windows':100,'mean_violation':max(0.,mean-point['quality_budget']),'mean_satisfied':mean<=point['quality_budget'],
  'ci_low':float(src['delta_nll_ci_low']),'ci_high':float(src['delta_nll_ci_high'])})
for filename,data in [('points.csv',rows),('window_deltas.csv',samples)]:
 with (OUT/filename).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
by_budget={}
for budget in design['quality_budgets']:
 selected=[r for r in rows if r['budget']==budget];by_budget[str(budget)]={'checks':len(selected),'violations':sum(not r['mean_satisfied'] for r in selected),
  'mae':statistics.mean(abs(r['measured_mean_delta_nll']-r['predicted_delta_nll']) for r in selected)}
violations=[r for r in rows if not r['mean_satisfied']];assert len(violations)==1
audit={'verified':True,'new_measurements':False,'source_run':'003','solver':'original DP','family':'DASO workload adapted',
 'all_predeclared_pd_checks_included':True,'settings':len(rows),'unique_policy_maps':len(maps),'workloads':len(workloads),'budgets':design['quality_budgets'],
 'satisfied':len(rows)-len(violations),'violations':len(violations),'satisfaction_rate':(len(rows)-len(violations))/len(rows),
 'by_budget':by_budget,'violating_settings':violations,'window_pairs_recomputed':len(samples),
 'source_sha256':sources,'builder_sha256':sha(Path(__file__)),'points_sha256':sha(OUT/'points.csv')}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
bad=violations[0];lines=['# R1：P+D多负载预算满足率（既有完整18设定）','',
 '统计范围为Qwen3-8B、Run003冻结的全部DASO workload adapted P+D设定：9种负载×ε=0.01/0.07两档预算，使用原DP七候选扩展，未作额外validation筛选。所有预设点均纳入，不按是否超限筛选。','',
 f"共18个预算—负载检查，17/18（94.4%）测试平均ΔNLL满足预算，1/18（5.6%）超限；对应{len(maps)}套唯一method-map。每点100个固定WikiText测试窗口，两个预算/不同负载复用文本，并非18次独立随机试验。",'',
 'ε=0.01：8/9满足、1/9超限；ε=0.07：9/9满足、0/9超限。只能描述这组预设网格的分布，不据此证明一般违约概率随预算收紧而上升。','',
 f"唯一超限：B={bad['batch']}、输入={bad['input']}、输出={bad['output']}，ε={bad['budget']}，预测ΔNLL={bad['predicted_delta_nll']:.8f}，实测平均ΔNLL={bad['measured_mean_delta_nll']:.8f}，超限{bad['mean_violation']:.8f} nats/token。配对SE={bad['paired_se']:.8f}，原描述性95%配对bootstrap区间[{bad['ci_low']:.8f}, {bad['ci_high']:.8f}]。",'',
 '质量模型保持中心负载校准参数不变；各负载更新动作延迟并重新分配，检验冻结质量代理的跨B/S/O迁移。模型并未在这些测试结果上重拟合。本次只独立重算历史NLL，不使用旧速度来替代最近053正式速度。','',
 f"选点预测MAE：ε=0.01为{by_budget['0.01']['mae']:.5f}，ε=0.07为{by_budget['0.07']['mae']:.5f}。较宽预算存在保守预测，不声称每个点均预测精确；均值满足率不等于单请求保证或下游任务等效。",'',
 '[全部18点与区间](points.csv) · [逐窗ΔNLL](window_deltas.csv) · [冻结选择/参数重算/原始来源SHA](audit.json) · [原始完整实验](../qwen3_run003/README.md)','']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in audit.items() if k not in ['source_sha256','violating_settings']},ensure_ascii=False))
