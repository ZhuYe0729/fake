"""Publish every frozen method and workload, including incomplete or unstable timing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

RUN=Path(__file__).resolve().parents[1]
BASE=RUN.parents[1]
OUT=BASE/'responses/reviewer2/qwen_nearby_dual_scenario_run049'
OUT.mkdir(parents=True,exist_ok=True)
DESIGN=json.loads((RUN/'configs/design.json').read_text())
sources={}
def read(p):
 sources[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text())
def save(p,rows):
 with p.open('w') as handle:
  fields=list(dict.fromkeys(key for row in rows for key in row))
  writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def fmt(value,n=3):return '待测' if value=='' else f'{float(value):.{n}f}'


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--tier',type=int,required=True);parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
 for path,digest in read(RUN/'configs/protected.json').items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 rounds=range(len(DESIGN['rounds']));all_rows=[];quality_count=0;speed_count=0;raw_rounds=[]
 cases=[c for c in DESIGN['workloads'] if c['tier']<=args.tier]
 for case in cases:
  w=case['label'];tier=case['tier'];basepath=RUN/f'results/quality/bf16/{w}.json'
  baseline=read(basepath) if basepath.exists() else None
  for spec in DESIGN['rows'][case['scenario']]:
   pid=spec['policy'];path=RUN/f'results/quality/{pid}/{w}.json';raw=read(path) if path.exists() else None
   mean=se=''
   if raw and baseline:
    assert raw['complete'] and baseline['complete'] and len(raw['blocks'])==len(baseline['blocks'])==64
    count=case['prompt']-1 if case['scenario']=='prefill_only' else case['output']
    assert all(r['block']==i and r['token_count']==count for i,r in enumerate(raw['blocks']))
    assert raw['case']==baseline['case']==case
    differences=[r['avg_nll']-b['avg_nll'] for r,b in zip(raw['blocks'],baseline['blocks'])]
    mean=statistics.mean(differences);se=statistics.stdev(differences)/8;quality_count+=1
   stats=[]
   for ri in rounds:
    anchors=[RUN/f'results/measurements/speed_t{tier}_r{ri}_bf16_bracket_{p}.json' for p in ['start','end']]
    own=RUN/f'results/measurements/speed_t{tier}_r{ri}_{pid}.json'
    if not all(p.exists() for p in anchors) or (pid!='bf16' and not own.exists()):continue
    def cell(p):return next(c for c in read(p)['cases'] if c['case']['label']==w)
    aa=[cell(p) for p in anchors]
    actual=cell(own) if pid!='bf16' else None
    keys=list(aa[0]['medians']);ref={k:statistics.mean(c['medians'][k] for c in aa) for k in keys}
    val=actual['medians'] if actual else ref
    row={'workload':w,'method':spec['method'],'policy':pid,'round':ri,**val,
     **{k.replace('_ms','_speedup'):ref[k]/val[k] for k in keys if k!='restore_ms'},
     'bf16_e2e_anchor_drift_fraction':abs(aa[0]['medians']['e2e_ms']-aa[1]['medians']['e2e_ms'])/ref['e2e_ms']}
    stats.append(row);raw_rounds.append(row)
   row={'workload':w,'scenario':case['scenario'],'tier':tier,'batch':case['batch'],'prompt':case['prompt'],'output':case['output'],
    'method':spec['method'],'policy':pid,'delta_nll':mean,'two_se':2*se if se!='' else '',
    'quality_windows':64 if mean!='' else 0,'completed_speed_rounds':len(stats)}
   for key in ['ttft_ms','tpot_ms','e2e_ms','restore_ms','cycle_ms','ttft_speedup','tpot_speedup','e2e_speedup','cycle_speedup']:
    values=[r[key] for r in stats if key in r];row[key]=statistics.median(values) if values else ''
   values=[r['e2e_speedup'] for r in stats]
   row.update(e2e_speedup_min=min(values) if values else '',e2e_speedup_max=max(values) if values else '',
    round_relative_range=(max(values)-min(values))/statistics.median(values) if values else '',
    bf16_anchor_max_drift=max(r['bf16_e2e_anchor_drift_fraction'] for r in stats) if stats else '')
   row['timing_status']='待测' if not stats else '轮数未齐' if len(stats)<3 else '需复核' if row['round_relative_range']>.05 or row['bf16_anchor_max_drift']>.05 else '三轮完成'
   speed_count+=len(stats)==3;all_rows.append(row)
 complete=quality_count==speed_count==len(all_rows)
 if not args.allow_partial:assert complete,(quality_count,speed_count,len(all_rows))
 save(OUT/'comparison.csv',all_rows)
 if raw_rounds:save(OUT/'speed_rounds.csv',raw_rounds)
 lines=['# Qwen邻近负载双场景：DASO与HAWQ-V3（R2 / Exi5）','',
  '**状态：'+('计划内当前批次完成' if complete else '运行中，未完成的格子保留为待测')+'。** 精度为实际ΔNLL±2SE，越低越好；速度统一相对同场景BF16。这里不包含下游任务Avg。','',
  '固定迁移中心主表的均衡/速度优先配置；速度优先是配置名称，不保证每个负载最快。Prefill两方法速度端点是同一map，共用测量；BF16为参考。全部预定点保留。','',
  '[完整CSV](comparison.csv) · [运行设计](../../../runs/049/configs/design.json) · [状态](../../../runs/049/results/worker.json)','']
 for scene,title in [('prefill_only','Prefill-only'),('prefill_decode','Prefill + Decode')]:
  lines+=['## '+title,'']
  for case in cases:
   if case['scenario']!=scene:continue
   header=f"### B={case['batch']} / 输入={case['prompt']}"+(f" / 输出={case['output']}" if scene=='prefill_decode' else '')
   columns=['方法','实测ΔNLL ±2SE ↓','TTFT ×']+(['TPOT ×','E2E ×','完整cycle ×'] if scene=='prefill_decode' else [])+['测速状态']
   lines+=[header,'','| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
   for r in [r for r in all_rows if r['workload']==case['label']]:
    nll='待测' if r['delta_nll']=='' else fmt(r['delta_nll'],5)+' ± '+fmt(r['two_se'],5)
    values=[r['method'],nll,fmt(r['ttft_speedup'])]+([fmt(r[k]) for k in ['tpot_speedup','e2e_speedup','cycle_speedup']] if scene=='prefill_decode' else [])+[r['timing_status']]
    lines.append('| '+' | '.join(values)+' |')
   lines.append('')
 lines+=['## 测量说明','',
  '- 各配置同一引擎：Graph B4/B8/B16，max_model_len=4232、max_num_batched_tokens=32768、KV6GiB；中心输入与048逐token相同，其他输入由同一固定prompt族构造。',
  '- 每策略/轮独立进程，负载顺序与策略顺序预定轮换；5次预热、10次正式计时、3轮全部保留，同GPU/CPU组/NUMA。各轮BF16首尾均值作速度参照，报告三轮速度比中位数；原始绝对耗时与范围另存。',
  '- NLL每负载64个已研究公开WikiText窗口，窗口前缀跨负载共享。Prefill用S−1个prompt token的logprobs，P+D用真实KV、逐token teacher forcing的O个目标token；按实际B运行，与同场景BF16逐窗配对。不作新盲测或硬质量预算保证。',
  '- 测速状态的需复核表示轮间E2E速度比范围或BF16首尾漂移超过5%；只标记，不删除轮次。不以是否对DASO有利来决定重跑或保留。质量统计±2SE与测速范围分开。',
  '- Prefill-only以1个输出token测实际首token回调TTFT；P+D另测TTFT/TPOT/E2E并分列恢复。速度进程不启用NLL日志处理器，NLL计时不当作性能。','']
 (OUT/'README.md').write_text('\n'.join(lines))
 audit={'verified':True,'complete':complete,'tier':args.tier,'display_rows':len(all_rows),'quality_rows':quality_count,
  'three_round_speed_rows':speed_count,'shared_prefill_endpoint':True,'protected_sources_unchanged':True,'source_sha256':sources,
  'unresolved_timing_rows':[{'workload':r['workload'],'method':r['method'],'status':r['timing_status']} for r in all_rows if r['timing_status']=='需复核']}
 (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
