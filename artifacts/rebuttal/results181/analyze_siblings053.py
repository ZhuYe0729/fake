"""Conservative interior timing intervals; passive diagnosis, no CUDA work."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/053';REF=BASE/'results181/reference'
OUT=BASE/'responses/reviewer1/qwen_cpu_sibling_diagnostic';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prefix=Path(read(REF/'cpu_siblings_active.json')['prefix']);raw=prefix.with_suffix('.jsonl');meta=read(prefix.with_suffix('.meta.json'))
assert sha(Path(meta['sampler_entrypoint']))==meta['sampler_sha256']
samples=[json.loads(line) for line in raw.read_text().splitlines()];topology={int(k):set(v) for k,v in meta['topology'].items()}
previous={};records=[];sources={str(raw):sha(raw),str(prefix.with_suffix('.meta.json')):sha(prefix.with_suffix('.meta.json'))}
for row in samples:
 main=row['main_thread']
 if main is None:continue
 forbidden=set().union(*(topology[v] for v in main['affinity']))
 assert not forbidden.intersection(row['sampler_affinity']) and not main['sibling_in_our_affinity']
 row['sample_wall_seconds']=row['time_unix']-previous.get(main['pid'],row['time_unix']);previous[main['pid']]=row['time_unix']
for marker_path in sorted((RUN/'results/accepted').glob('long_speed_*.json')):
 marker=read(marker_path);sources[str(marker_path)]=sha(marker_path)
 for job in marker['jobs']:
  result=read(RUN/job['result']);pid=result['placement']['pid'];label=job['full_label']
  for case in result['cases']:
   path=RUN/f"results/case_attempts/{label}/{case['case']['label']}.json";end=path.stat().st_mtime
   timed=[v for v in case['samples'] if not v['warmup']];start=end-sum(v['cycle_ms'] for v in timed)/1000
   selected=[v for v in samples if v['job']==label and v['main_thread'] is not None and v['main_thread']['pid']==pid and start+1<=v['time_unix']<=end-1]
   if not selected:continue
   sources[str(path)]=sha(path);sources[str(RUN/job['result'])]=sha(RUN/job['result'])
   sibling=[max(v['main_thread']['sibling_busy_fraction'].values(),default=0) for v in selected]
   wait=[v['main_thread']['schedstat_delta'][1]/1e6/v['sample_wall_seconds'] for v in selected if v['main_thread']['schedstat_delta'] is not None and 0<v['sample_wall_seconds']<=3]
   frequency=[v['main_thread']['cpu_khz']/1e6 for v in selected]
   records.append({'group':marker['group'],'round':marker['round'],'role':job['label'],'policy':job['policy'],'gpu':job['gpu'],
    'job':label,'pid':pid,'interior_start':start+1,'interior_end':end-1,'samples':len(selected),
    'sibling_busy_median_percent':100*statistics.median(sibling),'sibling_busy_max_percent':100*max(sibling),
    'main_queue_median_ms_per_wall_second':statistics.median(wait) if wait else '',
    'main_queue_max_ms_per_wall_second':max(wait) if wait else '',
    'main_frequency_median_ghz':statistics.median(frequency),
    'tpot_median_ms':statistics.median(v['tpot_ms'] for v in timed)})
fields=list(records[0]) if records else ['job','samples']
with (OUT/'jobs.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
audit={'diagnostic_only':True,'new_gpu_measurements':False,'sampler_samples':len(samples),'accepted_jobs_sampled':len(records),
 'source_sha256':sources,'analyzer_sha256':sha(Path(__file__)),'sampler_completion':read(prefix.with_suffix('.completion.json')) if prefix.with_suffix('.completion.json').exists() else None}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
lines=['# Decode主线程与SMT兄弟被动诊断','',f"采样{len(samples)}条；覆盖已接受作业{len(records)}个。仅诊断，不增加GPU工作、不替换速度结果。",'',
 '从每case落盘时刻减去十次正式cycle时长构造保守区间，首尾再各收缩1秒。只关联已接受作业的主线程PID；采样器始终位于其亲和物理核之外。',
 'CPU位置可能在两秒采样间迁移；兄弟忙碌和排队来自/proc，不含内核级完整因果轨迹。相关性不能证明波动根因，少量未覆盖作业不能据此排除干扰。','',
 '| 组 / 轮 | 方法 | 样本 | SMT兄弟忙碌 中位/最大% | 主线程排队中位 ms/s | TPOT ms |','|---|---|---:|---:|---:|---:|']
for r in records:lines.append(f"| {r['group']} / {r['round']} | {r['role']} | {r['samples']} | {r['sibling_busy_median_percent']:.2f} / {r['sibling_busy_max_percent']:.2f} | {r['main_queue_median_ms_per_wall_second']:.3f} | {r['tpot_median_ms']:.3f} |")
lines+=['','[逐作业CSV](jobs.csv) · [原始来源/审计](audit.json)','']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({'samples':len(samples),'accepted_jobs':len(records),'max_sibling_percent':max((r['sibling_busy_max_percent'] for r in records),default=None)}))
