"""Verify decode graph execution; profiled timings never replace formal results."""
from pathlib import Path
import hashlib
import json
import sqlite3
import statistics

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/049'
OUT=RUN/'diagnostics'
status=json.loads((RUN/'results/profile_diagnostic.json').read_text());assert status['status']=='complete'
data=json.loads((RUN/'results/measurements/diagnostic_nsys_daso_speed.json').read_text())
db=OUT/'daso_speed_trace.sqlite';connection=sqlite3.connect(str(db));connection.row_factory=sqlite3.Row
graphs=list(connection.execute("select r.start as cpu_start,r.end as cpu_end,g.start as gpu_start,g.end as gpu_end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId join CUPTI_ACTIVITY_KIND_GRAPH_TRACE g on g.correlationId=r.correlationId where s.value like 'cudaGraphLaunch%' order by r.start"))
expected=sum((c['case']['output']-1)*15 for c in data['cases'])
assert expected==4740==len(graphs)
cases=[];offset=0
for case in data['cases']:
 n=case['case']['output']-1;records=graphs[offset:offset+n*15];offset+=n*15;gpu=[];interval=[]
 for index in range(5,15):
  sample=records[index*n:(index+1)*n]
  gpu.append(statistics.mean(v['gpu_end']-v['gpu_start'] for v in sample)/1e6)
  interval.append((sample[-1]['cpu_start']-sample[0]['cpu_start'])/(n-1)/1e6)
 cases.append({'workload':case['case']['label'],'expected_decode_graphs':n*15,
  'profiled_tpot_ms':case['medians']['tpot_ms'],'profiled_graph_gpu_mean_ms':statistics.median(gpu),
  'profiled_inter_graph_cpu_launch_interval_ms':statistics.median(interval)})
assert offset==expected
cpu_samples=[]
for p in (RUN/'logs').glob('cpu_monitor_*.jsonl'):
 for line in p.read_text().splitlines():cpu_samples.extend(s for s in json.loads(line)['samples'] if 'runqueue_fraction' in s)
result={'verified':True,'expected_decode_graphs':expected,'observed_decode_graphs':len(graphs),
 'all_decode_forwards_have_graph_replay_in_this_trace':True,'cases':cases,
 'profiled_timings_excluded_from_formal_results':True,'cpu_scheduling_samples':len(cpu_samples),
 'sampled_main_thread_runqueue_fraction_max':max(s['runqueue_fraction'] for s in cpu_samples),
 'cause_of_formal_timing_variation':'Not uniquely identified. This trace confirms graph replay; it does not establish a causal explanation for the formal run-to-run latency changes.',
 'source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [db,OUT/'daso_speed_trace.nsys-rep',RUN/'results/measurements/diagnostic_nsys_daso_speed.json']}}
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(OUT/'README.md').write_text('''# Decode执行路径短诊断（不作为性能结果）

正式三轮完成后，对同一个DASO速度点、四种核心P+D负载使用Nsight。原数值/计时代码、权重及引擎配置保持一致；采集包含CUDA、OS runtime和Python GIL，所有文件保留。

**确认：4,740个Decode模型前向均对应CUDA graph replay。** 数量与15次生成×(63＋63＋63＋127)完全相等，未发现本次诊断中的Decode退回eager。该结论检查执行结构，不把带profiler的耗时当成正式速度。

正式速度仍有部分点跨轮变化超过5%。当前没有定位唯一原因；不能仅凭GPU未混跑或图已启用就称测速稳定，也不能把Graph之外的开销全部归因于CPU。主线程调度等待采样和Graph/外部流程统计保存在analysis.json，仅作为后续定位线索。

[分析数据与来源SHA](analysis.json) · [原始Nsight](daso_speed_trace.nsys-rep) · [SQLite](daso_speed_trace.sqlite) · [正式双场景结果](../../../responses/reviewer2/qwen_nearby_dual_scenario_run049/README.md)
''')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','cases']},ensure_ascii=False,indent=2))
