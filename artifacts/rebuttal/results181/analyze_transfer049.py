"""Read an existing trace; profiled component times are not formal speeds."""
from pathlib import Path
from bisect import bisect_left, bisect_right
import csv
import hashlib
import json
import sqlite3
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/049'
OUT=BASE/'responses/reviewer1/qwen_phase_transfer_trace';OUT.mkdir(parents=True,exist_ok=True)
db=RUN/'diagnostics/daso_speed_trace.sqlite';data_path=RUN/'results/measurements/diagnostic_nsys_daso_speed.json'
data=json.loads(data_path.read_text());connection=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
assert connection.execute('select label from ENUM_CUDA_MEMCPY_OPER where id=1').fetchone()[0]=='Host-to-Device'
graphs=list(connection.execute("select r.start as cpu_start,g.start,g.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId join CUPTI_ACTIVITY_KIND_GRAPH_TRACE g on g.correlationId=r.correlationId where s.value like 'cudaGraphLaunch%' order by r.start"))
assert len(graphs)==sum((c['case']['output']-1)*15 for c in data['cases'])==4740
copies=list(connection.execute('select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes>=1048576 order by start'))
kernels=list(connection.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
embeddings=list(connection.execute("select k.start,k.end,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value='vectorized_gather_kernel' order by k.start"))
copy_starts=[r['start'] for r in copies];kernel_starts=[r['start'] for r in kernels]
def inside(rows,starts,lower,upper):return [r for r in rows[bisect_left(starts,lower):bisect_right(starts,upper)] if r['end']<=upper]
def merged(rows):
 result=[]
 for r in rows:
  if result and r['start']<=result[-1][1]:result[-1][1]=max(result[-1][1],r['end'])
  else:result.append([r['start'],r['end']])
 return result
def duration(intervals):return sum(b-a for a,b in intervals)
def overlap(a,b):
 i=j=total=0
 while i<len(a) and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return total

samples=[];offset=0
for case in data['cases']:
 spec=case['case'];n=spec['output']-1
 for index in range(5,15):
  lower=graphs[offset+index*n-1]['end'];first=graphs[offset+index*n]['start'];last=graphs[offset+(index+1)*n-1]['end']
  matching=[r for r in embeddings if lower<=r['start']<=first and r['gridX']==spec['batch']*spec['prompt']]
  assert len(matching)==1,(spec['label'],index,len(matching))
  embedding=matching[0]['start'];between=inside(copies,copy_starts,lower,first)
  restore=[r for r in between if r['end']<=embedding];prefetch=[r for r in between if r['start']>=embedding]
  assert len(restore)+len(prefetch)==len(between) and len(restore)==177 and len(prefetch)==288
  restore_union=merged(restore);prefetch_union=merged(prefetch);compute_union=merged(inside(kernels,kernel_starts,embedding,first))
  restore_span=(restore[-1]['end']-restore[0]['start'])/1e6;wall=case['samples'][index]['restore_ms']
  assert 0<=wall-restore_span<10,(spec['label'],index,wall,restore_span)
  steady=inside(copies,copy_starts,first,last);assert not steady
  samples.append({'workload':spec['label'],'batch':spec['batch'],'prompt':spec['prompt'],'output':spec['output'],'iteration':index,
   'restore_large_bytes':sum(r['bytes'] for r in restore),'restore_profiled_wall_ms':wall,'restore_copy_span_ms':restore_span,
   'restore_copy_union_ms':duration(restore_union)/1e6,'prefetch_large_bytes':sum(r['bytes'] for r in prefetch),
   'prefetch_copy_union_ms':duration(prefetch_union)/1e6,'prefetch_span_ms':(prefetch[-1]['end']-prefetch[0]['start'])/1e6,
   'prefetch_overlap_fraction':overlap(prefetch_union,compute_union)/duration(prefetch_union),
   'large_h2d_inside_steady_decode':len(steady)})
 offset+=n*15
assert offset==len(graphs) and len(samples)==40
assert len({r['restore_large_bytes'] for r in samples})==len({r['prefetch_large_bytes'] for r in samples})==1
summary=[]
for case in data['cases']:
 subset=[r for r in samples if r['workload']==case['case']['label']];row={k:subset[0][k] for k in ['workload','batch','prompt','output']}
 for field in samples[0]:
  if field not in row and field!='iteration':row[field]=statistics.median(r[field] for r in subset)
 summary.append(row)
for name,rows in [('samples.csv',samples),('summary.csv',summary)]:
 with (OUT/name).open('w') as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(RUN/'code/arena_runtime.py')==sha(BASE/'runs/053/code/arena_runtime.py')
sources=[db,data_path,RUN/'code/arena_runtime.py',BASE/'runs/053/code/arena_runtime.py',Path(__file__).resolve()]
result={'diagnostic_only':True,'new_gpu_measurements':False,'source_run':'049 Nsight, one DASO speed policy',
 'large_transfer_threshold_bytes':1048576,'profiled_timed_generations':40,'decode_graphs_checked':4740,
 'all_steady_decode_windows_have_no_large_h2d':True,'restore_span_agrees_with_profiled_wall':True,
 'current_arena_runtime_matches_profiled_source':True,'source_sha256':{str(p.relative_to(BASE)):sha(p) for p in sources},'summary':summary}
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# Qwen阶段传输诊断（R1 / R3 / R4）','',
 '**只读复用049的旧Nsight轨迹，不是053新增GPU测量。** 单个DASO速度点、四种负载，每种五预热后十次计时，共40次生成。带profiler的时间仅解释传输结构，不替换正式速度。','',
 '按输入embedding核定位Prefill开始；前一生成最后Decode图结束到本次第一Decode图开始之间，大块H2D按此边界分为恢复Prefill和预取Decode。只统计≥1MiB复制，小scale/token/元数据传输不在大块字节数内。','',
 '| B / 输入 / 输出 | 恢复大块GiB | profile恢复wall ms | 预取大块GiB | 预取设备复制ms | 复制与Prefill相关核重叠 | 稳态Decode大块H2D |',
 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in summary:
 lines.append(f"| {r['batch']} / {r['prompt']} / {r['output']} | {r['restore_large_bytes']/1024**3:.3f} | {r['restore_profiled_wall_ms']:.2f} | {r['prefetch_large_bytes']/1024**3:.3f} | {r['prefetch_copy_union_ms']:.2f} | {100*r['prefetch_overlap_fraction']:.1f}% | {r['large_h2d_inside_steady_decode']:.0f} |")
lines+=['','设备复制时间使用复制区间并集，重叠量使用其与GPU核区间并集的交集；不是把并发区间重复相加。表中比例为逐生成比例的中位数。预取分散在Prefill期间，不能将其复制总时间直接视作额外E2E延迟；时间重叠也不等于端到端零开销，带宽竞争等影响仍体现在实测延迟中。恢复在下一次生成前完成，其成本由正式表的cycle记录。','',
 '全部预热与计时的4,740次Decode前向均有CUDA graph replay；其中40个计时生成的稳态Decode窗口没有≥1MiB H2D。因此，不能将本轨迹的TPOT解释成每个token都重新搬运整套权重。该检查不定位053轮间速度变化的唯一原因，也不证明所有传输都无开销。','',
 '可继续验证的实现方向是显存允许时保留双表示或部分GPU缓存，以减少下一批恢复；需对baseline采用同样资源规则并另测实际显存与速度，本次没有进行该优化实验。','',
 '[逐生成数据](samples.csv) · [汇总CSV](summary.csv) · [诊断与来源SHA](analysis.json) · [解析代码](../../../results181/analyze_transfer049.py) · [原Nsight轨迹](../../../runs/049/diagnostics/daso_speed_trace.nsys-rep) · [原graph诊断](../../../runs/049/diagnostics/README.md) · [053正式速度](../../reviewer2/qwen_night_cohorts_run053/README.md)','']
(OUT/'README.md').write_text('\n'.join(lines));print(json.dumps({'diagnostic_only':True,'new_gpu_measurements':False,'timed_generations':40,'summary':summary},ensure_ascii=False))
