"""Audit existing real mixed forwards; PD separation remains a proposed test."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

BASE=Path(__file__).resolve().parents[1];RUN=BASE/'runs/047';SOURCE=BASE/'responses/reviewer2/qwen_updated_mixed_run047'
OUT=BASE/'responses/reviewer2/mixed_pd_compatibility';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sources={}
def tracked(p):sources[str(p.resolve())]=sha(p);return p
design=read(tracked(RUN/'configs/design.json'));prior=read(tracked(SOURCE/'provenance.json'))
assert prior['verified'] and prior['complete'] and prior['maps']==9 and prior['cases']==2
for path,digest in prior['source_sha256'].items():assert sha(Path(path))==digest
results=list(csv.DictReader(tracked(SOURCE/'mixed_results.csv').open()));policies=[v['policy'] for v in design['rows']]
records={};timed=0;routes=[];layer_checks=0
for ri in range(3):
 for policy in [*policies,'uniform_dense_bf16_bracket_end']:
  raw=read(tracked(RUN/f'results/measurements/round{ri}_{policy}.json'));records[ri,policy]=raw
  assert raw['complete'] and not raw['quality'] and raw['routing_audit']['passed']
  assert all(v['exact_equal'] for v in raw['routing_audit']['mixed_layer_cases'])
  assert all(v['exact_equal'] for v in raw['exact_weight_audit'].values());layer_checks+=len(raw['routing_audit']['mixed_layer_cases'])
  for c in raw['cases']:
   case=c['case'];iters=[v for v in c['iterations'] if not v['warmup']];assert len(iters)==5
   for iteration in iters:
    sched=iteration['scheduler'];steps=sched['steps'];assert all(s['prefill_tokens']+s['decode_tokens']==s['scheduled_tokens']<=1024 for s in steps)
    assert sum(s['prefill_tokens'] for s in steps)==sched['total_prefill_rows']==sum(case['prompts'])
    assert sum(s['decode_tokens'] for s in steps)==sched['total_decode_rows']==8*(case['output']-1)
    assert sched['mixed_forwards']==sum(s['prefill_tokens']>0 and s['decode_tokens']>0 for s in steps)>0
    assert sched['mixed_forwards']==(15 if case['output']==64 else 10)
    timed+=1;routes.append({'round':ri,'policy':policy,'case':case['id'],'mixed_forwards':sched['mixed_forwards']})
for row in results:
 ratios=[]
 for ri in range(3):
  case=row['case'];anchors=[]
  for label in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']:
   c=next(c for c in records[ri,label]['cases'] if c['case']['id']==case)
   anchors.append(statistics.median(v['e2e_ms'] for v in c['iterations'] if not v['warmup']))
  c=next(c for c in records[ri,row['policy']]['cases'] if c['case']['id']==case)
  denominator=statistics.mean(anchors);actual=denominator if row['policy']=='uniform_dense_bf16' else statistics.median(v['e2e_ms'] for v in c['iterations'] if not v['warmup'])
  ratios.append(denominator/actual)
 assert abs(statistics.mean(ratios)-float(row['e2e_vs_bf16']))<1e-12
 quality=read(tracked(RUN/f"results/quality/quality_{row['policy']}.json"));baseline=read(tracked(RUN/'results/quality/quality_uniform_dense_bf16.json'))
 q=next(v for v in quality['cases'] if v['case']['id']==row['case']);b=next(v for v in baseline['cases'] if v['case']['id']==row['case']);assert len(q['blocks'])==len(b['blocks'])==64
 delta=[a['avg_nll']-v['avg_nll'] for a,v in zip(q['blocks'],b['blocks'],strict=True)]
 assert abs(statistics.mean(delta)-float(row['delta_nll']))<1e-12
 assert abs(statistics.stdev(delta)/4-float(row['delta_nll_2se']))<1e-12
normal=read(RUN/'results/quality/quality_uniform_dense_bf16.json');noop=read(tracked(RUN/'results/quality/quality_bf16_noop.json'))
assert noop['noop'] and len(noop['cases'])==2
for a,b in zip(normal['cases'],noop['cases'],strict=True):assert a['case']==b['case'] and a['blocks']==b['blocks']
for name,data in [('all_results.csv',results),('routing.csv',routes)]:
 with (OUT/name).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
for file in ['mixed_runtime.py','resident.py','benchmark_mixed.py']:tracked(RUN/'code'/file)
audit={'verified':True,'new_measurements':False,'policy_count':9,'cases':2,'rounds':3,'timed_bursts_including_bf16_end':timed,
 'mixed_forwards_per_burst':[15,10],'all_real_forward_row_totals_verified':True,'layer_reference_checks':layer_checks,
 'bf16_noop_window_checks':128,'measured_nll_windows_per_policy_case':64,'all_speed_ratios_recomputed':True,
 'runtime':'dual-resident eager with actual vLLM V1 scheduler; all requests admitted at t0',
 'pd_disaggregated_experiment':'not performed','arrival_stream_experiment':'not performed',
 'prior_provenance_sources_verified':len(prior['source_sha256']),'source_sha256':sources,'builder_sha256':sha(Path(__file__))}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
def speed(case,policy):return float(next(v['e2e_vs_bf16'] for v in results if v['case']==case and v['policy']==policy))
lines=['# R2：mixed/chunked执行与PD分离的回复证据','',
 '已有结果：047真实vLLM V1 chunked-prefill（token budget=1024）逐行阶段路由，9方法、2负载、3轮；本次只重算原始记录，没有新GPU实验。PD两worker/KV交接及错峰到达尚未实测。','',
 '## 中文回复草稿','',
 '我们同意continuous batching和chunked prefill会使同一次forward包含不同请求的Prefill与Decode token。原始整批阶段切换路径需要针对这种调度扩展，但阶段异构策略本身仍可执行：Linear计算按token行进行，可根据调度器的原始prompt长度、已计算位置和当前token位置划分两组行，分别调用相应阶段权重，再scatter回原顺序；Attention继续使用原请求边界和KV元数据，不改变Transformer结构。单token的prompt chunk仍归为Prefill，不能仅按query长度判定Decode。','',
 '我们已实现并测量这一扩展。实际运行中，标准B8/S2048/O64负载每个计时burst包含15次真正同时含两阶段token的forward；输入512/2048交替、输出256的异构负载每个burst包含10次。逐层gather/阶段kernel/scatter与同一行分组的独立参考计算完全一致，BF16 no-op在两负载各64窗上也一致。','',
 '| mixed负载 | 每burst真实mixed forward | DASO均衡 E2E×BF16 | DASO速度点 E2E×BF16 |','| --- | ---: | ---: | ---: |',
 f"| B8 / S2048 / O64 | 15 | {speed('chunk1024_b8_s2048_o64','w1_e0.01'):.3f} | {speed('chunk1024_b8_s2048_o64','w32_e0.03'):.3f} |",
 f"| B8 / S512、2048交替 / O256 | 10 | {speed('chunk1024_b8_heterogeneous_o256','w1_e0.01'):.3f} | {speed('chunk1024_b8_heterogeneous_o256','w32_e0.03'):.3f} |",'',
 '每方法/负载三轮各5次正式burst，完整九方法和实际负载64窗NLL均保留。该实现使用双表示常驻的eager路径，包含额外权重驻留及gather/scatter成本，与主表单arena/Graph路径分开；所有请求在t0提交，是closed-loop burst，不是已完成的生产open-loop连续到达测试。局部路由一致不等于chunked与non-chunked全模型逐位相同，动态激活量化尺度可能随分组改变，故实际mixed NLL另测。','',
 '对于PD分离，策略也有自然的架构映射：Prefill worker采用Prefill配置，Decode worker采用Decode配置，按一致的KV dtype/layout和请求位置语义交接KV。vLLM官方已有通过两个实例和KV connector实现PD分离的机制；据此可讨论兼容性，但我们尚未完成该两worker实验，不能将现有单实例速度等同于PD系统端到端收益，KV传输和首个生成token归属仍需验证。','',
 '## 建议的最小补充（尚未执行）','',
 '1. 优先补真实错峰到达：复用异构512/2048输入、输出256及冻结权重，采用同一预先固定的请求/到达序列；chunk token budget取512/1024，对比BF16、SQ-INT8→W4A16、DASO两点。记录请求加入/离开后逐步mixed计数、TTFT/TPOT、吞吐和峰值显存，并在实际调度下检查NLL；速度只在持续空闲GPU上统一测量。这直接补现有all-at-t0与生产持续到达的差距。','',
 '2. PD分离作为次优先补充：同样两worker/GPU资源下，先跑BF16，再跑一个冻结DASO阶段配置；真实交接Prefill产生的KV，对齐最后prompt token/首个生成token、position与cache布局。对比单实例同策略的逐token logprob/NLL并记录KV字节、传输耗时、TTFT/TPOT及每worker内存。它验证集成与语义，不能预先保证吞吐提升。','',
 '## 官方资料与本地来源','',
 '- [vLLM 0.11.1 chunked prefill](https://docs.vllm.ai/en/v0.11.1/configuration/optimization/#chunked-prefill)：同batch混合Prefill/Decode是实际调度行为。',
 '- [vLLM 0.11.1 PD分离](https://docs.vllm.ai/en/v0.11.1/features/disagg_prefill/)：两个实例经KV connector交接；该版本标注experimental。不将支持该形态解读为所有生产部署都使用PD分离。',
 '- [完整九方法实测](../qwen_updated_mixed_run047/README.md) · [重算数据](all_results.csv) · [逐burst路由](routing.csv) · [来源核验](audit.json)',
 '- [路由实现](../../../runs/047/code/mixed_runtime.py) · [主机/显存解释](../../reviewer1/phase_memory_reply/README.md)','']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},ensure_ascii=False))
