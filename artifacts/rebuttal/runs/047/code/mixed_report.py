"""Publish every fixed map, actual quality and measured mixed-resident costs."""
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

RUN=Path(__file__).resolve().parents[1];read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def save(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    design=read(RUN/'configs/design.json');digest=sha(RUN/'configs/design.json');sources={};rounds=[];brackets=[];routes=[]
    for path,h in {**design['input_sha256'],**design['protected_sha256']}.items():assert sha(Path(path))==h,path
    for pid,h in design['policies'].items():assert sha(RUN/f'configs/policies/{pid}.json')==h
    gate=read(RUN/'results/bf16_noop_gate.json');assert gate['verified']
    sources[str(RUN/'results/bf16_noop_gate.json')]=sha(RUN/'results/bf16_noop_gate.json')
    assert sha(RUN/'results/quality/quality_uniform_dense_bf16.json')==gate['routing_sha256']
    assert sha(RUN/'results/quality/quality_bf16_noop.json')==gate['noop_sha256']
    device=read(RUN/'configs/baseline_formal_device.json');records={};quality={}
    def load(path,formal=False):
        value=read(path);sources[str(path)]=sha(path);assert value['complete']and value['design_sha256']==digest
        assert value['routing_audit']['passed']and all(v['exact_equal']for v in value['exact_weight_audit'].values())
        if formal:
            assert value['physical_gpu']==device['physical_gpu']
            assert value['placement']['cpu_affinity']==device['cpu_affinity']
        return value
    for pid in design['policies']:
        value=load(RUN/f'results/quality/quality_{pid}.json');quality[pid]=value
        for case in value['cases']:
            assert len(case['blocks'])==64 and [r['block']for r in case['blocks']]==list(range(64))
            assert all(np.isfinite(r['avg_nll'])and r['token_count']==case['case']['output']for r in case['blocks'])
        for path in(RUN/f'results/quality_routes/quality_{pid}/{value["attempt"]}').glob('*/*.json'):
            route=read(path);sources[str(path)]=sha(path);assert route['mixed_forwards']>0
    for ri,order in enumerate(design['rounds']):
        for label in order:records[ri,label]=load(RUN/f'results/measurements/round{ri}_{label}.json',True)
        for ci,case in enumerate(design['cases']):
            first=records[ri,'uniform_dense_bf16']['cases'][ci]['medians']['e2e_ms']
            end=records[ri,'uniform_dense_bf16_bracket_end']['cases'][ci]['medians']['e2e_ms']
            reference=(first+end)/2;int8=records[ri,'uniform_int8']['cases'][ci]['medians']['e2e_ms']
            brackets.append({'round':ri,'case':case['id'],'bf16_start_ms':first,'bf16_end_ms':end,'end_over_start':end/first})
            for pid in design['policies']:
                record=records[ri,pid];observed=record['cases'][ci];assert observed['case']==case
                timed=[r for r in observed['iterations']if not r['warmup']];assert len(timed)==5
                for row in timed:
                    route=row['scheduler'];assert route['mixed_forwards']>0
                    assert route['total_prefill_rows']==sum(case['prompts'])and route['total_decode_rows']==8*(case['output']-1)
                    assert row['gpu_before']['physical_gpu_index']==row['gpu_after']['physical_gpu_index']==device['physical_gpu']
                value=reference if pid=='uniform_dense_bf16'else observed['medians']['e2e_ms']
                rounds.append({'round':ri,'case':case['id'],'policy':pid,'e2e_ms':value,'e2e_vs_bf16':reference/value,
                    'e2e_vs_int8':int8/value,'torch_peak_allocated_gib':observed['torch_peak_allocated_bytes']/1024**3,
                    'torch_peak_reserved_gib':observed['torch_peak_reserved_bytes']/1024**3,
                    'non_torch_gpu_max_gib':max(r['nvml_used_after_bytes']-r['torch_reserved_after_bytes']for r in timed)/1024**3,
                    'gpu_adapter_fields_gib':record['memory']['gpu_linear_bytes']/1024**3,
                    'host_adapter_fields_gib':record['memory']['host_linear_bytes']/1024**3,
                    'initialization_seconds':record['initialization_seconds']})
                routes.append({'round':ri,'case':case['id'],'policy':pid,'mixed_forwards_min':min(r['scheduler']['mixed_forwards']for r in timed),
                    'mixed_forwards_max':max(r['scheduler']['mixed_forwards']for r in timed),'timed_bursts':len(timed)})
    result=[]
    for ci,case in enumerate(design['cases']):
        baseline=np.array([r['avg_nll']for r in quality['uniform_dense_bf16']['cases'][ci]['blocks']])
        for method in design['rows']:
            pid=method['policy'];local=[r for r in rounds if r['policy']==pid and r['case']==case['id']]
            assert len(local)==3
            delta=np.array([r['avg_nll']for r in quality[pid]['cases'][ci]['blocks']])-baseline
            mean=float(delta.mean());two_se=float(2*delta.std(ddof=1)/8);guard=0. if pid=='uniform_dense_bf16'else mean+two_se+.005
            requests=[r for ri in range(3)for burst in records[ri,pid]['cases'][ci]['iterations']if not burst['warmup']for r in burst['requests']]
            row={'case':case['id'],'method':method['method'],'policy':pid,'nominal_q':method.get('budget'),
                'delta_nll':mean,'delta_nll_2se':two_se,'guard':guard,'quality_windows':64,
                'e2e_ms':statistics.mean(r['e2e_ms']for r in local),
                **{key:statistics.mean(r[key]for r in local)for key in ['e2e_vs_bf16','e2e_vs_int8','gpu_adapter_fields_gib','host_adapter_fields_gib','initialization_seconds']},
                'e2e_vs_int8_min':min(r['e2e_vs_int8']for r in local),'e2e_vs_int8_max':max(r['e2e_vs_int8']for r in local),
                'torch_peak_allocated_gib':max(r['torch_peak_allocated_gib']for r in local),
                'torch_peak_reserved_gib':max(r['torch_peak_reserved_gib']for r in local),
                'non_torch_gpu_max_gib':max(r['non_torch_gpu_max_gib']for r in local),
                **{f'p{q}_{key}':float(np.quantile([r[key]for r in requests],q/100))for q in [50,95,99]for key in ['ttft_ms','tpot_ms']}}
            result.append(row)
    assert len(result)==18
    dest=RUN.parents[1]/'responses/reviewer2/qwen_updated_mixed_run047';dest.mkdir(parents=True,exist_ok=True)
    for name,values in [('mixed_results.csv',result),('mixed_rounds.csv',rounds),('bf16_brackets.csv',brackets),('routing_counts.csv',routes)]:
        save(RUN/'results'/name,values);shutil.copy2(RUN/'results'/name,dest/name)
    lines=['# Qwen新点真实chunked/mixed调度（R2 Exi5 / R1 AK6a / R5 gkP9）','',
        '040固定九map、006原两个代表case，实际vLLM V1 chunked prefill预算1024、同forward的prefill/decode逐行路由。所有请求t0进入；这是closed-loop burst、dual-resident eager变体，不能称HTTP/open-loop生产服务或原单表示主表性能。','',
        f'固定GPU{device["physical_gpu"]}/NUMA{device["numa_node"]}、KV6GiB/no prefix cache。三轮每轮2warmup+5timed，首尾BF16；每轮BF16两端中位数算术平均为分母，再取三轮加速算术均值。BF16展示此锚点均值，其他方法展示各轮中位数均值。p50/p95/p99来自15个timed burst的120条请求观测，是描述性burst分位数。','',
        '质量为019原long_quality中64个已查看窗口，在各case实际prompt/输出长度上paired teacher forcing；不是新盲测或下游生成任务分数。保护值=mean+2SE+.005（BF16为0），不把旧名义q当此场景质量保证。BF16 routed/no-op逐窗口NLL通过；真实mixed forward及阶段总token数逐burst核验，跨格式模块精确gather/apply/scatter审计，计时外核验GPU/CPU阶段权重一致。','',
        '| Case | 方法 | E2E ×BF16 | ×INT8 三轮范围 | ΔNLL±2SE | 保护值 | p95 TTFT ms | p95 TPOT ms | GPU adapter字段GiB | Torch峰值GiB |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result:
        lines.append(f'| {r["case"]} | {r["method"]} | {r["e2e_vs_bf16"]:.3f} | {r["e2e_vs_int8_min"]:.3f}–{r["e2e_vs_int8_max"]:.3f} | {r["delta_nll"]:.5f}±{r["delta_nll_2se"]:.5f} | {r["guard"]:.5f} | {r["p95_ttft_ms"]:.1f} | {r["p95_tpot_ms"]:.2f} | {r["gpu_adapter_fields_gib"]:.2f} | {r["torch_peak_allocated_gib"]:.2f} |')
    lines+=['','GPU/host adapter字段按实际storage去重，是该字段子集；Torch峰值另含KV、workspace和其余模型张量，不能将二者相加。dual-resident为消除阶段释放约束付出显存，主表arena恢复成本不移植为这里的异步收益。完整CSV另列初始化、host字段、reserved峰值、非Torch显存诊断和p50/p99。两case/三轮不构成稳定生产SLA或所有负载全局最优保证。','',
        '[完整表](mixed_results.csv) · [三轮](mixed_rounds.csv) · [首尾BF16](bf16_brackets.csv) · [实际mixed计数](routing_counts.csv) · [设计/代码/原始captures](../../../runs/047/README.md) · [来源](provenance.json)','']
    (dest/'README.md').write_text('\n'.join(lines))
    for path in(RUN/'code').rglob('*'):
        if path.is_file():sources[str(path)]=sha(path)
    audit={'complete':True,'verified':True,'maps':9,'cases':2,'quality_windows_per_case':64,'formal_processes':30,
        'formal_device':device,'design_sha256':digest,'source_sha256':sources,
        'parent_qwen_repeat_audit_sha256':sha(RUN.parent/'048/results/final_audit.json'),
        'representation_handoff_sha256':{p.name:sha(p)for p in(RUN/'results/checkpoint_reuse').glob('*.json')},
        'delivery_sha256':{p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'}}
    (dest/'provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
    (RUN/'results/final_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Published047:18 rows,30 formal processes',flush=True)


if __name__=='__main__':main()
