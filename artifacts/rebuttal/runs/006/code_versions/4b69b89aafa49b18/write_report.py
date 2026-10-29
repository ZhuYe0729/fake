"""Evidence for actual mixed-token execution, numerical checks and tradeoffs."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import RUN,write_json


def main():
    for stage in ['speed','quality']:assert json.loads((RUN/f'results/worker_{stage}.json').read_text())['status']=='complete'
    assert json.loads((RUN/'results/full_quality_worker.json').read_text())['status']=='complete'
    design=json.loads((RUN/'configs/design.json').read_text());rows=[];equivalence=[];routes=[]
    for policy in design['policies']:
        mapping=json.loads((RUN/f'configs/policies/{policy}.json').read_text())['method_map']
        cross=sum(p['prefill_method']!=p['decode_method'] for p in mapping.values())
        for case in design['cases']:
            r=json.loads((RUN/f'results/serving/{policy}/{case["id"]}.json').read_text())
            base=json.loads((RUN/f'results/serving/uniform_dense_bf16/{case["id"]}.json').read_text())
            assert len(r['iterations'])==7 and sum(not x['warmup'] for x in r['iterations'])==5
            for sample in r['iterations']:
                scheduler=sample['scheduler'];assert scheduler['mixed_forwards']>0
                assert scheduler['total_prefill_rows']==sum(case['prompts'])
                assert scheduler['total_decode_rows']==len(case['prompts'])*(case['output']-1)
                if cross:assert scheduler['linear_counts']['mixed']==cross*scheduler['mixed_forwards']
            rows.append({'policy':policy,'case':case['id'],'e2e_ms':r['e2e_ms'],'speedup':base['e2e_ms']/r['e2e_ms'],
                **{key:r[key] for key in ['output_tokens_per_second','p50_ttft_ms','p95_ttft_ms','p50_tpot_ms','p95_tpot_ms','p95_completion_ms']},
                'mixed_forwards_per_iteration':r['iterations'][-1]['scheduler']['mixed_forwards'],
                'mixed_linears_per_iteration':r['iterations'][-1]['scheduler']['linear_counts'].get('mixed',0),
                'gpu_linear_weight_gib':r['memory']['gpu_linear_bytes']/1024**3,
                'host_linear_weight_gib':r['memory']['host_linear_bytes']/1024**3,
                'peak_allocated_gib':r['peak_allocated_bytes']/1024**3,'gpu':r['gpu']})
    assert len(rows)==18 and {r['gpu'] for r in rows}=={'6'}
    for case in design['cases']:
        base=json.loads((RUN/f'results/serving/uniform_dense_bf16/{case["id"]}.json').read_text())
        noop=json.loads((RUN/f'results/serving/uniform_dense_bf16_noop/{case["id"]}.json').read_text())
        hashes={x['output_sha256'] for r in [base,noop] for x in r['iterations']};assert len(hashes)==1
        equivalence.append({'case':case['id'],'exact_token_match':True,'routed_ms':base['e2e_ms'],'original_apply_ms':noop['e2e_ms'],
            'scope':'Both use same actual scheduler and row classifier; original uniform apply vs experiment fast path. Python dispatch bookkeeping differs.'})
    for path in (RUN/'results/routing_audits').glob('*.json'):
        r=json.loads(path.read_text());assert r['passed'];routes.append({'file':str(path.relative_to(RUN)),**r})
    quality=[]
    for policy in ['uniform_dense_bf16','phase_sq_int8_w4a16_ours','transfer_daso8_prefill_decode_01']:
        reference=json.loads((RUN/f'results/quality/{policy}/nonchunked.json').read_text())
        for budget in [1024,4096]:
            actual=json.loads((RUN/f'results/quality/{policy}/{budget}.json').read_text());assert actual['scheduler']['mixed_forwards']>0
            delta=np.array([a['avg_nll']-b['avg_nll'] for a,b in zip(actual['blocks'],reference['blocks'],strict=True)])
            quality.append({'policy':policy,'token_budget':budget,'nonchunked_nll':reference['avg_nll'],
                'mixed_nll':actual['avg_nll'],'delta_nll':float(delta.mean()),'mean_abs_block_delta':float(abs(delta).mean()),
                'max_abs_block_delta':float(abs(delta).max()),'blocks':len(delta),'mixed_forwards':actual['scheduler']['mixed_forwards']})
    actual_quality=[];rng=np.random.default_rng(144967);bootstrap=rng.integers(0,16,(4000,16))
    for case in design['cases']:
        reference=json.loads((RUN/f'results/full_quality/uniform_dense_bf16/{case["id"]}.json').read_text())
        for policy in design['policies']:
            q=json.loads((RUN/f'results/full_quality/{policy}/{case["id"]}.json').read_text());assert len(q['blocks'])==16
            assert all(s['mixed_forwards']>0 for s in q['schedulers'])
            delta=np.array([a['avg_nll']-b['avg_nll'] for a,b in zip(q['blocks'],reference['blocks'],strict=True)])
            ci=np.quantile(delta[bootstrap].mean(1),[.025,.975]);speed=next(r for r in rows if r['policy']==policy and r['case']==case['id'])
            actual_quality.append({'policy':policy,'case':case['id'],'nll':q['avg_nll'],'delta_nll':float(delta.mean()),
                'delta_nll_ci_low':float(ci[0]),'delta_nll_ci_high':float(ci[1]),'speedup':speed['speedup'],
                'blocks':16,'token_count':sum(b['token_count'] for b in q['blocks'])})
    for name,records in [('serving_summary',rows),('chunking_quality',quality),('actual_workload_quality',actual_quality),('uniform_equivalence',equivalence)]:
        with (RUN/f'results/{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,list(records[0]));writer.writeheader();writer.writerows(records)
    write_json(RUN/'results/mixed_runtime_audit.json',{'passed':True,'uniform_equivalence':equivalence,
        'routing_audits':routes,'quality_checks':quality,'actual_workload_quality':actual_quality,'speed_cases':21,'main_comparisons':18,
        'all_iterations_real_mixed':True,'all_prefill_decode_row_counts_exact':True,
        'scope':'Correct phase routing and fixed target scoring; not production open-loop SLA or general quality guarantee.'})
    lines=['# Run006：真实chunked-prefill混合token执行','',
        '原global-phase、异步单表示释放实现无法直接处理同一forward中的不同请求阶段。本轮是独立进程安装的dual-resident变体：按vLLM实际scheduled token位置与原prompt长度区分每行，分别执行两种真实Linear适配器，再合并BF16输出。共享vLLM代码未修改。','',
        '三种冻结burst负载，实际开启vLLM V1 chunked prefill；所有请求在一次burst起点进入，尚非HTTP/open-loop到达负载。每次实际观测9–15个mixed forward，所有prefill/decode行数与请求token数精确核对；异格式模块调用数=真实mixed forward数×异格式模块数。','',
        '## 真实serving测量','',
        '同一GPU6、6GiB BF16 KV、eager、无prefix cache、2 warmups+5次完整burst。p50/p95基于最终5轮中的请求观测，包含排队；TPOT为每请求首末token间平均间隔。均匀BF16也走相同实验fast path，另有原apply数值控制。','',
        '| 负载 | 策略 | E2E ms | 相对BF16 | 输出tokens/s | p50 TTFT | p95 TTFT | p50 TPOT | p95 TPOT | mixed forward |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f'| {r["case"]} | {r["policy"]} | {r["e2e_ms"]:.1f} | {r["speedup"]:.3f}× | {r["output_tokens_per_second"]:.1f} | {r["p50_ttft_ms"]:.1f} | {r["p95_ttft_ms"]:.1f} | {r["p50_tpot_ms"]:.2f} | {r["p95_tpot_ms"]:.2f} | {r["mixed_forwards_per_iteration"]} |')
    lines+=['','## 数值核验与chunking影响','',
        '- Uniform BF16原apply与routed fast path在三负载全部生成token逐个相同；两者Python bookkeeping不同，不能把时间差全部归为压缩收益。',
        '- 对每个异格式模块，用相同随机输入、相同行子集分别调用prefill/decode适配器，再和实际gather/scatter路径逐元素比较，精确一致。',
        '- 另用16个validation文本块、固定相同32个目标token，比较非chunked与两档chunked调度。以下用于数值/阶段检查，不是完整下游任务或测试分布的质量保证。NVFP4的全张量activation scale可能随chunk分组改变，因此同一方法的跨chunk结果无需逐位相同。','',
        '首次质量试跑捕获评估器问题：原非chunked teacher-forcing计数器会消费vLLM为未完成prompt chunk生成、随后丢弃的sample。固定目标断言失败后，独立修正为仅在prompt完成时计分，并按真实output_ids长度索引target。非chunked旧基准512条capture的step均等于实际output_length，仍有效；失败的chunked capture和日志保留，未写入最终NLL。全部冻结速度结果保持原样。','',
        '| 策略 | token预算 | 非chunked NLL | mixed NLL | ΔNLL | 最大block绝对Δ |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in quality:lines.append(f'| {r["policy"]} | {r["token_budget"]} | {r["nonchunked_nll"]:.6f} | {r["mixed_nll"]:.6f} | {r["delta_nll"]:.6f} | {r["max_abs_block_delta"]:.6f} |')
    lines+=['','## 与实际serving负载对齐的质量检查','',
        '短数值探针显示chunk分组可以改变FP4误差，因此进一步固定16个长度4352的非重叠validation窗口，对全部六个benchmark策略按真实B/S/O和token预算执行teacher forcing。模型策略与原速度结果保持冻结；没有用这些标签修改策略。每场景均以相同mixed调度的BF16作为质量参考。','',
        '| 场景 | 策略 | 实际ΔNLL | 描述性95%窗口区间 | 已测速度 |','| --- | --- | ---: | --- | ---: |']
    for r in actual_quality:lines.append(f'| {r["case"]} | {r["policy"]} | {r["delta_nll"]:.5f} | [{r["delta_nll_ci_low"]:.5f}, {r["delta_nll_ci_high"]:.5f}] | {r["speedup"]:.3f}× |')
    lines+=['',
        '共288个计分窗口/36864个目标token；不同场景重复使用同16个窗口，不是288份独立文本。区间为配对窗口bootstrap描述，样本较少，不应把单个均值落在预算内写成严格保证。',
        '固定五候选策略在三个负载的实测ΔNLL约−0.009至+0.036，七候选约+0.005至+0.023；新混合执行与原同质校准的实际质量并不完全一致。所有点与质量区间保留，不能只引速度或把原预算直接迁移为mixed保证。']
    lines+=['','## 显存与适用范围','',
        '| 策略 | GPU Linear权重GiB | host Linear权重GiB | 峰值allocated GiB |','| --- | ---: | ---: | ---: |']
    for policy in design['policies']:
        values=[r for r in rows if r['policy']==policy];r=values[0]
        lines.append(f'| {policy} | {r["gpu_linear_weight_gib"]:.3f} | {r["host_linear_weight_gib"]:.3f} | {max(x["peak_allocated_gib"] for x in values):.3f} |')
    lines+=['',
        '双表示保留在GPU，额外显存是支持任意混合阶段的明确代价；host缓存本轮未删除，CPU/pinned内存仍需计入。gather/scatter、两个子batch kernel与行分类的开销均包含在最终E2E。这里只报告总体实测效果，不将全部差异归为纯DMA或纯gather开销。',
        '策略仍是003冻结的同质负载策略，本轮未根据serving测试质量/速度重新挑选策略。固定策略迁移的限制与原global-phase实现、此dual-resident变体的新增能力必须分别表述。',
        '本轮证实真实调度中能正确选择两种阶段格式；不主张原文已有生产兼容性，不主张完成多机、多租户、preemption或严格open-loop SLA验证。','',
        '![真实混合调度吞吐](results/figures/serving_speed.png)','',
        '[完整CSV](results/serving_summary.csv) · [路由/数值审计](results/mixed_runtime_audit.json) · [chunking质量](results/chunking_quality.csv) · [冻结负载](configs/design.json)。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    directory=RUN/'results/figures';directory.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,case in zip(axes,design['cases'],strict=True):
        values=[r for r in rows if r['case']==case['id']];ax.bar(range(len(values)),[r['speedup'] for r in values])
        ax.axhline(1,color='black',lw=.7);ax.set_xticks(range(len(values)),[r['policy'].replace('transfer_','').replace('_prefill_decode_','_pd_') for r in values],rotation=65,ha='right',fontsize=8)
        ax.set_title(case['id'],fontsize=9);ax.set_ylabel('E2E speedup over BF16');ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(directory/f'serving_speed.{ext}',dpi=180)
    plt.close(fig)
    write_json(RUN/'results/report_complete.json',{'complete':True,'speed_cases':21,'comparisons':18,'quality_checks':6,
        'actual_workload_quality_files':18,'actual_workload_quality_blocks':288,'actual_workload_scored_tokens':36864,'real_mixed':True})


if __name__=='__main__':main()
