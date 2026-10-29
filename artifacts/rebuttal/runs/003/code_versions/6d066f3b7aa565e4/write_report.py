"""Create complete workload evidence, including unfavorable comparisons."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import RUN, write_json


def main():
    progress=json.loads((RUN/'results/progress.json').read_text());assert progress['complete']
    rows=list(csv.DictReader((RUN/'results/summary_final.csv').open()))
    design=json.loads((RUN/'configs/design.json').read_text())
    profile=json.loads((RUN/'results/profile_complete.json').read_text())
    replay=json.loads((RUN/'results/allocation_replay_audit.json').read_text())
    card=json.loads((RUN/'configs/measurement_card_change.json').read_text())
    for row in rows:
        for k in ['ttft_speedup','e2e_speedup','ttft_ms','e2e_ms','tpot_ms','output_tokens_per_second',
                  'measured_delta_nll','delta_nll_ci_low','delta_nll_ci_high','quality_budget','budget_violation']:
            if row.get(k) not in ['',None]:row[k]=float(row[k])
    def speed(row):return row['ttft_speedup'] if row['scenario']=='prefill_only' else row['e2e_speedup']
    def best(workload,families,cap):
        selected=[r for r in rows if r['workload']==workload and r['family'] in families and r['measured_delta_nll']<=cap]
        return max(selected,key=speed) if selected else None
    families={'uniform':['uniform'],'phase-uniform':['phase-uniform'],
        'DASO fixed':['DASO fixed transfer'],'DASO adapted':['DASO workload adapted'],
        'HAWQ':['HAWQ workload adapted','HAWQ FP4 fixed transfer']}
    frontiers=[]
    for w in design['workloads']:
        for cap in [.01,.02,.04,.07]:
            for group,items in families.items():
                b=best(w['id'],items,cap)
                frontiers.append({'workload':w['id'],'scenario':w['scenario'],'actual_delta_nll_cap':cap,
                    'group':group,'label':b['label'] if b else None,'speedup':speed(b) if b else None,
                    'actual_delta_nll':b['measured_delta_nll'] if b else None})
    write_json(RUN/'results/quality_matched_frontiers.json',frontiers)
    lines=['# Qwen3-8B：batch、输入长度与生成长度补充测试','',
        '配置在测量前固定：9组 P+D、7组 prefill-only；267个比较项，完整 method-map 合并为47份策略。',
        '同时报告原策略迁移和按实际 shape 延迟重新分配。原002质量拟合/局部误差和2000-bin DP均保持冻结，以下实际质量筛选仅用于分析，不回写策略。','',
        '## 工作负载与绝对延迟','',
        '| 场景 | B | 输入 | 输出 | BF16 TTFT ms | BF16 E2E ms | BF16 TPOT ms | BF16 prefill tokens/s | BF16 输出tokens/s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for w in design['workloads']:
        r=next(r for r in rows if r['workload']==w['id'] and r['label']=='uniform_dense_bf16')
        lines.append(f'| {w["id"]} | {w["batch"]} | {w["input"]} | {w["output"]} | {r["ttft_ms"]:.1f} | {r["e2e_ms"]:.1f} | {r["tpot_ms"]:.2f} | {float(r["prefill_input_tokens_per_second"]):.1f} | {r["output_tokens_per_second"]:.1f} |')
    for cap in [.01,.04]:
        lines+=['',f'## 实际 ΔNLL≤{cap:g} 的已测点比较','',
            '各组取该实际质量上限内已测点中的最快值；“—”表示预定策略没有落入该区间。它不代表预测预算保证或连续前沿的数学最优。','',
            '| 场景 | Uniform最佳 | Phase-uniform | DASO固定迁移 | DASO按负载重分配 | HAWQ已测点 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for w in design['workloads']:
            values=[best(w['id'],v,cap) for v in families.values()]
            lines.append('| '+w['id']+' | '+' | '.join(f'{speed(v):.3f}×' if v else '—' for v in values)+' |')
    violations={}
    for scenario in ['prefill_only','prefill_decode']:
        selected=[r for r in rows if r['scenario']==scenario and r['family']=='DASO workload adapted']
        violations[scenario]={'points':len(selected),'violations':sum(r['budget_violation']>0 for r in selected),
            'max_violation':max(r['budget_violation'] for r in selected)}
    write_json(RUN/'results/budget_audit.json',violations)
    lines+=['','## 预算与解释范围','',
        '| 场景 | 重分配策略比较项 | 超预算 | 最大超预算 nats/token |','| --- | ---: | ---: | ---: |']
    for scenario,v in violations.items():lines.append(f'| {scenario} | {v["points"]} | {v["violations"]} | {v["max_violation"]:.6f} |')
    lines+=['',
        f'- 完整shape计时耗时{profile["seconds"]:.2f}s（360项）；独立CPU搜索重放{replay["replay_seconds"]:.2f}s，47份策略逐字节重现。不含从零质量校准成本，原拟合与局部误差继承002。',
        '- 质量模型在原 B8/S2048/O64 校准；本轮跨输入/输出长度仅迁移，不重新拟合。不得将本轮预测预算写成实际质量保证。',
        '- Prefill质量沿用 B1 prompt-logprob 协议；P+D质量按目标B执行teacher forcing，实际计分100块。最后不足一批时重复样本补齐，只计原100块。',
        '- ΔNLL区间为2000次配对块bootstrap的描述性区间。测试窗口可能重叠，它不是对所有文本分布的覆盖保证。',
        f'- 最终正式速度全部在同一物理GPU{card["final_speed_gpu"]}，统一6GiB BF16 KV、2 warmups + 5 timed；报告相对本轮同卡同负载BF16的速度。与001/002的3GiB结果不混用。',
        '- 初始GPU3在实验中被外部任务占用，启动显存检查在新策略计时前拒绝继续；全部最终速度迁到空闲GPU0重测，GPU3部分结果完整保存在diagnostics，不参与最终比较。模块profile仍来自原GPU3，策略与质量拟合保持冻结。运行间GPU0临时保留12GiB，模型初始化和所有计时前释放，握手记录可审计。',
        '- 输入使用冻结seed的随机token，固定输出数量并忽略EOS；质量使用独立冻结WikiText测试token。TTFT/O=1与完整生成分别计时，TPOT=(E2E−TTFT)/(O−1)。',
        '- GPU peak列是进程累计allocator峰值，包含初始化及此前负载；不把它解释为单负载权重常驻量。',
        '- 同步同质batch、eager、无prefix cache/chunked prefill；本轮未验证混合continuous batching。',
        '- 普通INT8与HAWQ共用本轮融合实现，权重量化不变；SQ-INT8包含独立平滑预处理，FP8是真实CUTLASS E4M3 W8A8。普通INT8激活舍入可有一档差异，所有NLL重新测量。',
        '- HAWQ为Qwen3 PTQ移植：沿用001完整Hessian、signed sensitivity和INT4/INT8/BF16绑定phase的ILP；按新shape成本重分配。延迟上限对齐DASO预测Linear成本，低于其最小可行成本时显式抬到该最小值，详见CSV的infeasible_requested_limit。',
        '- HAWQ FP4列另含001固定FP4策略的迁移控制，原Hessian负值保留。没有将PTQ移植表述成原CNN QAT/TVM流程。',
        '- 延迟搜索仍是Linear成本之和；阶段切换、传输与非Linear成本由最终整模型计时包含，可能改变实际排序。','',
        '- 本轮新增策略评测目标负载的NLL；完整下游任务证据沿用002中的固定策略结果，不主张每份新增策略都已重测MMLU/ROUGE/BLEU。','',
        '## 图与数据','',
        '![工作负载加速](results/figures/workload_speed.png)','',
        '![实际质量与速度](results/figures/quality_speed.png)','',
        '[全范围质量—速度图](results/figures/quality_speed_full.png) 保留低损失放大图范围以外的所有已测点。','',
        '[完整CSV](results/summary_final.csv) · [实际质量前沿](results/quality_matched_frontiers.json) · [候选比例](results/candidate_usage.json) · [完整性审计](results/final_audit.json)','',
        '代码、冻结配置与数据、所有原始计时和teacher-forcing记录位于本目录；失败试跑、数值检查与重建记录均保留。原001/002通过继承manifest校验。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    figures=RUN/'results/figures';figures.mkdir(exist_ok=True)
    colors={'uniform':'#777777','phase-uniform':'#c89324','DASO fixed transfer':'#2466af',
            'DASO workload adapted':'#d24836','HAWQ workload adapted':'#648e3b','HAWQ FP4 fixed transfer':'#98b877'}
    pd=[w for w in design['workloads'] if w['scenario']=='prefill_decode']
    fig,ax=plt.subplots(figsize=(13,5))
    for label in ['uniform_sq_int8','uniform_fp8','transfer_daso8_prefill_decode_01',
                  'transfer_daso5_measured_prefill_decode_04','phase_sq_int8_w4a16_ours']:
        values=[next(r['e2e_speedup'] for r in rows if r['workload']==w['id'] and r['label']==label) for w in pd]
        ax.plot(range(len(pd)),values,marker='o',label=label.replace('transfer_',''))
    ax.axhline(1,color='black',lw=.7);ax.set_xticks(range(len(pd)),[f'B{w["batch"]}\nS{w["input"]}\nO{w["output"]}' for w in pd])
    ax.set_ylabel('E2E speedup over BF16');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
    ax.set_title('Frozen-policy workload transfer (quality reported separately)');fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'workload_speed.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(16,13))
    for ax,w in zip(axes.flat,design['workloads'],strict=True):
        for family,color in colors.items():
            values=[r for r in rows if r['workload']==w['id'] and r['family']==family]
            if values:ax.scatter([r['measured_delta_nll'] for r in values],[speed(r) for r in values],c=color,label=family,s=24)
        ax.axhline(1,color='black',lw=.5);ax.axvline(0,color='black',lw=.5)
        ax.set_xlim(-.04,.16);ax.set_title(w['id'],fontsize=10);ax.grid(alpha=.2)
        ax.set_xlabel('Measured delta NLL');ax.set_ylabel('Speedup')
    legend={}
    for ax in axes.flat:
        handles,labels=ax.get_legend_handles_labels()
        legend.update(zip(labels,handles))
    fig.legend(legend.values(),legend.keys(),loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('All prespecified workload points (low-loss zoom; full range in CSV)');fig.tight_layout(rect=(0,.045,1,.975))
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'quality_speed.{ext}',dpi=180)
    for ax,w in zip(axes.flat,design['workloads'],strict=True):
        values=[r['measured_delta_nll'] for r in rows if r['workload']==w['id']]
        low,high=min(values),max(values);margin=max(.01,(high-low)*.06)
        ax.set_xlim(low-margin,high+margin)
    fig.suptitle('All prespecified workload points (full measured quality range)')
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'quality_speed_full.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({'report':str(RUN/'REPORT.md'),'rows':len(rows),'violations':violations}))


if __name__=='__main__':main()
