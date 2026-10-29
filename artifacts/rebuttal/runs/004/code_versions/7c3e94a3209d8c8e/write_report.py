"""Summarize measured preparation, model, ABI, transfer and memory costs."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import RUN,write_json


def main():
    design=json.loads((RUN/'configs/design.json').read_text());rows=[];equal=[]
    assert json.loads((RUN/'results/runtime_worker.json').read_text())['status']=='complete'
    for policy in design['policies']:
        modes=design['uniform_modes'] if policy.startswith('uniform_') else design['modes']
        for mode in modes:
            for w in design['workloads']:
                r=json.loads((RUN/f'results/runtime/{policy}/{mode}/{w["id"]}.json').read_text());e=r['e2e'];t=r['ttft']
                base=json.loads((RUN/f'results/runtime/uniform_dense_bf16/async/{w["id"]}.json').read_text())['e2e']
                rows.append({'policy':policy,'mode':mode,'workload':w['id'],'ttft_ms':t['generation_ms'],
                    'e2e_ms':e['generation_ms'],'restore_ms':e['restore_ms'],'cycle_ms':e['cycle_ms'],'tpot_ms':r['tpot_ms'],
                    'e2e_speedup':base['generation_ms']/e['generation_ms'],'cycle_speedup':base['cycle_ms']/e['cycle_ms'],
                    'gpu_weight_gib_initial':r['initial_memory']['gpu_linear_weights_bytes']/1024**3,
                    'gpu_allocated_gib_initial':r['initial_memory']['allocated_bytes']/1024**3,
                    'gpu_peak_allocated_gib':e['memory']['peak_allocated_bytes']/1024**3,
                    'host_weights_gib':e['memory']['host_weights_bytes']/1024**3,
                    'pinned_weights_gib':e['memory']['pinned_weights_bytes']/1024**3,
                    'host_rss_gib':e['memory']['rss_bytes']/1024**3,
                    'host_peak_rss_gib':e['memory']['process_peak_rss_bytes']/1024**3,
                    'decode_transfer_gib':r['theoretical_decode_transfer_bytes']/1024**3,
                    'restore_transfer_gib':r['theoretical_prefill_restore_bytes']/1024**3,
                    'output_sha256':e['output_sha256'],'ttft_sha256':t['output_sha256'],'gpu':r['gpu']})
        if len(modes)>1:
            for w in design['workloads']:
                selected=[r for r in rows if r['policy']==policy and r['workload']==w['id']]
                passed=len({(r['output_sha256'],r['ttft_sha256']) for r in selected})==1
                equal.append({'policy':policy,'workload':w['id'],'all_mode_tokens_identical':passed})
    assert len(rows)==36 and all(r['all_mode_tokens_identical'] for r in equal),equal
    assert {r['gpu'] for r in rows}=={'6'}
    with (RUN/'results/runtime_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/runtime_semantics_audit.json',{'passed':True,'cases':equal,'scope':'Same frozen prompts, phase methods and exact generated token hashes across async/sync/resident controls; all untraced timings on GPU6.'})
    interfaces=json.loads((RUN/'results/interfaces_complete.json').read_text());large=json.loads((RUN/'results/large_shapes_complete.json').read_text())
    broad=list(csv.DictReader((RUN/'results/broad_latency_validation.csv').open()))
    local=json.loads((RUN/'results/cpu_analysis_complete.json').read_text());additive=json.loads((RUN/'results/additive_cost_complete.json').read_text())
    profiles=[]
    for path in (RUN/'results/profiles').glob('*/*/*_summary.json'):
        r=json.loads(path.read_text());profiles.append({'policy':path.parent.parent.name,'mode':path.parent.name,
            'workload':path.name.replace('_summary.json',''),**r})
    assert len(profiles)==12,len(profiles)
    write_json(RUN/'results/profile_summary.json',profiles)
    lines=['# Run004：准备、预测、接口、传输与内存开销','',
        '固定六份已完成质量评测的策略，三种代表负载。所有新增无插桩速度在同一GPU6；异步/同步/双驻留使用相同权重与输入，9组跨模式输出逐token相同。',
        'generation从提交请求计时，cycle额外包含下一次prefill的同步恢复。双驻留是额外显存控制，不等同于原文单表示释放流水线。','',
        '## 实际运行时成本','',
        '| 策略 | 模式 | 负载 | TTFT ms | E2E ms | 恢复 ms | 完整cycle ms | E2E加速 | cycle加速 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f'| {r["policy"]} | {r["mode"]} | {r["workload"]} | {r["ttft_ms"]:.1f} | {r["e2e_ms"]:.1f} | {r["restore_ms"]:.1f} | {r["cycle_ms"]:.1f} | {r["e2e_speedup"]:.3f}× | {r["cycle_speedup"]:.3f}× |')
    lines+=['','## 实际内存与理论传输字节','',
        '以下为standard负载，GiB=2^30 bytes。GPU allocator包含6GiB KV及工作区；单独列出Linear权重storage。host RSS与pinned权重不应相加，后者已经属于进程内存。峰值包含初始化，不能解释为稳定权重大小。','',
        '| 策略 | 模式 | GPU权重 | GPU初始化allocated | GPU峰值allocated | host权重 | pinned权重 | host RSS | decode H2D | 恢复 H2D |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if r['workload']!='standard':continue
        keys=['gpu_weight_gib_initial','gpu_allocated_gib_initial','gpu_peak_allocated_gib','host_weights_gib','pinned_weights_gib','host_rss_gib','decode_transfer_gib','restore_transfer_gib']
        lines.append('| '+r['policy']+' | '+r['mode']+' | '+' | '.join(f'{r[k]:.3f}' for k in keys)+' |')
    lines+=['','## 延迟预测准确性与边界','',
        '原广覆盖profiling数据按新固定seed重新划分，shape和N/K组分别严格留出，所有模型仅用各自训练分区重新拟合。此CPU重放使用历史profiling设备/协议，不能充当新的跨硬件泛化实验。','',
        '| 方法 | 留出 | 训练shape | 测试shape | MAPE | median APE | p90 APE | 拟合s | 批查询s |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in broad:lines.append(f'| {r["method"]} | {r["split"]} | {r["train_shapes"]} | {r["test_shapes"]} | {float(r["mape_percent"]):.2f}% | {float(r["median_ape_percent"]):.2f}% | {float(r["p90_ape_percent"]):.2f}% | {float(r["fit_seconds"]):.3f} | {float(r["query_seconds"]):.3f} |')
    lines+=['',
        '另做极少局部数据控制：每方法仅16个训练点，6种M留出。部分方法误差很大，sparse原模型在小M不支持的点显式记为unsupported；实际部署PaddedLinear会padding，但本控制不把已见的padding后shape算作独立留出。完整负面结果保存在latency_holdout_metrics.csv，不能声称16点足够校准新硬件。','',
        '整模型附加误差使用003冻结profile与最终GPU0测量：用每负载BF16的E2E−Linear和作为固定项，其他策略均未参与校准。下面给出全部其他预定点的误差；固定项包含非Linear/引擎成本，策略相关传输残差仍可能很大。模块profile来自GPU3、整模型来自GPU0，都是5090；这不是同一卡的kernel因果拆分。','',
        '| 场景 | 模型 | 点数 | MAPE | median APE | p90 APE |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in additive['metrics']:lines.append(f'| {r["scenario"]} | {r["metric"]} | {r["rows"]} | {r["mape_percent"]:.2f}% | {r["median_ape_percent"]:.2f}% | {r["p90_ape_percent"]:.2f}% |')
    lines+=['','## 准备与搜索成本','',
        '完整[原始成本表](results/preparation_costs.csv)区分原001、8bit增量002和新shape增量003。SparseGPT两种pattern各约305s，HAWQ Hessian约1766s，原DASO prefill标签壁钟约4856s（单GPU），P+D约2583s（先串行、再多卡）；8bit新增标签另计并行壁钟与累计标签工作。质量拟合/搜索只有数秒到十几秒，但标签采集不能省略。历史阶段有并行重叠，不能将全部wall time简单相加为一次串行用时。',
        '003的360个完整forward shape测量约131.5s；47份策略CPU重放4.91s，均复用既有质量模型。从零部署与更换B/S/O的增量成本不同。',
        '精确/反向求解与36/64/80层规模的搜索计时见[run005](../005/results/exact_solver_audit.json)。扩展重复真实8B动作表，只验证搜索复杂度，不代表32B质量。','',
        '## 接口、非Linear与规模','',
        f'- 真实MLP接口检查{interfaces["rows"]}个组合：7×7方法对、M=8/16384。gate/up输出及SiLU后输入均为contiguous BF16；每个Linear内部完成对应激活量化/打包和输出缩放，无压缩激活跨层ABI。逐组件与完整链均实测；链残差还包括cache、launch与同步，不能全算作格式转换。',
        '- 12份独立CPU/CUDA trace只运行1个prefill+3个decode；kernel、H2D与重叠区间保留。类别由kernel名称判定，other明确保留。trace耗时不混入无插桩速度；copy未重叠时长本身也不等于关键路径损失。',
        f'- 完成{large["rows"]}个真实GPU大shape forward，采用[官方Qwen3-32B配置](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)的四种N/K与四种M。特别是Q宽8192而hidden_size=5120，O投影K=8192。权重/激活为合成数据，未部署完整预训练32B模型，不提供32B质量/E2E结论。',
        '- 本机没有30B权重缓存，现有phase运行时TP=1；无法以单卡常驻BF16公平完成30B整模型对照。此限制保留，不以shape微基准替代整模型。','',
        '## 图与原始证据','',
        '![运行时对照](results/figures/runtime_controls.png)','',
        '![广覆盖留出误差](results/figures/latency_validation.png)','',
        '[运行时CSV](results/runtime_summary.csv) · [逐token控制审计](results/runtime_semantics_audit.json) · [MLP接口](results/mlp_interfaces.csv) · [大shape](results/large_shapes.csv) · [profiler汇总](results/profile_summary.json)。',
        '所有原始计时、trace、命令源码快照、失败记录与可重建权重释放记录保留。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    directory=RUN/'results/figures';directory.mkdir(exist_ok=True)
    short={p:p.replace('transfer_','').replace('_prefill_decode_','_pd_') for p in design['policies']}
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
    for ax,w in zip(axes,design['workloads'],strict=True):
        for index,mode in enumerate(design['modes']):
            selected=[r for r in rows if r['workload']==w['id'] and r['mode']==mode]
            xx=[design['policies'].index(r['policy'])+(index-1)*.23 for r in selected]
            ax.bar(xx,[r['cycle_speedup'] for r in selected],width=.23,label=mode)
        ax.axhline(1,color='black',lw=.7);ax.set_title(w['id']);ax.set_xticks(range(6),[short[p] for p in design['policies']],rotation=65,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Full cycle speedup over BF16');axes[-1].legend();fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(directory/f'runtime_controls.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));methods=list(dict.fromkeys(r['method'] for r in broad))
    for offset,split in [(-.18,'shape'),(.18,'nk')]:
        values=[float(next(r for r in broad if r['method']==m and r['split']==split)['mape_percent']) for m in methods]
        ax.bar([i+offset for i in range(len(methods))],values,width=.36,label=split)
    ax.set_xticks(range(len(methods)),methods,rotation=15);ax.set_ylabel('Held-out MAPE (%)');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(directory/f'latency_validation.{ext}',dpi=180)
    plt.close(fig)
    write_json(RUN/'results/report_complete.json',{'complete':True,'runtime_rows':len(rows),'profiles':len(profiles),
        'interfaces':interfaces['rows'],'large_shapes':large['rows'],'semantic_equivalence':True})


if __name__=='__main__':main()
