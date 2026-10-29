"""Simple pruning controls with the same real structured sparse kernel."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import RUN,write_json


def main():
    assert json.loads((RUN/'results/evaluation_worker.json').read_text())['status']=='complete'
    formula=json.loads((RUN/'results/pruning_formula_audit.json').read_text());assert formula['passed']
    rows=[];rng=np.random.default_rng(144961);bootstrap=rng.integers(0,64,(4000,64))
    for scenario in ['prefill_only','prefill_decode']:
        reference=json.loads((RUN/f'results/test/dense_bf16/{scenario}.json').read_text())
        baseline=json.loads((RUN/f'results/test_speed/dense_bf16/{scenario}.json').read_text())
        for method in ['dense_bf16','magnitude','wanda','sparsegpt']:
            r=json.loads((RUN/f'results/test/{method}/{scenario}.json').read_text());speed=json.loads((RUN/f'results/test_speed/{method}/{scenario}.json').read_text())
            assert r['quality_score_batch']==8 and len(r['blocks'])==64 and speed['gpu']=='5'
            delta=np.array([a['avg_nll']-b['avg_nll'] for a,b in zip(r['blocks'],reference['blocks'],strict=True)])
            lo,hi=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            rows.append({'method':method,'scenario':scenario,'nll':r['avg_nll'],'delta_nll':float(delta.mean()),
                'delta_nll_ci_low':float(lo),'delta_nll_ci_high':float(hi),'latency_ms':speed['median_ms'],
                'speedup':baseline['median_ms']/speed['median_ms'],'blocks':64,'quality_batch':8,'speed_gpu':5})
    with (RUN/'results/summary_final.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    prep=[]
    for method in ['magnitude','wanda']:
        r=json.loads((RUN/f'results/canonical_{method}.json').read_text())
        prep.append({'method':method,'calibration_blocks':r['sample_count'],'load_seconds':r['load_seconds'],
            'pruning_calibration_seconds':r['pruning_and_calibration_seconds'],'total_including_save_seconds':r['total_seconds_including_save'],
            'modules':len(r['modules'])})
    old=json.loads((RUN.parent/'001/results/canonical_sparse_bf16.json').read_text())
    prep.append({'method':'sparsegpt','calibration_blocks':old['sample_count'],'total_before_save_seconds':old['elapsed_seconds'],
        'modules':len(old['modules']),'source':'run001 immutable canonical BF16 state; save accounting differs'})
    write_json(RUN/'results/preparation_summary.json',prep)
    lines=['# Run007：magnitude / Wanda / SparseGPT结构化剪枝对照','',
        '三种剪枝均为每输出行连续4个输入权重保留至多2个非零，使用相同真实cuSPARSELt sparse BF16 kernel；转换时prune=False，不能偷偷再剪一次。Embedding、LM head和其余非Linear模块保持原模型。BF16为同卡锚点。','',
        'Magnitude不使用校准数据；Wanda与原SparseGPT使用相同128个WikiText训练块、每块前2048 token，并在每个pruned decoder layer之后重新传播校准激活。Wanda分数为abs(weight)×sqrt(input平方和/sample数)。',
        '这是官方Wanda规则的Qwen3适配和相同数据对照，不是原论文C4设置的准确率复现。仅BF16普通2:4，未把它冒充Sparse NVFP4所需的pairwise4:8。','',
        '## 实际速度与质量','',
        'GPU5、B8/S2048、O=1或64、6GiB KV、2 warmups+5 timed。64个独立WikiText test非重叠窗口；prefill实际B8 prompt-logprob，decode实际B8 teacher forcing。完整上下文/目标、seed和起点预先固定。','',
        '| 方法 | 场景 | NLL | ΔNLL | 描述性95%区间 | ms | 相对BF16 |','| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for r in rows:lines.append(f'| {r["method"]} | {r["scenario"]} | {r["nll"]:.5f} | {r["delta_nll"]:.5f} | [{r["delta_nll_ci_low"]:.5f}, {r["delta_nll_ci_high"]:.5f}] | {r["latency_ms"]:.1f} | {r["speedup"]:.3f}× |')
    lines+=['','## 准备成本与算法检查','',
        '| 方法 | 校准块 | 总时间s | 口径 |','| --- | ---: | ---: | --- |']
    for r in prep:lines.append(f'| {r["method"]} | {r["calibration_blocks"]} | {r.get("total_including_save_seconds",r.get("total_before_save_seconds")):.2f} | '+('含模型载入、校准/剪枝、保存' if r['method']!='sparsegpt' else '001原记录，保存前；继承canonical权重')+' |')
    lines+=['',
        f'- 与固定commit官方WrappedGPT norm accumulator的最大相对差{formula["max_relative_norm_accumulator_difference"]:.3g}；向量化组内topk与官方逐组loop的mask完全一致。',
        '- magnitude/Wanda全模型各252个Linear的mask均为精确50%选中、每4个至多2个非零。准备代码与官方源、MIT license、commit与SHA一并归档。',
        '- 同样稀疏kernel并不意味着相同质量；此实验把剪枝准备算法与硬件格式分开。DASO的异构权衡应与这些完整uniform点共同看待，不能只拿同压缩率或理论FLOPs比较。',
        '- 新magnitude/Wanda完整下游五个likelihood任务按预定规则在run010执行；SparseGPT/BF16原完整任务作为相同权重锚点继承。不得用本轮NLL代替MMLU等任务。',
        '- 64个窗口在test split内不重叠；bootstrap区间为描述性配对窗口区间，不是跨分布质量保证。不同数据起点与B8协议应与001/002结果分开解释。','',
        '![简单剪枝对照](results/figures/pruning_quality_speed.png)','',
        '[完整CSV](results/summary_final.csv) · [准备成本](results/preparation_summary.json) · [公式/结构检查](results/pruning_formula_audit.json) · [官方源commit](reference/wanda/provenance.json)。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n');figures=RUN/'results/figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,scenario in zip(axes,['prefill_only','prefill_decode'],strict=True):
        for r in rows:
            if r['scenario']!=scenario:continue
            ax.scatter(r['delta_nll'],r['speedup'],label=r['method'],s=55)
        ax.axhline(1,color='black',lw=.7);ax.set_title(scenario);ax.set_xlabel('Measured delta NLL');ax.set_ylabel('Speedup over BF16');ax.grid(alpha=.2)
    axes[1].legend();fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'pruning_quality_speed.{ext}',dpi=180)
    plt.close(fig)
    write_json(RUN/'results/report_complete.json',{'complete':True,'methods':4,'rows':8,'formula_and_patterns_passed':True,
        'quality_blocks':512,'same_gpu_speed':5,'full_downstream_results':'run010'})


if __name__=='__main__':main()
