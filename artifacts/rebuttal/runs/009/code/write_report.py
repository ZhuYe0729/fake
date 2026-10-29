"""Keep every predeclared HAWQ sign/format control and measured outcome."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import RUN,write_json


def main():
    for role in ['speed','quality']:assert json.loads((RUN/f'results/worker_{role}.json').read_text())['status']=='complete'
    pool=json.loads((RUN/'configs/candidate_pool.json').read_text());sign=json.loads((RUN/'results/hessian_sign_audit.json').read_text())
    rows=[];cache={};rng=np.random.default_rng(144971);bootstrap=rng.integers(0,64,(4000,64))
    for point in pool['points']:
        pid=point['policy_id'];scenario=point['scenario'];key=pid,scenario
        if key not in cache:
            r=json.loads((RUN/f'results/test/{pid}/{scenario}.json').read_text());base=json.loads((RUN/f'results/test/uniform_dense_bf16/{scenario}.json').read_text())
            speed=json.loads((RUN/f'results/calibration_speed/{pid}/{scenario}.json').read_text());baseline=json.loads((RUN/f'results/calibration_speed/uniform_dense_bf16/{scenario}.json').read_text())
            assert len(r['blocks'])==64 and r['quality_score_batch']==8 and speed['gpu']=='0'
            delta=np.array([a['avg_nll']-b['avg_nll'] for a,b in zip(r['blocks'],base['blocks'],strict=True)])
            ci=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            cache[key]={'actual_delta_nll':float(delta.mean()),'delta_ci_low':float(ci[0]),'delta_ci_high':float(ci[1]),
                'actual_ms':speed['median_ms'],'speedup':baseline['median_ms']/speed['median_ms'],'speed_gpu':0}
        rows.append({**point,**cache[key]})
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (RUN/'results/summary_final.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    lines=['# Run009：HAWQ敏感度符号与INT4/FP4移植稳健性','',
        f'冻结Hessian平均特征值中{sign["negative_blocks"]}/{sign["blocks"]}项为负。原001/003 signed处理保持不变，本轮额外固定max(h,0)与abs(h)控制；不依据实测NLL选择处理方式。35条预定比较，完整map合并为16份模型。','',
        'INT4列为Marlin INT4 / INT8 / BF16，precision绑定prefill/decode；FP4列沿用001移植的4bit精度对：prefill dense NVFP4、decode W4A16 NVFP4，另有INT8/BF16。FP4 weight-error代理沿用001 dense NVFP4量化误差，不将其写成原CNN integer-only QAT/TVM流程。三种Hessian处理在原unfused block上执行后再合并成runtime fused module。',
        '相同profile、相同原DASO预测Linear latency cap，两个预算标签0.01/0.07只是cap的来源，不是HAWQ的实际NLL保证。若低于该HAWQ动作集合的最小成本，明确记录infeasible并clamp。策略在质量测量前固定。','',
        '## 全部预定点','',
        '速度统一GPU0、B8/S2048/O1或64、2 warmups+5 timed、6GiB KV；质量同一64个非重叠test窗口，B8 prompt logprobs或实际teacher-forced decode。原DASO固定策略和uniform锚点在本轮统一重测，不能将003/009跨run绝对延迟混用。','',
        '| 场景 | 策略 | 实测ΔNLL | 描述性95%区间 | ms | 加速 | 请求cap不可行 |','| --- | --- | ---: | --- | ---: | ---: | --- |']
    for r in rows:lines.append(f'| {r["scenario"]} | {r["label"]} | {r["actual_delta_nll"]:.5f} | [{r["delta_ci_low"]:.5f}, {r["delta_ci_high"]:.5f}] | {r["actual_ms"]:.1f} | {r["speedup"]:.3f}× | {r.get("infeasible_requested_limit","—")} |')
    frontiers=[]
    groups=['uniform','fixed DASO']+[f'{fmt}/{transform}' for fmt in ['marlin_int4','fp4_phase_pair'] for transform in ['signed','clip_zero','absolute']]
    lines+=['','## 相同实际质量上限内的已测点','',
        '以下是事后实际质量切片；不是从test中挑一种Hessian处理重新部署，所有原点均在上表和CSV中保留。','',
        '| 场景 | 实际ΔNLL cap | 组 | 最快已测加速 | 策略 |','| --- | ---: | --- | ---: | --- |']
    for scenario in ['prefill_only','prefill_decode']:
        for cap in [.01,.02,.04,.07]:
            for group in groups:
                candidates=[r for r in rows if r['scenario']==scenario and r['actual_delta_nll']<=cap and
                    (r['family']==group if group in ['uniform','fixed DASO'] else r.get('format','')+'/'+r.get('sensitivity_transform','')==group)]
                best=max(candidates,key=lambda r:r['speedup']) if candidates else None
                value={'scenario':scenario,'actual_quality_cap':cap,'group':group,'speedup':best['speedup'] if best else None,'label':best['label'] if best else None};frontiers.append(value)
                lines.append(f'| {scenario} | {cap:g} | {group} | '+(f'{best["speedup"]:.3f}× | {best["label"]} |' if best else '— | — |'))
    write_json(RUN/'results/quality_matched_frontiers.json',frontiers)
    lines+=['','## 解释范围','',
        '- 强化控制用于排除Hessian负值处理及实际4bit后端选择造成的移植偶然性；不把signed/clip/abs中对test最有利者当作事先选定的算法。',
        '- 使用相同原Hessian样本和权重量化误差；没有重训、裁剪测试标签、对global NLL重新拟合Hessian目标。',
        '- HAWQ优化其权重敏感度代理，DASO使用校准NLL代理；两者可行性都应看最终实际NLL。HAWQ的ILP最优不等于实际质量全局最优，DASO也不作此保证。',
        '- BF16 attention/KV与非Linear保持相同vLLM路径，INT8为真实W8A8，FP4/INT4为实际对应kernel；本轮只测中心负载，扩展B/S/O与mixed证据分别见003/006。',
        '- 全部源、参数、数值目标、最小可行cap与失败记录保留。已有001–003档案与共享运行时完整性由008的protected_integrity审计核对。','',
        '![HAWQ稳健性](results/figures/hawq_robustness.png)','',
        '[完整CSV](results/summary_final.csv) · [Hessian符号检查](results/hessian_sign_audit.json) · [冻结策略](configs/candidate_pool.json)。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n');figures=RUN/'results/figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));colors={'signed':'#5f7ea6','clip_zero':'#b36c35','absolute':'#6a9959'}
    for ax,scenario in zip(axes,['prefill_only','prefill_decode'],strict=True):
        for family in ['uniform','fixed DASO']:
            values=[r for r in rows if r['scenario']==scenario and r['family']==family]
            ax.scatter([r['actual_delta_nll'] for r in values],[r['speedup'] for r in values],s=50,label=family,marker='x' if family=='uniform' else '*')
        for fmt,marker in [('marlin_int4','o'),('fp4_phase_pair','s')]:
            for transform,color in colors.items():
                values=[r for r in rows if r['scenario']==scenario and r.get('format')==fmt and r.get('sensitivity_transform')==transform]
                ax.scatter([r['actual_delta_nll'] for r in values],[r['speedup'] for r in values],s=35,c=color,marker=marker,label=fmt+'/'+transform)
        ax.axhline(1,color='black',lw=.7);ax.set_title(scenario);ax.set_xlabel('Measured delta NLL');ax.set_ylabel('Speedup over BF16');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'hawq_robustness.{ext}',dpi=180)
    plt.close(fig)
    write_json(RUN/'results/report_complete.json',{'complete':True,'points':len(rows),'unique_policies':len(pool['policies']),
        'unique_policy_scenarios':len(cache),'speed_gpu':0,'quality_batch':8,'all_transforms_retained':True})


if __name__=='__main__':main()
