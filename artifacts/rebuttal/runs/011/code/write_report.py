"""All frozen baseline and verified points, with common timing anchors."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    assert read(RUN/'results/worker_speed.json')['status']=='complete'
    quiet=(RUN/'configs/quiet_campaign.json').exists()
    if quiet:assert read(RUN/'results/worker_quiet.json')['status']=='complete'
    speed_folder='calibration_speed_quiet' if quiet else 'calibration_speed'
    pool=read(RUN/'configs/candidate_pool.json');design=read(RUN/'configs/design.json')
    assert digest(RUN/'configs/design.json')==pool['design_sha256']
    assert digest(RUN.parent/'005/configs/selected_policies.json')==design['source_selection_sha256']
    assert digest(RUN.parent/'009/configs/candidate_pool.json')==design['source_hawq_pool_sha256']
    inherited=read(RUN/'reference/inherited_quality.json')
    for r in inherited['records']:assert digest(RUN/r['target'])==r['sha256']==digest(Path(r['source']))
    for pid,info in pool['policies'].items():
        p=read(RUN/f'configs/policies/{pid}.json')
        assert hashlib.sha256(json.dumps(p['method_map'],sort_keys=True).encode()).hexdigest()==info['signature']
    rows=[];brackets=[];rng=np.random.default_rng(144987);resample=rng.integers(0,64,(4000,64))
    initial={}
    if quiet:
        with (RUN/'results/diagnostics_first_campaign/summary_final.csv').open() as f:
            initial={(r['policy_id'],r['scenario']):r for r in csv.DictReader(f)}
    for scenario in ['prefill_only','prefill_decode']:
        pair=[read(RUN/f'results/{speed_folder}/{pid}/{scenario}.json') for pid in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']]
        reference=float(np.median([x['elapsed_ms'] for r in pair for x in r['iterations'] if not x['warmup']]))
        brackets.append({'scenario':scenario,'start_ms':pair[0]['median_ms'],'end_ms':pair[1]['median_ms'],
            'end_over_start':pair[1]['median_ms']/pair[0]['median_ms'],'pooled_10_bf16_ms':reference})
        qbase=read(RUN/f'results/test/uniform_dense_bf16/{scenario}.json')
        for point in pool['points']:
            if point['scenario']!=scenario:continue
            pid=point['policy_id'];s=read(RUN/f'results/{speed_folder}/{pid}/{scenario}.json');q=read(RUN/f'results/test/{pid}/{scenario}.json')
            assert s['gpu']=='0' and s['seed']==144983 and len(s['iterations'])==7
            assert q['data_sha256']==qbase['data_sha256'] and len(q['blocks'])==64 and q['quality_score_batch']==8
            values=[x['elapsed_ms'] for r in (pair if pid=='uniform_dense_bf16' else [s]) for x in r['iterations'] if not x['warmup']]
            delta=np.array([a['avg_nll']-b['avg_nll'] for a,b in zip(q['blocks'],qbase['blocks'],strict=True)])
            ci=np.quantile(delta[resample].mean(1),[.025,.975]);median=float(np.median(values))
            row={k:v for k,v in point.items() if k not in ['validation','solver']}
            row.update({'measured_delta_nll':float(delta.mean()),'delta_ci_low':float(ci[0]),'delta_ci_high':float(ci[1]),
                'measured_ms':median,'bf16_ms':reference,'speedup':reference/median,
                'speedup_using_start_bf16':pair[0]['median_ms']/median,'speedup_using_end_bf16':pair[1]['median_ms']/median,
                'min_timed_ms':min(values),'max_timed_ms':max(values),'timed_bursts':len(values),'speed_gpu':0,'quality_blocks':64})
            if 'quality_budget' in row:row['quality_budget_violation']=max(0,row['measured_delta_nll']-row['quality_budget'])
            if quiet:
                prior=initial[pid,scenario];row.update({'initial_measured_ms':float(prior['measured_ms']),
                    'initial_speedup':float(prior['speedup']),'quiet_over_initial_ms':median/float(prior['measured_ms'])})
            rows.append(row)
    with (RUN/'results/summary_final.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
    write(RUN/'results/bf16_brackets.json',brackets);frontiers=[]
    for scenario in ['prefill_only','prefill_decode']:
        for cap in [.005,.01,.02,.04,.07]:
            for group in ['uniform','fixed DASO','HAWQ INT4','HAWQ FP4','DASO plus validation']:
                v=[r for r in rows if r['scenario']==scenario and r['measured_delta_nll']<=cap and
                    ((r['family']==group) if not group.startswith('HAWQ ') else r.get('format')==('marlin_int4' if group=='HAWQ INT4' else 'fp4_phase_pair'))]
                best=max(v,key=lambda r:r['speedup']) if v else None
                frontiers.append({'scenario':scenario,'actual_delta_cap':cap,'group':group,
                    'speedup':best['speedup'] if best else None,'label':best['label'] if best else None})
    write(RUN/'results/quality_matched_frontiers.json',frontiers)
    lines=['# Run011：验证后策略与强化HAWQ的同卡标准负载复核','',
        '005验证后策略与009强化基线此前分别在GPU2/GPU0计时。本轮把009全部35比较点与005全部8个质量预算验证选择合并，同一GPU0计时；19份唯一map、23个唯一policy/scenario，所有43个标签保留。没有新策略、质量拟合或规则调整。','',
        '质量直接继承005/009同64个test窗口、B8 prompt-logprobs/真实KV teacher forcing的结果，逐SHA核验，重复map的均值也核对一致。所有canonical/scales/kernel相同。这里新增的是确认速度，不把再次计时称为新的独立质量测试。','',
        '速度为B8/S2048/O1或64、eager、无chunked/prefix cache、6GiB KV、2 warmups+5 timed，统一新seed144983，BF16开头/结尾各一次，合并10个timed burst作锚。','',
        '首轮固定seed144985打乱方法顺序，但部分时段与另一GPU上的006共享host。全部其他GPU实验结束后，将同一19map完整反向顺序再测一次，作为本报告最终主结果；不混合两轮环境。首轮25速度文件、完整报告/CSV/图保留在diagnostics_first_campaign，质量与策略完全不改，也不按输赢继续增加重复。','',
        '| 场景 | BF16起点ms | 终点ms | 终点/起点 | 合并10轮ms |','| --- | ---: | ---: | ---: | ---: |']
    for r in brackets:lines.append(f'| {r["scenario"]} | {r["start_ms"]:.1f} | {r["end_ms"]:.1f} | {r["end_over_start"]:.3f} | {r["pooled_10_bf16_ms"]:.1f} |')
    lines+=['','| 场景 | 比较点 | 实测ΔNLL | 描述性95%窗口区间 | E2E ms | 相对BF16 |','| --- | --- | ---: | --- | ---: | ---: |']
    for r in rows:lines.append(f'| {r["scenario"]} | {r["label"]} | {r["measured_delta_nll"]:.5f} | [{r["delta_ci_low"]:.5f}, {r["delta_ci_high"]:.5f}] | {r["measured_ms"]:.1f} | {r["speedup"]:.3f}× |')
    if quiet:
        lines+=['','## 时段复核与排名变化','',
            '首轮与最后单GPU作业时段的部分phase策略有明显变化，不能据主表单次中位数宣称跨时段稳定最优。CSV同时列出两轮ms、加速及其比值；首轮完整点均保留。下表只列P+D主要phase点以明确展示排序反转。','',
            '| P+D策略 | 首轮加速 | 最后单GPU作业加速 | 后轮/首轮ms |','| --- | ---: | ---: | ---: |']
        for r in rows:
            if r['scenario']=='prefill_decode' and (r['family']=='fixed DASO' or r['policy_id'].startswith('hawq_fp4_phase_pair_signed_')):
                lines.append(f'| {r["label"]} | {r["initial_speedup"]:.3f}× | {r["speedup"]:.3f}× | {r["quiet_over_initial_ms"]:.3f} |')
        lines+=['',
            '例如五候选DASO约1.247→1.438×，HAWQ全NV4/W4控制约1.396→1.198×，排序反转。其他GPU作业/host传输之外，缓存、时钟和分配也可能影响时段差异，本轮未分别建立因果解释。006中B16异长mixed两轮均约1.54×的证据更适合支持稳定的负载优势；这里完整保留条件和波动。']
    lines+=['','## 适用边界','',
        '- DASO+validation是原算法之外新增的候选实测验证阶段，其额外离线成本约38.7分钟wall（该次三GPU并行选择，含I/O/初始化）见005；不能将效果归于原文已有质量保证。',
        '- .005/.01/.02/.04的8个验证预算点包含重复选择。很紧的P+D预算回退BF16；另有预算.04点在test均值低于.01，也不能将它称为部署前epsilon=.01验证规则选出的策略。',
        '- 质量上限切片是已测点的事后Pareto比较。相同实际ΔNLL上限不等于同一预测约束、同一准备成本或全局最优；全部Hessian sign/clip/abs变体与min-cost clamp明确保留。',
        '- BF16起终点、所有计时min/max以及使用各端点归一化的速度均归档；小差距和时间漂移不能解释成统计显著优势。',
        '- HAWQ为PTQ精度分配移植，不是原CNN QAT/TVM全栈复现。真实mixed调度另见006，完整下游质量另见010；本轮没有30B整模型实验。','',
        '![统一标准负载](results/figures/matched_standard.png)','',
        '[全部43点CSV](results/summary_final.csv) · [全部质量上限切片](results/quality_matched_frontiers.json) · [冻结比较池](configs/candidate_pool.json) · [质量来源](reference/inherited_quality.json)。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    directory=RUN/'results/figures';directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,scenario in zip(axes,['prefill_only','prefill_decode'],strict=True):
        for group,marker in [('uniform','x'),('fixed DASO','*'),('HAWQ INT4','o'),('HAWQ FP4','s'),('DASO plus validation','D')]:
            v=[r for r in rows if r['scenario']==scenario and (r['family']==group if not group.startswith('HAWQ ') else r.get('format')==('marlin_int4' if group=='HAWQ INT4' else 'fp4_phase_pair'))]
            ax.scatter([r['measured_delta_nll'] for r in v],[r['speedup'] for r in v],label=group,marker=marker,s=50)
        ax.axhline(1,color='black',lw=.7);ax.axvline(.01,color='black',lw=.7,ls=':');ax.set_title(scenario)
        ax.set_xlabel('Measured test delta NLL (inherited)');ax.set_ylabel('Speedup over same-card BF16');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(directory/f'matched_standard.{ext}',dpi=180)
    plt.close(fig)
    write(RUN/'results/report_complete.json',{'complete':True,'comparison_points':43,'unique_maps':19,
        'unique_quality_files':23,'new_quality_labels':0,'speed_files_including_bf16_end':25,'speed_gpu':0,
        'inherited_quality_verified':True,'quiet_host_primary':quiet,'initial_diagnostic_speed_files':25,
        'source_selection_sha256':design['source_selection_sha256']})


if __name__=='__main__':main()
