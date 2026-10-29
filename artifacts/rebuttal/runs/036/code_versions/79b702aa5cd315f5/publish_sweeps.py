"""Append all Llama sweep points in new masters, retaining sealed original tables."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS=Path(__file__).resolve().parents[2]
RESPONSES=RUNS.parent/'responses/reviewer2'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:list(csv.DictReader(p.open()))


def main():
    for rid in ['034','035']:
        root=RUNS/rid;spec=read(root/'configs/model.json')
        audit=read(root/'results/sweep_final_audit.json')
        assert audit['complete']and audit['verified']
        for name,digest in audit['csv_sha256'].items():assert sha(root/'results'/name)==digest
        old_root=RESPONSES/(spec['model_id']+'_table2');old=load(old_root/'main_table.csv')
        assert len(old)==15
        old_by_name={r['method']:r for r in old}
        data={r['policy']:r for r in load(root/'results/sweep_all_results.csv')}
        points=[{'campaign':rid,**p}for p in load(root/'results/sweep_results.csv')]
        supplements={}
        if rid=='034'and(RUNS/'036').exists():
            extra=RUNS/'036';a=read(extra/'results/sweep_final_audit.json')
            assert a['complete']and a['verified']
            for name,digest in a['csv_sha256'].items():assert sha(extra/'results'/name)==digest
            points.extend({'campaign':'036',**p}for p in load(extra/'results/sweep_results.csv'))
            data.update({r['policy']:r for r in load(extra/'results/sweep_all_results.csv')if r['policy']not in data})
            supplements['036']={'audit_sha256':sha(extra/'results/sweep_final_audit.json'),'formal_device':a['formal_device']}
        dest=RESPONSES/(spec['model_id']+'_table2_sweep');dest.mkdir(parents=True,exist_ok=True)
        legacy=dest/'reference/original15';legacy.mkdir(parents=True,exist_ok=True)
        for name in ['README.md','main_table.csv','provenance.json']:
            if not(legacy/name).exists():(legacy/name).write_bytes((old_root/name).read_bytes())
            assert (legacy/name).read_bytes()==(old_root/name).read_bytes()
        fields=['model','method','arc_c','arc_e','wino','mmlu','prefill_avg_acc','prefill_ttft_speedup',
                'cnn_dm','dialogsum','iwslt','pd_avg_score','pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']

        def averages(row):
            return {**row,'prefill_avg_acc':statistics.mean(float(row[k])for k in ['arc_c','arc_e','wino','mmlu']),
                'pd_avg_score':statistics.mean(float(row[k])for k in ['cnn_dm','dialogsum','iwslt'])}

        rows=[averages(r)for r in old];origins=[]
        references=[('uniform_dense_bf16','BF16'),('uniform_int8','INT8 W8A8'),
            ('uniform_sq_int8','SmoothQuant INT8'),('uniform_fp8','FP8 W8A8'),
            ('phase_sq_int8_w4a16_ours','Uniform SQ-INT8 prefill / W4A16 decode')]
        targets=[(pid,method+f' ({rid} reference)',method,rid)for pid,method in references]
        reused_method='DASO-8 measured-LUT + exact/validation epsilon=0.04'
        targets += [(p['policy'],f'DASO-8 exploratory P+D, q={float(p["budget"]):g}, wP={float(p["prefill_weight"]):g} ({p["campaign"]})',reused_method,p['campaign'])for p in points]
        for pid,method,pre_name,campaign in targets:
            d=data[pid];pre=old_by_name[pre_name]
            row={'model':old[0]['model'],'method':method,
                **{k:pre[k]for k in ['arc_c','arc_e','wino','mmlu','prefill_ttft_speedup']},
                **{k:d[k]for k in ['cnn_dm','dialogsum','iwslt']},
                **{target:d[source]for target,source in [('pd_ttft_speedup','ttft_ms_vs_bf16'),
                    ('pd_tpot_speedup','tpot_ms_vs_bf16'),('pd_e2e_speedup','e2e_ms_vs_bf16')]}}
            rows.append(averages(row))
            origins.append({'method':method,'campaign':campaign,'pd_policy':pid,'pd_source':str(RUNS/campaign/'results/sweep_all_results.csv'),
                'prefill_only_reused_method':pre_name,'prefill_only_newly_optimized':False,
                'prefill_only_source':str(old_root/'main_table.csv'),'same_campaign_e2e_vs_int8':float(d['e2e_vs_int8'])})
        assert len(rows)==20+len(points)
        for before,after in zip(old,rows):assert all(after[k]==v for k,v in before.items())
        with(dest/'main_table.csv').open('w')as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        for name in ['sweep_results.csv','sweep_all_results.csv','sweep_formal_rounds.csv']:
            (dest/name).write_bytes((root/'results'/name).read_bytes())
        with(dest/'all_exploratory_points.csv').open('w')as handle:
            writer=csv.DictWriter(handle,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
        for extra_id in supplements:
            directory=dest/'reference'/extra_id;directory.mkdir(parents=True,exist_ok=True)
            for name in ['sweep_results.csv','sweep_all_results.csv','sweep_formal_rounds.csv','sweep_final_audit.json']:
                (directory/name).write_bytes((RUNS/extra_id/'results'/name).read_bytes())
        lines=[f'# {spec["model_id"]}完整Table2：原15行＋{len(rows)-15}行新参考/探索点','',
            f'**原15行的全部数值保留，新增两类Avg为派生值。** 原15行目录与审计不改；本追加汇总列出本轮5个参考和全部{len(points)}个预算/权重点，不只展示获胜点。原六点在本轮GPU输出前冻结；若含036，其为看到部分原组结果后、依据CPU分配独立设计的wP16补充，另行标注探索来源。', '',
            f'原Prefill-only独立结果沿用：新增DASO行复用原`{reused_method}`对应配置。新q/wP只描述P+D；本轮未新增MC或重新优化Prefill-only。参考行也复用各自原Prefill-only四项精度及独立速度。', '',
            f'原组新P+D为GPU{audit["formal_device"]["physical_gpu"]}/NUMA{audit["formal_device"]["numa_node"]}、B8/S2048/O64、KV{spec["evaluation_kv_gib"]}GiB/native decode Graph，三轮3warmup+5timed、首尾BF16。原15行仍使用其原3warmup+8timed；不同campaign各自同轮归一化，不称全部行统一重测。'+('036补充使用GPU'+str(supplements['036']['formal_device']['physical_gpu'])+'，也与自身同轮BF16/INT8比较，未混用绝对延迟。'if supplements else''), '',
            '同map/执行代码/输入协议的原完整生成按SHA复用；新map完整CNN/DM1000、DialogSum1500、IWSLT333。P+D Avg为ROUGE-L%/BLEU三项算术摘要，Prefill Avg为四项准确率等权平均，均由未舍入值计算。', '',
            '| Method | ARC-C | ARC-E | Wino | MMLU | Prefill Avg | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D Avg | P+D TTFT × | TPOT × | E2E × |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:lines.append('| '+row['method']+' | '+' | '.join(f'{float(row[k]):.2f}'for k in fields[2:])+' |')
        lines+=['','## 预算诊断与完整探索','','q是代理/加权偏好预算，不是实测ΔNLL保证。以下NLL来自原模型已评估的固定64个WikiText test窗口，不称新盲测；保护值=mean+2SE+.005。全部预算通过/未通过情况和参数点保留，不据最终任务分数重新训练或宣称通用最优。', '',
            '| 设置 | Gen Avg | ×同轮INT8 | 实测ΔNLL | 保护值 | 名义q | 保护值通过 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for p in points:
            lines.append('| '+p['label']+' | '+' | '.join(f'{float(p[k]):.5f}'for k in
                ['generation_avg','e2e_vs_int8','delta_nll','guarded_delta_nll','budget'])+' | '+p['guard_within_nominal_budget']+' |')
        lines+=['','[主表CSV](main_table.csv) · [全部探索标签及分配](all_exploratory_points.csv) · [原组六标签](sweep_results.csv) · [原组唯一map](sweep_all_results.csv) · [原组三轮速度](sweep_formal_rounds.csv) · [精度—速度图](quality_speed.png) · [PDF](quality_speed.pdf) · [来源](provenance.json)', '',
            f'[原始代码/日志/报告](../../../runs/{rid}/SWEEP_REPORT.md) · [原15行表](../{spec["model_id"]}_table2/README.md)','']
        if supplements:lines+=['[036单点补充报告](../../../runs/036/SWEEP_REPORT.md) · [补充三轮速度](reference/036/sweep_formal_rounds.csv)','']
        (dest/'README.md').write_text('\n'.join(lines))
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        threshold=max(float(data[p]['generation_avg'])for p in ['uniform_int8','uniform_sq_int8','uniform_fp8'])
        ax.axhline(threshold,color='gray',linestyle='--',linewidth=.9,label='Highest 8-bit average')
        ax.axvline(1.,color='gray',linestyle=':',linewidth=.9)
        plot=[(pid,data[pid],None)for pid,_ in references]+[(p['label'],p,float(p['prefill_weight']))for p in points]
        labels={'uniform_dense_bf16':'BF16','uniform_int8':'INT8','uniform_sq_int8':'SQ-INT8',
                'uniform_fp8':'FP8','phase_sq_int8_w4a16_ours':'SQ/W4'}
        for label,row,weight in plot:
            x,y=float(row['e2e_vs_int8']),float(row['generation_avg'])
            color='#555555'if weight is None else '#2469a0'if weight==1 else '#a44e16'
            ax.errorbar(x,y,xerr=[[x-float(row['e2e_vs_int8_min'])],[float(row['e2e_vs_int8_max'])-x]],
                fmt='none',ecolor=color,alpha=.45,capsize=2,linewidth=.8)
            ax.scatter(x,y,marker='s'if weight is None else'o'if weight==1 else'D',color=color,s=30,zorder=3)
            short=labels[label] if weight is None else f'q={row["budget"]}' if weight==1 else f'wP={weight:g}'
            offset={'uniform_sq_int8':(-40,-13),'phase_sq_int8_w4a16_ours':(-36,-13)}.get(label,(4,5))
            ax.annotate(short,(x,y),xytext=offset,textcoords='offset points',fontsize=7)
        ax.set(xlabel='Single-batch E2E speedup over same-campaign INT8',ylabel='Generation average score',title=spec['model_id'])
        ax.grid(alpha=.2);ax.margins(x=.15,y=.2);ax.legend(fontsize=7);fig.tight_layout()
        fig.savefig(dest/'quality_speed.png',dpi=200,bbox_inches='tight');fig.savefig(dest/'quality_speed.pdf',bbox_inches='tight');plt.close(fig)
        meta={'complete':True,'rows':len(rows),'retained_rows':15,'metric_cells':len(rows)*13,
            'source_run':rid,'audit_sha256':sha(root/'results/sweep_final_audit.json'),
            'old_sources_sha256':{name:sha(old_root/name)for name in ['README.md','main_table.csv','provenance.json']},
            'run_csv_sha256':audit['csv_sha256'],'supplemental_audits':supplements,'appended_rows':origins,
            'scope':'Exploratory full sweep; independent Prefill-only reuse; no test-blind or nominal-budget guarantee claim.'}
        meta['delivery_sha256']={p.name:sha(p)for p in dest.iterdir()if p.is_file()and p.name!='provenance.json'}
        (dest/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
        print('Published',spec['model_id'],len(rows),'rows',dest,flush=True)


if __name__=='__main__':main()
