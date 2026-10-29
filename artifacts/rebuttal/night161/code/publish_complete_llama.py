"""Append fixed phase controls and SQ-FP8 in new complete Llama master tables."""
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs';RESP=ROOT.parent/'responses/reviewer2'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:list(csv.DictReader(p.open()))


def main():
    for model,phase,baseline,expected in [('llama31_8b_instruct','038','042',30),('llama2_7b_chat','039','043',29)]:
        core=RESP/(model+'_table2_sweep');rows=load(core/'main_table.csv');original=[dict(r)for r in rows]
        core_meta=read(core/'provenance.json')
        assert sha(core/'main_table.csv')==core_meta['delivery_sha256']['main_table.csv']
        fields=list(rows[0]);legacy={r['method']:r for r in rows[:15]}
        stage=RUNS/phase;stage_audit=read(stage/'results/sweep_final_audit.json');assert stage_audit['verified']
        for name,digest in stage_audit['csv_sha256'].items():assert sha(stage/'results'/name)==digest
        phase_rows={r['policy']:r for r in load(stage/'results/sweep_all_results.csv')}
        phase_dest=RESP/(model+'_phase_controls_run'+phase);phase_dest.mkdir(parents=True,exist_ok=True)
        for name in ['sweep_all_results.csv','sweep_results.csv','sweep_formal_rounds.csv']:
            target=phase_dest/'results'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(stage/'results'/name,target)
        (phase_dest/'configs').mkdir(exist_ok=True);shutil.copy2(stage/'configs/sweep_design.json',phase_dest/'configs/sweep_design.json')
        shutil.copy2(stage/'results/sweep_final_audit.json',phase_dest/'provenance.json')
        (phase_dest/'README.md').write_text((stage/'SWEEP_REPORT.md').read_text()+f'\n[原始代码与完整记录](../../../runs/{phase}/README.md)\n')
        sources={}
        for pid,prefill,name in [('phase_int8_w4a16_ours','INT8 W8A8','Uniform INT8 prefill / W4A16 decode'),
            ('phase_fp8_w4a16_ours','FP8 W8A8','Uniform FP8 prefill / W4A16 decode')]:
            value=phase_rows[pid];row={**legacy[prefill],'method':name+f' ({phase} fixed control)',
                **{k:value[k]for k in ['cnn_dm','dialogsum','iwslt']},
                'pd_avg_score':value['generation_avg'],'pd_ttft_speedup':value['ttft_ms_vs_bf16'],
                'pd_tpot_speedup':value['tpot_ms_vs_bf16'],'pd_e2e_speedup':value['e2e_ms_vs_bf16']}
            rows.append(row);sources[row['method']]={'pd_run':phase,'policy':pid,'role':'Fixed control, not new DASO solver output',
                'prefill_only_reuse':prefill,'prefill_only_source':str(core/'reference/original15/main_table.csv')}
        base=RUNS/baseline;baseline_audit=read(base/'results/final_audit.json');assert baseline_audit['verified']
        baseline_folder=RESP/(model+'_sqfp8_run'+baseline)
        assert sha(baseline_folder/'baselines.csv')==baseline_audit['delivery_sha256']['baselines.csv']
        value=next(r for r in load(base/'results/baselines.csv')if r['policy']=='uniform_sq_fp8')
        row={k:value[k]for k in fields};row['model']=rows[0]['model'];row['method']='SmoothQuant FP8 alpha=.5 ('+baseline+')'
        rows.append(row);sources[row['method']]={'run':baseline,'policy':'uniform_sq_fp8','role':'Strong baseline, all MC/GEN new, same-campaign speed'}
        assert len(rows)==expected
        for before,after in zip(original,rows):assert before==after
        for row in rows:
            assert abs(float(row['prefill_avg_acc'])-statistics.mean(float(row[k])for k in ['arc_c','arc_e','wino','mmlu']))<1e-10
            assert abs(float(row['pd_avg_score'])-statistics.mean(float(row[k])for k in ['cnn_dm','dialogsum','iwslt']))<1e-10
        dest=RESP/(model+'_table2_night161');dest.mkdir(parents=True,exist_ok=True)
        with(dest/'main_table.csv').open('w')as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        snapshot=dest/'reference/before_phase_sqfp8';snapshot.mkdir(parents=True,exist_ok=True)
        for name in ['README.md','main_table.csv','provenance.json']:shutil.copy2(core/name,snapshot/name)
        lines=[f'# {rows[0]["model"]}完整Table2：{expected}行（R1/R2/R3/R5）','',
            f'原15行及预算扫点追加表的全部数值保留；再补两个固定阶段控制和SQ-FP8强基线。前{len(original)}行来自既有追加汇总，不是所有行重新统一测量。各速度使用各自同轮BF16归一化，完整来源另列。','',
            '新固定控制的Prefill-only四项精度/独立速度分别复用原uniform INT8或FP8结果，P+D使用新阶段配置实测；它们不是DASO自动求解结果。SQ-FP8的MC/GEN均完整新测、两场景速度同轮新测。原DASO q/wP只描述相应P+D探索点，真实NLL预算通过情况见原扫点和后续独立窗口复核，不能把偏好q称为严格实际epsilon保证。','',
            '| 方法 | ARC-C | ARC-E | Wino | MMLU | Prefill Avg | TTFT × | CNN/DM | DialogSum | IWSLT | P+D Avg | P+D TTFT × | TPOT × | E2E × |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:lines.append('| '+r['method']+' | '+' | '.join(f'{float(r[k]):.2f}'for k in fields[2:])+' |')
        lines+=['','Avg由未舍入值计算：Prefill为四项准确率等权摘要，P+D为ROUGE-L%/BLEU三项摘要，二者不是统一准确率。HAWQ为PTQ硬件移植；固定控制和探索点均保留，不能据此宣称所有模型/负载上DASO全面最优。','',
            f'[完整CSV](main_table.csv) · [来源](provenance.json) · [原追加表](../{model}_table2_sweep/README.md) · [七阶段配置完整对照](../{phase_dest.name}/README.md) · [五个8bit/BF16基线同轮对照](../{baseline_folder.name}/README.md)','']
        (dest/'README.md').write_text('\n'.join(lines))
        meta={'complete':True,'rows':expected,'retained_rows':len(original),'retained_numeric_cells_unchanged':True,
            'core_provenance_sha256':sha(core/'provenance.json'),'core_table_sha256':sha(core/'main_table.csv'),
            'phase_audit_sha256':sha(stage/'results/sweep_final_audit.json'),'baseline_audit_sha256':sha(base/'results/final_audit.json'),
            'appended_rows':sources,'code_sha256':sha(Path(__file__))}
        meta['delivery_sha256']={str(p.relative_to(dest)):sha(p)for p in dest.rglob('*')if p.is_file()and p.name!='provenance.json'}
        (dest/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
        print('Published complete master',model,len(rows),'rows',dest,flush=True)


if __name__=='__main__':main()
