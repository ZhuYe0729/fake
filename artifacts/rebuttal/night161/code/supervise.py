"""Non-invasive status tracker and gated CPU publication for this night's jobs."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs'
read=lambda p:json.loads(p.read_text())
DEADLINE=1793148789


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def action(label,argv):
    path=ROOT/'actions'/f'{label}.json'
    if path.exists():return
    path.parent.mkdir(exist_ok=True)
    log=ROOT/'actions'/f'{label}.log';start=time.time()
    with log.open('w')as handle:
        result=subprocess.run(argv,stdout=handle,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/daso_night161_mpl'})
    write(path,{'returncode':result.returncode,'command':list(map(str,argv)),'start_unix':start,'seconds':time.time()-start,
        'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
    print('CPU action',label,'exit',result.returncode,flush=True)


def main():
    os.sched_setaffinity(0,set(range(48,56)));previous=None
    while time.time()<DEADLINE:
        subprocess.run([sys.executable,ROOT/'code/cache_sweep_exports.py'],stdout=subprocess.DEVNULL,check=True)
        if all((RUNS/r/'results/sweep_final_audit.json').exists()for r in ['034','035','036']):
            action('publish_llama_masters',[sys.executable,RUNS/'034/code/publish_sweeps.py'])
            action('new_point_overheads_final',[sys.executable,ROOT/'code/report_new_overheads.py','--final'])
        if all((RUNS/r/('results/sweep_final_audit.json'if r in ['038','039']else'results/final_audit.json')).exists()
            for r in ['038','039','042','043']):
            action('publish_complete_llama',[sys.executable,ROOT/'code/publish_complete_llama.py'])
        audit=RUNS/'037/results/final_audit.json'
        if audit.exists()and read(audit)['all_token_sequences_exact']and read(audit)['all_nll_within_original_gate']:
            for model in ['llama31','llama2']:
                action('bootstrap_'+model,[sys.executable,ROOT/'code/bootstrap_generation.py','--model',model])
        if (RUNS/'048/results/final_audit.json').exists():
            for model in ['qwen3','llama31','llama2']:
                action('bootstrap_all_'+model,[sys.executable,ROOT/'code/bootstrap_all_baselines.py','--model',model])
        states=[]
        for rid in ['034','035','036','037','038','039','040','041','042','043','044','045','046','047','048']:
            run=RUNS/rid;state={}
            for name in ['results/sweep_worker.json','results/worker.json']:
                if(run/name).exists():state=read(run/name);break
            errors=[{'file':str(p.relative_to(run)),'error':read(p)}for p in(run/'results').glob('*failure*.json')]
            states.append({'run':rid,**state,'retained_failures':errors})
        now=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
        write(ROOT/'status.json',{'updated':now,'deadline_utc':'2026-10-28 00:53:09 UTC','runs':states})
        signature=json.dumps(states,sort_keys=True)
        if signature!=previous:
            previous=signature;print(now,json.dumps(states,ensure_ascii=False),flush=True)
            with(ROOT/'status_history.jsonl').open('a')as handle:handle.write(json.dumps({'updated':now,'runs':states})+'\n')
            lines=['# 夜间补测进展（161）','',f'更新：{now}。计划窗口至2026-10-28 00:53 UTC，继续监督实际完成状态；排队不等于完成。','',
                '| Run | 内容 | 当前状态 | 阶段 | 进度 |','| --- | --- | --- | --- | --- |']
            labels=['Llama3.1六个预算/权重点','Llama2六个预算/权重点','Llama3.1单点wP16','两Llama基线数值复验',
                'Llama3.1固定阶段控制','Llama2固定阶段控制','Qwen新点六负载迁移','Llama3.1同候选/反约束18点',
                'Llama3.1 SQ-FP8完整基线','Llama2 SQ-FP8完整基线','Qwen新窗口预算复核',
                'Llama3.1新窗口预算复核','Llama2新窗口预算复核','Qwen新点真实mixed调度','Qwen十配置统一生成重复']
            for row,label in zip(states,labels,strict=True):
                lines.append(f'| [{row["run"]}](../runs/{row["run"]}/README.md) | {label} | {row.get("status","pending")} | {row.get("stage","")} | {row.get("complete",row.get("index",""))}/{row.get("total","")} |')
            lines+=['','[Llama3.1追加主表](../responses/reviewer2/llama31_8b_instruct_table2_sweep/README.md) · [Llama2追加主表](../responses/reviewer2/llama2_7b_chat_table2_sweep/README.md) · [Qwen统计区间](../responses/reviewer2/quality_uncertainty_night161/qwen3/README.md)','',
                '主表链接仅在发布后存在。所有原始点/失败记录保留；探索点的代理q不视为实际精度保证；固定控制不标为DASO自动选择。','']
            (ROOT/'TRACKER.md').write_text('\n'.join(lines))
        time.sleep(30)
    print('Nine-hour monitoring window elapsed; inspect any unfinished campaigns before stopping.',flush=True)


if __name__=='__main__':main()
