"""Fixed sample-size comparison with new, never-selected-on confirmation blocks."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
SOURCE=RUN.parent/'018'


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def invoke(pid,split,gpu):
    command=['numactl','--all',f'--physcpubind={gpu*8}-{gpu*8+7}',f'--membind={0 if gpu<4 else 1}',
        sys.executable,str(RUN/'code/score_worker.py'),'--policy',pid,'--split',split]
    log=RUN/f'logs/{split}_{pid}_{time.time_ns()}.log'
    write(log.with_suffix('.command.json'),{'command':command,'gpu':gpu,'worker_sha256':sha(RUN/'code/score_worker.py')})
    started=time.perf_counter()
    with log.open('w') as handle:
        result=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=handle,stderr=subprocess.STDOUT)
    write(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-started})
    assert result.returncode==0,str(log)


def worker(split,gpu):
    policies=read(RUN/'configs/design.json')['policies'] if split=='validation' else read(RUN/'configs/confirmation_policies.json')
    for pid in policies:
        target=RUN/f'results/{split}/{pid}/prefill_decode.json'
        if target.exists():continue
        lock=RUN/f'results/locks/{split}_{pid}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if target.exists():continue
            partial=RUN/f'results/captures/{split}/{pid}/prefill_decode'
            if partial.exists():partial.rename(partial.with_name('prefill_decode_failed_'+str(time.time_ns())))
            invoke(pid,split,gpu)


def queue(split,cards):
    children=[]
    for gpu in cards:
        handle=(RUN/f'logs/queue_{split}_gpu{gpu}_{time.time_ns()}.log').open('w')
        p=subprocess.Popen([sys.executable,__file__,'--worker',split,'--gpu',str(gpu)],stdout=handle,stderr=subprocess.STDOUT)
        children.append((p,handle))
    while any(p.poll() is None for p,_ in children):
        write(RUN/'results/worker.json',{'status':'running','stage':split,'gpus':cards,
            'complete':len(list((RUN/f'results/{split}').glob('*/prefill_decode.json')))})
        time.sleep(10)
    codes=[p.returncode for p,_ in children]
    for _,handle in children:handle.close()
    assert all(code==0 for code in codes),codes


def delta(path,base):
    a=read(path);b=read(base);assert a['data_sha256']==b['data_sha256']
    values=[x['avg_nll']-y['avg_nll'] for x,y in zip(a['blocks'],b['blocks'],strict=True)]
    assert all(math.isfinite(x) for x in values)
    return values


def select(count):
    design=read(RUN/'configs/design.json');candidates=[]
    for pid in design['policies']:
        values=delta(RUN/f'reference/validation/{pid}.json',RUN/'reference/validation/uniform_dense_bf16.json')
        if count==64:values+=delta(RUN/f'results/validation/{pid}/prefill_decode.json',RUN/'results/validation/uniform_dense_bf16/prefill_decode.json')
        assert len(values)==count
        mean=statistics.mean(values);se=statistics.stdev(values)/len(values)**.5
        raw=read(RUN/f'reference/calibration/{pid}.json')
        latency=next(c['medians']['e2e_ms'] for c in raw['cases'] if c['output']==64)
        candidates.append({'policy':pid,'validation_blocks':count,'mean':mean,'se':se,
            'guard':0. if pid=='uniform_dense_bf16' else mean+2*se+.005,'calibration_e2e_ms':latency})
    choices=[]
    for budget in design['budgets']:
        feasible=[c for c in candidates if c['guard']<=budget]
        choice=min(feasible,key=lambda c:(c['calibration_e2e_ms'],c['policy']))
        choices.append({**choice,'quality_budget':budget,'feasible_candidates':len(feasible)})
    return choices,candidates


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.worker:worker(args.worker,args.gpu);return
    design=read(RUN/'configs/design.json')
    assert read(SOURCE/'results/final_audit.json')['verified']
    assert sha(SOURCE/'results/manifest.json')==design['source_manifest_sha256']
    for pid in design['policies']:
        assert sha(SOURCE/f'configs/policies/{pid}.json')==design['policy_sha256'][pid]
        assert sha(RUN/f'reference/validation/{pid}.json')==design['old_validation_sha256'][pid]
        assert sha(RUN/f'reference/calibration/{pid}.json')==design['calibration_sha256'][pid]
    for name,digest in design['source_code_sha256'].items():assert sha(SOURCE/'code'/name)==digest
    original,_=select(32)
    for row in original:
        assert row['policy']==next(c['policy'] for c in design['old_selections'] if c['quality_budget']==row['quality_budget'])
    sys.path.insert(0,str(SOURCE/'code'))
    from resources import quality_gpus
    write(RUN/'results/worker.json',{'status':'waiting','stage':'free_quality_cards'})
    with quality_gpus(maximum=2,minimum=1) as cards:
        queue('validation',cards)
        if (RUN/'configs/selected_policies.json').exists():selected=read(RUN/'configs/selected_policies.json')['selected']
        else:
            assert not list((RUN/'results/confirmation').glob('*/*.json'))
            selected,candidates=select(64)
            write(RUN/'results/validation_candidates.json',candidates)
            write(RUN/'configs/selected_policies.json',{'selected':selected,'old32':original,
                'written_at_unix':time.time(),'confirmation_read':False})
        write(RUN/'configs/confirmation_policies.json',sorted({'uniform_dense_bf16',*[r['policy'] for r in original+selected]}))
        queue('confirmation',cards)
    sys.path.insert(0,str(RUN.parents[1]/'night151/code'))
    from publish_budget_table import core_speed
    full={}
    for r in csv.DictReader((SOURCE/'results/full_task_metrics.csv').open()):
        if r['scenario']=='prefill_decode':full[r['policy'],r['metric']]=float(r['value'])
    rows=[]
    for choice in original+selected:
        pid=choice['policy']
        values=delta(RUN/f'results/confirmation/{pid}/prefill_decode.json',RUN/'results/confirmation/uniform_dense_bf16/prefill_decode.json')
        assert len(values)==64
        mean=statistics.mean(values);se=statistics.stdev(values)/8
        rows.append({**choice,'fresh_confirmation_delta_nll':mean,'fresh_confirmation_two_se':2*se,
            'mean_budget_violation':mean>choice['quality_budget'],'inherited_e2e_speedup':core_speed('018',pid,'prefill_decode'),
            **{metric:full[pid,metric] for metric in ['cnn_dm','dialogsum','iwslt']}})
    with (RUN/'results/validation_size.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Validation窗口数量敏感性：Llama-2 P+D（R1 AK6a / R5 gkP9）','',
        '14个预先固定的主表内DASO可选incumbent，32窗口选择先严格复现原结果。新增32个不重叠validation窗口，'
        '仍按mean+2SE+.005选择，BF16保护值0。两种选择在同64个新确认窗口比较，确认窗口与018原窗口不重叠；'
        '选择在打开新确认之前冻结。原主表及原完整候选池结果保持不变。','',
        '| Val windows | ε | Policy | Val guard | Fresh test ΔNLL ±2SE | E2E × | CNN/DM | DialogSum | IWSLT | Mean exceeds ε |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        lines.append(f'| {r["validation_blocks"]} | {r["quality_budget"]:.2f} | {r["policy"]} | {r["guard"]:.5f} | '
            f'{r["fresh_confirmation_delta_nll"]:.5f} ± {r["fresh_confirmation_two_se"]:.5f} | {r["inherited_e2e_speedup"]:.3f} | '
            f'{r["cnn_dm"]:.2f} | {r["dialogsum"]:.2f} | {r["iwslt"]:.2f} | {r["mean_budget_violation"]} |')
    lines+=['','速度和三项完整生成复用已审计018相同有效map的结果，不是新增独立下游测试；本实验只新增NLL窗口。'
        '这是有界incumbent池的样本量跟进研究，不是重新优化完整候选空间，不能替换原主表或据此宣称普遍最优。', '',
        '[完整CSV](results/validation_size.csv) · [冻结设计](configs/design.json) · [新窗口](configs/frozen_design.json)','']
    (RUN/'REPORT.md').write_text('\n'.join(lines))
    assert sha(SOURCE/'results/manifest.json')==design['source_manifest_sha256']
    write(RUN/'results/final_audit.json',{'complete':True,'verified':True,'fixed_incumbents':14,'rows':len(rows),
        'source_manifest_sha256':design['source_manifest_sha256'],'design_sha256':sha(RUN/'configs/design.json'),
        'code_sha256':sha(Path(__file__)),'csv_sha256':sha(RUN/'results/validation_size.csv'),
        'scope':'New32 validation and fresh64 confirmation; speed/downstream inherited, bounded pool, old main unchanged.'})
    write(RUN/'results/worker.json',{'status':'complete','stage':'validation_size_audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/'results/worker.json',{'status':'failed','error':repr(error)});raise
