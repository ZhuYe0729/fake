"""Full SQ-FP8 quality plus matched references; original numerical code intact."""
import argparse
import fcntl
import hashlib
import os
import random
import shutil
import subprocess
import sys
import time

from launch import RUN,REPO,MODEL,SPEC,TEMP,WRAPPER,checkpoint,invoke,read,write


def state(stage,**details):write(RUN/'results/worker.json',{'status':'running','stage':stage,**details})


def worker():
    from baseline_resources import lease
    for job in read(RUN/'configs/design.json')['jobs']:
        label=job['label'];pid=job['policy'];marker=RUN/f'results/completed/{label}.json';lock=RUN/f'results/locks/{label}.lock'
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a')as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            with lease()as gpu:
                model=checkpoint(pid,gpu)
                if job['stage']=='mc':
                    manifest=RUN/'configs/manifest_sqfp8.json';policy=RUN/f'configs/policies/{pid}.json'
                    write(manifest,{'model_path':str(MODEL),'policies':[{'label':pid,'kind':'ours','checkpoint':str(model),
                        'policy_json':str(policy),'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest()}]})
                    argv=[sys.executable,RUN/'code/run_workflow.py','tasks','--manifest',manifest,'--policy',pid,
                        '--output-root',RUN/'results/tasks','--vllm-root',REPO.parent/'test/vllm','--cutlass-wrapper-path',WRAPPER,'--audit']
                elif job['stage']=='nll':
                    argv=[sys.executable,RUN/'code/score_nll.py','--checkpoint',model,'--policy',pid,'--split','confirmation',
                        '--scenarios','prefill_only','prefill_decode']
                else:argv=[sys.executable,RUN/'code/generate_tasks.py','--checkpoint',model,'--policy',pid,'--dataset',job['dataset']]
                invoke(argv,label,gpu);write(marker,{'complete':True,'gpu':gpu,**job})


def seed():
    cache=RUN.parents[1]/'night161/checkpoints'/read(RUN/'configs/design.json')['source_current']
    with(RUN.parents[1]/'night151/export.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        for pid in read(RUN/'configs/design.json')['anchors']:
            src=cache/pid;dst=TEMP/'checkpoints'/pid
            assert read(src/'config.json')['quantization_config']['method_map']==read(RUN/f'configs/policies/{pid}.json')['method_map']
            if(dst/'model.safetensors').exists():continue
            dst.mkdir(parents=True,exist_ok=True)
            for path in src.iterdir():
                if path.is_file()and path.name!='model.safetensors':shutil.copy2(path,dst/path.name)
            os.link(src/'model.safetensors',dst/'model.safetensors')
            write(RUN/f'results/checkpoint_reuse/{pid}.json',{'exact_hardlink':True,'source':str(src),'physical_copy_bytes':0})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:return worker()
    with(RUN/'results/driver.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not(RUN.parent/'037/results/final_audit.json').exists():state('wait037');time.sleep(10)
        replay=read(RUN.parent/'037/results/final_audit.json');assert replay['all_token_sequences_exact']and replay['all_nll_within_original_gate']
        seed()
        if not(RUN/'results/quality_complete.json').exists():
            children=[]
            for i in range(3):
                log=(RUN/f'logs/worker{i}_{time.time_ns()}.log').open('w')
                children.append((subprocess.Popen([sys.executable,__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT),log))
            while any(p.poll()is None for p,_ in children):
                state('quality',complete=len(list((RUN/'results/completed').glob('*.json'))),total=5);time.sleep(10)
            assert all(p.returncode==0 for p,_ in children)
            for _,log in children:log.close()
            assert len(list((RUN/'results/completed').glob('*.json')))==5
            write(RUN/'results/quality_complete.json',{'complete':True})
        while not(RUN.parent/'041/results/final_audit.json').exists():state('wait041');time.sleep(10)
        if RUN.name=='043':
            while not(RUN.parent/'042/results/final_audit.json').exists():state('wait042');time.sleep(10)
        from baseline_resources import lease
        rounds=RUN/'configs/baseline_formal_rounds.json'
        if not rounds.exists():
            orders=[]
            for ri in range(3):
                middle=[p for p in read(RUN/'configs/design.json')['policies']if p!='uniform_dense_bf16'];random.Random(167042+ri).shuffle(middle)
                orders.append(['uniform_dense_bf16',*middle,'uniform_dense_bf16_bracket_end'])
            write(rounds,orders)
        for ri,order in enumerate(read(rounds)):
            for index,label in enumerate(order):
                tag=f'baseline_round{ri}_{label}'
                if(RUN/f'results/measurements/{tag}.json').exists():continue
                pid='uniform_dense_bf16'if label.endswith('_bracket_end')else label
                state('formal_speed',round=ri,index=index,total=len(order),policy=pid)
                with lease(exclusive=True)as gpu:
                    model=checkpoint(pid,gpu)
                    invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',pid,'--mode','arena_graph',
                        '--label',tag,'--outputs','1','64','--iterations','5','--audit-weights','--input-file',RUN/'configs/speed_inputs.json'],tag,gpu)
        for path in(TEMP/'checkpoints').glob('*/model.safetensors'):
            write(RUN/f'results/cleanup/{path.parent.name}.json',{'path':str(path),'bytes':path.stat().st_size,'reason':'Own reproducible export/hardlink, all quality and formal jobs done; retain metadata/code/logs/results.'});path.unlink()
        state('report')
        with(RUN.parents[1]/'night151/measurement.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            subprocess.run([sys.executable,RUN/'code/baseline_report.py'],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        write(RUN/'results/worker.json',{'status':'complete','stage':'audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)});raise
