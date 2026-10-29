"""Time all frozen matched maps after the fixed-load transfer campaign."""
import fcntl
import os
import shutil
import subprocess
import sys
import time

from launch import RUN,TEMP,checkpoint,invoke,read,write


def state(stage,**details):write(RUN/'results/worker.json',{'status':'running','stage':stage,**details})


def main():
    with(RUN/'results/driver.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not(RUN.parent/'040/results/final_audit.json').exists():state('wait040');time.sleep(10)
        assert read(RUN.parent/'040/results/final_audit.json')['verified']
        cache=RUN.parents[1]/'night161/checkpoints/034/uniform_dense_bf16'
        target=TEMP/'checkpoints/uniform_dense_bf16'
        if not(target/'model.safetensors').exists():
            assert read(cache/'config.json')['quantization_config']['method_map']==read(RUN/'configs/policies/uniform_dense_bf16.json')['method_map']
            target.mkdir(parents=True,exist_ok=True)
            for path in cache.iterdir():
                if path.is_file()and path.name!='model.safetensors':shutil.copy2(path,target/path.name)
            os.link(cache/'model.safetensors',target/'model.safetensors')
            write(RUN/'results/bf16_cache_reuse.json',{'exact_hardlink':True,'source':str(cache),'physical_copy_bytes':0})
        design=read(RUN/'configs/design.json');scenes={p['policy_id']:p['scenario']for p in design['points']}
        from resources import lease
        for ri,order in enumerate(design['rounds']):
            for index,label in enumerate(order):
                tag=f'matched_round{ri}_{label}';raw=RUN/f'results/measurements/{tag}.json'
                if raw.exists():continue
                pid='uniform_dense_bf16'if label.endswith('_bracket_end')else label
                outputs=['1','64']if pid=='uniform_dense_bf16'else['1'if scenes[pid]=='prefill_only'else'64']
                state('formal_speed',round=ri,index=index,total=len(order),policy=pid)
                with lease(exclusive=True)as gpu:
                    model=checkpoint(pid,gpu)
                    invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',pid,'--mode','arena_graph',
                        '--label',tag,'--outputs',*outputs,'--iterations','8','--audit-weights','--input-file',RUN/'configs/speed_inputs.json'],tag,gpu)
                    if pid!='uniform_dense_bf16':
                        path=model/'model.safetensors'
                        if ri<2 and shutil.disk_usage(RUN).free>=96*1024**3:
                            write(RUN/f'results/cache_reuse/{tag}.json',{'path':str(path),'bytes':path.stat().st_size,
                                'inode':path.stat().st_ino,'reason':'Retain identical own export for remaining rounds; keep at least96GiB free.'})
                        else:
                            write(RUN/f'results/cleanup/{tag}.json',{'path':str(path),'bytes':path.stat().st_size,'reason':'Own reproducible export, this round measured; retain all metadata/code/logs/results.'});path.unlink()
        write(RUN/'results/formal_complete.json',{'complete':True,'rounds':3})
        path=target/'model.safetensors'
        if path.exists():path.unlink()
        state('report')
        with(RUN.parents[1]/'night151/measurement.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            subprocess.run([sys.executable,RUN/'code/matched_report.py'],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        write(RUN/'results/worker.json',{'status':'complete','stage':'audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)});raise
